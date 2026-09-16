#!/usr/bin/env python3
"""
validate_afterburner.py   (v2)
==============================

Did the EIC afterburner transform my event record, and was it a legitimate
LORENTZ transformation?

WHY v2 EXISTS  (read this, it is the whole point)
-------------------------------------------------
v1 fitted a 3x3 matrix to the 3-MOMENTA of each event, on the assumption
that the crossing angle is applied as a pure rotation.  IT IS NOT.

The ePIC afterburner implements the crossing angle the way the accelerator
does: a BOOST by half the crossing angle along x, composed with a rotation.
The net effect leaves the electron beam along -z and gives the hadron beam
the full ~25 mrad tilt toward -x.  Because there is a boost in there:

    * particle ENERGIES change        (v1's check A4 "failed" -- correctly
                                       observing a boost, wrongly calling
                                       it a problem)
    * no 3x3 momentum-only matrix can describe the mapping, because
      energy mixes into momentum   (v1's residual "failure")

v1 had a second, worse bug: with 5-6 particles per event, most of them
nearly collinear with z, the per-event 3x3 system is badly ILL-CONDITIONED.
The x-y block is essentially unconstrained, so the solver returns numerical
garbage -- which is why v1 reported det(M) ranging from -65 to +12.

v2 fixes both:

    * fits a single global 4x4 matrix  Lambda,  p_after = Lambda . p_before,
      POOLING particles from many events, so the system is massively
      overdetermined and well conditioned
    * tests the correct invariance condition for a Lorentz transformation,
        Lambda^T g Lambda = g          with g = diag(-1,-1,-1,+1)
      rather than the orthogonality condition M^T M = I, which only holds
      for pure rotations
    * reports the CONDITION NUMBER, so this failure mode can never again
      be mistaken for physics
    * polar-decomposes  Lambda = B(beta) . R  and reports the boost
      velocity and the rotation axis/angle separately

WHAT STILL MATTERS
------------------
    det Lambda = +1     -> proper: not a reflection.  A reflection would
                           silently invert every spin asymmetry.
    Lambda^0_0 > 0      -> orthochronous: does not reverse time.
    Lambda^T g Lambda=g -> it is in the Lorentz group at all: not a shear,
                           not a scaling, not an energy-spread smear.
    residual ~ 1e-12    -> ONE matrix describes EVERY particle.  If the
                           afterburner rotated the beams but forgot a
                           spectator, or if per-particle beam divergence
                           is enabled, this blows up.
    invariants, weights -> unchanged, always.

USAGE
-----
    python3 validate_afterburner.py AFTER.root --before BEFORE.root
    python3 validate_afterburner.py AFTER.root --before BEFORE.root \
            --nmax 20000 --nfit 2000 --expect-angle 25.0

    # no reference file available
    python3 validate_afterburner.py AFTER.root

LAYOUT NOTE
-----------
    4-vectors are stored as m_v1..m_v4 = (px, py, pz, E), so the metric is
    g = diag(-1, -1, -1, +1), NOT the diag(+1,-1,-1,-1) you may be used to.
"""

import argparse
import math
import sys

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)

# metric for the (px, py, pz, E) ordering used in this file format
G = np.diag([-1.0, -1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def ok(flag):
    return "\033[92mPASS\033[0m" if flag else "\033[91mFAIL\033[0m"


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------
class Reader:
    """Per-event reader for the HepMC3-in-ROOT layout."""

    LEAVES = ("particles.pid", "particles.status", "particles.mass",
              "particles.momentum.m_v1", "particles.momentum.m_v2",
              "particles.momentum.m_v3", "particles.momentum.m_v4",
              "event_pos.m_v1", "event_pos.m_v2",
              "event_pos.m_v3", "event_pos.m_v4",
              "event_number")

    def __init__(self, path):
        self.path = path
        self.f = ROOT.TFile.Open(path)
        if not self.f or self.f.IsZombie():
            sys.exit(f"ERROR: cannot open {path}")
        self.t = self.f.Get("hepmc3_tree")       # highest cycle
        if not self.t:
            sys.exit(f"ERROR: no hepmc3_tree in {path}")
        self.n = self.t.GetEntries()
        self.L = {}
        for name in self.LEAVES:
            leaf = self.t.GetLeaf(name)
            if not leaf:
                sys.exit(f"ERROR: missing leaf {name} in {path}")
            self.L[name] = leaf

    def event(self, i):
        self.t.GetEntry(i)
        npart = self.L["particles.pid"].GetLen()

        pid = np.fromiter((self.L["particles.pid"].GetValue(j)
                           for j in range(npart)), np.int64, npart)
        st = np.fromiter((self.L["particles.status"].GetValue(j)
                          for j in range(npart)), np.int64, npart)
        mass = np.fromiter((self.L["particles.mass"].GetValue(j)
                            for j in range(npart)), np.float64, npart)

        p = np.empty((npart, 4))
        for k, nm in enumerate(("particles.momentum.m_v1",
                                "particles.momentum.m_v2",
                                "particles.momentum.m_v3",
                                "particles.momentum.m_v4")):
            L = self.L[nm]
            for j in range(npart):
                p[j, k] = L.GetValue(j)

        pos = np.array([self.L[f"event_pos.m_v{k}"].GetValue(0)
                        for k in (1, 2, 3, 4)])
        try:
            w = list(self.t.weights)
        except Exception:
            w = []

        return dict(pid=pid, st=st, mass=mass, p=p, pos=pos, w=w,
                    evtnum=int(self.L["event_number"].GetValue(0)))


# ---------------------------------------------------------------------------
# physics helpers
# ---------------------------------------------------------------------------
def m2(v):
    return v[3] ** 2 - v[0] ** 2 - v[1] ** 2 - v[2] ** 2


def find(pid, st, want_pid, want_st):
    for i in range(len(pid)):
        if (want_pid == 0 or pid[i] == want_pid) and st[i] == want_st:
            return i
    return -1


def find_ion(pid, st):
    for i in range(len(pid)):
        if st[i] == 4 and pid[i] != 11:
            return i
    return -1


def invariants(ev):
    """{Q2, xB, y, W2, s} from the written 4-vectors, per nucleon."""
    pid, st, p = ev["pid"], ev["st"], ev["p"]
    a, b, c = find(pid, st, 11, 4), find(pid, st, 11, 1), find_ion(pid, st)
    if a < 0 or b < 0 or c < 0:
        return None
    k_in, k_out, Pnuc = p[a], p[b], p[c]

    nuc = int(pid[c])
    A = float((nuc // 10) % 1000) if nuc > 1000000000 else 1.0
    P = Pnuc / A

    q = k_in - k_out
    Q2 = -m2(q)
    Pq = P[3] * q[3] - P[:3] @ q[:3]
    Pk = P[3] * k_in[3] - P[:3] @ k_in[:3]
    if Pq == 0 or Pk == 0:
        return None
    return dict(Q2=Q2, xB=Q2 / (2 * Pq), y=Pq / Pk,
                W2=m2(P + q), s=m2(P + k_in))


def boost_matrix(beta):
    """Pure boost with velocity beta (3-vector), in (px,py,pz,E) layout."""
    b2 = float(beta @ beta)
    if b2 < 1e-30:
        return np.eye(4)
    g = 1.0 / math.sqrt(1.0 - b2)
    B = np.eye(4)
    B[:3, :3] = np.eye(3) + (g - 1.0) * np.outer(beta, beta) / b2
    B[:3, 3] = g * beta
    B[3, :3] = g * beta
    B[3, 3] = g
    return B


def rot_axis_angle(R):
    """Axis and angle of a 3x3 rotation."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    ax = ax / n if n > 1e-14 else np.array([0.0, 0.0, 0.0])
    return ax, ang


# ---------------------------------------------------------------------------
# MODE A : before / after
# ---------------------------------------------------------------------------
def mode_a(after_path, before_path, args):
    Rb, Ra = Reader(before_path), Reader(after_path)
    nmax = min(Rb.n, Ra.n, args.nmax) if args.nmax > 0 else min(Rb.n, Ra.n)
    nfit = min(args.nfit, nmax)

    print(f"\nMODE A: before/after comparison")
    print(f"  before : {before_path}   ({Rb.n:,} events)")
    print(f"  after  : {after_path}   ({Ra.n:,} events)")
    print(f"  comparing {nmax:,} events; global Lorentz fit on {nfit:,}")

    # ---------------- collect ----------------
    bad_num = bad_np = bad_pdg = 0
    Xb, Xa = [], []            # pooled 4-vectors for the global fit
    ev_slices = []             # (start, stop) into the pooled arrays
    dE = dm2 = 0.0
    inv_b, inv_a, inv_rel = [], [], []
    wsum_b = wsum_a = 0.0
    wdiff = 0
    vtx = []
    ion_px, ion_ang, ele_ang = [], [], []
    ev0 = None

    row = 0
    for i in range(nmax):
        eb, ea = Rb.event(i), Ra.event(i)

        if eb["evtnum"] != ea["evtnum"]:
            bad_num += 1
            continue
        if len(eb["pid"]) != len(ea["pid"]):
            bad_np += 1
            continue
        if not np.array_equal(eb["pid"], ea["pid"]):
            bad_pdg += 1
            continue

        if i == 0:
            ev0 = (eb, ea)

        if i < nfit:
            n = len(eb["pid"])
            Xb.append(eb["p"])
            Xa.append(ea["p"])
            ev_slices.append((row, row + n))
            row += n

        dE = max(dE, np.abs(ea["p"][:, 3] - eb["p"][:, 3]).max())
        dm2 = max(dm2, max(abs(m2(ea["p"][j]) - m2(eb["p"][j]))
                           for j in range(len(eb["pid"]))))

        ib, ia = invariants(eb), invariants(ea)
        if ib and ia:
            inv_b.append(ib)
            inv_a.append(ia)
            inv_rel.append(max(abs(ia[k] - ib[k]) / max(abs(ib[k]), 1e-30)
                               for k in ib))

        wb, wa = eb["w"], ea["w"]
        if len(wb) and len(wa):
            wsum_b += wb[0]
            wsum_a += wa[0]
            if len(wb) != len(wa) or any(
                    abs(x - y) > 1e-12 * max(abs(x), 1.0)
                    for x, y in zip(wb, wa)):
                wdiff += 1

        vtx.append(ea["pos"][:3])

        j = find_ion(ea["pid"], ea["st"])
        k = find(ea["pid"], ea["st"], 11, 4)
        if j >= 0:
            ion_px.append(ea["p"][j, 0])
            ion_ang.append(math.atan2(ea["p"][j, 0], ea["p"][j, 2]))
        if k >= 0:
            ele_ang.append(math.atan2(ea["p"][k, 0], -ea["p"][k, 2]))

    Xb = np.vstack(Xb)
    Xa = np.vstack(Xa)

    # ---------------- A1 ----------------
    section("A1. EVENT MATCHING")
    print(f"  particles pooled for the fit      : {len(Xb):,}")
    print(f"  event_number mismatches           : {bad_num}   {ok(bad_num == 0)}")
    print(f"  particle-count mismatches         : {bad_np}   {ok(bad_np == 0)}")
    print(f"  PDG-sequence mismatches           : {bad_pdg}   {ok(bad_pdg == 0)}")

    # ---------------- A2 : global 4x4 fit ----------------
    section("A2. GLOBAL 4x4 LORENTZ FIT   (the decisive test)")
    print("  Solving  p_after = Lambda . p_before  over ALL pooled particles.")
    print("  A single 4x4 matrix must describe every particle in every event.\n")

    cond = np.linalg.cond(Xb)
    sol, *_ = np.linalg.lstsq(Xb, Xa, rcond=None)
    Lam = sol.T
    resid = np.abs(Xb @ Lam.T - Xa).max()
    scale = np.abs(Xa).max()

    defect = np.abs(Lam.T @ G @ Lam - G).max()
    det = np.linalg.det(Lam)
    ortho = Lam[3, 3]

    good_cond = cond < 1e6
    good_def = defect < args.tol
    good_det = abs(det - 1.0) < 1e-9
    good_time = ortho > 0
    good_res = resid < args.rtol * max(scale, 1.0)

    print(f"  condition number of the fit       : {cond:.3e}   {ok(good_cond)}")
    if not good_cond:
        print("      Ill-conditioned. Increase --nfit; the particles in")
        print("      this sample are too collinear to constrain Lambda.")
    print(f"  max|Lambda^T g Lambda - g|        : {defect:.3e}   {ok(good_def)}")
    print(f"      (must be < {args.tol:.0e}: this is membership in the Lorentz group)")
    print(f"  det Lambda                        : {det:+.12f}   {ok(good_det)}")
    print("      (must be EXACTLY +1; -1 is a reflection and flips spin asymmetries)")
    print(f"  Lambda^0_0                        : {ortho:+.9f}   {ok(good_time)}")
    print("      (must be > 0: orthochronous, does not reverse time)")
    print(f"  max residual |Lambda.before-after|: {resid:.3e} GeV   {ok(good_res)}")
    print(f"      (relative to max|p| = {scale:.3g} GeV)")
    if not good_res:
        print("      One matrix does NOT fit every particle. Either a")
        print("      particle was missed by the transform, or per-particle")
        print("      beam divergence / energy spread is enabled. See A4.")

    print("\n  Lambda =")
    for r in range(4):
        print("      [" + "  ".join(f"{Lam[r, c]:+.9f}" for c in range(4)) + "]")

    # ---------------- A3 : decomposition ----------------
    section("A3. DECOMPOSITION   Lambda = B(beta) . R")
    u = Lam @ np.array([0.0, 0.0, 0.0, 1.0])   # image of a particle at rest
    beta = u[:3] / u[3]
    gam = u[3]
    Rfull = boost_matrix(-beta) @ Lam
    R3 = Rfull[:3, :3]
    mix = max(np.abs(Rfull[3, :3]).max(), np.abs(Rfull[:3, 3]).max())
    axis, ang = rot_axis_angle(R3)

    print(f"  boost velocity beta   : ({beta[0]:+.8f}, {beta[1]:+.8f}, {beta[2]:+.8f})")
    print(f"  |beta|                : {np.linalg.norm(beta):.8f}    gamma = {gam:.9f}")
    print(f"  implied half-angle    : {math.asin(min(1.0, np.linalg.norm(beta)))*1e3:.5f} mrad")
    print(f"  residual boost/rot mixing after removing B : {mix:.3e}")
    print(f"  rotation angle        : {ang*1e3:.5f} mrad")
    print(f"  rotation axis         : ({axis[0]:+.6f}, {axis[1]:+.6f}, {axis[2]:+.6f})")
    dom = int(np.argmax(np.abs(axis)))
    print(f"  dominant axis         : {'xyz'[dom]}   {ok(dom == 1)}"
          "   (expect y for an x-z crossing)")
    print("\n  The ePIC afterburner applies a boost by HALF the crossing angle")
    print("  plus a rotation, so that the electron stays along -z and the")
    print("  hadron beam takes the full angle. A nonzero boost here is CORRECT.")

    # ---------------- A4 : divergence check ----------------
    section("A4. IS PER-PARTICLE SMEARING ENABLED?")
    print("  Refit Lambda event by event and look at the spread. If beam")
    print("  divergence or energy spread is on, each event gets its own")
    print("  slightly different transform.\n")
    dets, defs = [], []
    for (s0, s1) in ev_slices[:min(len(ev_slices), 500)]:
        b, a = Xb[s0:s1], Xa[s0:s1]
        if s1 - s0 < 5:
            continue
        sol_i, *_ = np.linalg.lstsq(b, a, rcond=None)
        Li = sol_i.T
        dets.append(np.linalg.det(Li))
        defs.append(np.abs(Li - Lam).max())
    if dets:
        spread = float(np.max(defs))
        print(f"  max|Lambda_event - Lambda_global| : {spread:.3e}")
        print(f"  per-event det range               : "
              f"[{min(dets):+.6f}, {max(dets):+.6f}]")
        print("  NOTE: per-event fits use only 5-6 particles and are poorly")
        print("        conditioned; treat large scatter here as inconclusive,")
        print("        not as evidence of a problem. The global fit is the test.")
    print(f"\n  max |dE| per particle   : {dE:.6e} GeV")
    print(f"  max |dm^2| per particle : {dm2:.6e} GeV^2   {ok(dm2 < 1e-6)}")
    if dm2 < 1e-6 and dE > 1e-6:
        print("  Mass preserved, energy moved -> a BOOST. Expected here.")
    if dm2 >= 1e-6:
        print("  Mass NOT preserved -> this is not a Lorentz transformation.")

    # ---------------- A5 : invariants and weights ----------------
    section("A5. LORENTZ INVARIANTS AND WEIGHTS")
    print("  Nothing here may move, whatever the afterburner did.\n")
    print(f"    {'quantity':>10} {'<before>':>16} {'<after>':>16} {'max rel diff':>14}")
    print("    " + "-" * 60)
    worst = 0.0
    for k in ("Q2", "xB", "y", "W2", "s"):
        mb = np.mean([d[k] for d in inv_b])
        ma = np.mean([d[k] for d in inv_a])
        rd = max(abs(a[k] - b[k]) / max(abs(b[k]), 1e-30)
                 for a, b in zip(inv_a, inv_b))
        worst = max(worst, rd)
        print(f"    {k:>10} {mb:16.9g} {ma:16.9g} {rd:14.3e}")
    print(f"\n  all invariants within 1e-09 : {ok(worst < 1e-9)}")

    rel_w = abs(wsum_a - wsum_b) / max(abs(wsum_b), 1e-30)
    print(f"\n  sum(w) before : {wsum_b:.10e}")
    print(f"  sum(w) after  : {wsum_a:.10e}")
    print(f"  relative diff : {rel_w:.3e}   {ok(rel_w < 1e-12)}")
    print(f"  events whose weight vector changed : {wdiff}   {ok(wdiff == 0)}")

    # ---------------- A6 : beam geometry and vertex ----------------
    section("A6. BEAM GEOMETRY AND VERTEX SMEARING (after)")
    if ion_px:
        mpx = float(np.mean(ion_px))
        mang = float(np.mean(ion_ang)) * 1e3
        print(f"  <ion beam px>       : {mpx:+.6f} GeV   "
              f"{ok(mpx < 0)}   (ePIC: must be NEGATIVE, toward -x)")
        print(f"  <ion beam angle>    : {mang:+.5f} mrad")
    if ele_ang:
        eang = float(np.mean(ele_ang)) * 1e3
        print(f"  <e- beam angle>     : {eang:+.5f} mrad")
        print("      (the afterburner keeps the electron essentially along -z)")
    if ion_ang and ele_ang:
        tot = abs(float(np.mean(ion_ang)) - float(np.mean(ele_ang))) * 1e3
        good_ang = abs(tot - args.expect_angle) < args.angle_tol
        print(f"  beam-beam crossing  : {tot:.5f} mrad   {ok(good_ang)}"
              f"   (expect {args.expect_angle})")

    V = np.array(vtx)
    print(f"\n  vertex mean (mm)    : ({V[:,0].mean():+.6f}, "
          f"{V[:,1].mean():+.6f}, {V[:,2].mean():+.6f})")
    print(f"  vertex rms  (mm)    : ({V[:,0].std():.6f}, "
          f"{V[:,1].std():.6f}, {V[:,2].std():.6f})")
    smeared = V.std(axis=0).max() > 1e-9
    print(f"  vertex smearing     : {'ON' if smeared else 'OFF'}"
          "   (informational, not a pass/fail)")

    # ---------------- event 0 dump ----------------
    if ev0:
        section("A7. EVENT 0, SIDE BY SIDE")
        eb, ea = ev0
        print(f"    {'#':>3} {'PDG':>11} {'st':>3} |{'px_before':>11}{'px_after':>12}"
              f" |{'pz_before':>13}{'pz_after':>13} |{'E_before':>12}{'E_after':>12}")
        print("    " + "-" * 97)
        for j in range(len(eb["pid"])):
            print(f"    {j:3d} {eb['pid'][j]:11d} {eb['st'][j]:3d} |"
                  f"{eb['p'][j,0]:11.5f}{ea['p'][j,0]:12.5f} |"
                  f"{eb['p'][j,2]:13.5f}{ea['p'][j,2]:13.5f} |"
                  f"{eb['p'][j,3]:12.5f}{ea['p'][j,3]:12.5f}")
        print(f"\n    vertex before : {eb['pos'][:3]}")
        print(f"    vertex after  : {ea['pos'][:3]}")

    # ---------------- verdict ----------------
    section("VERDICT")
    checks = [
        ("events matched cleanly", bad_num == 0 and bad_np == 0 and bad_pdg == 0),
        ("fit is well conditioned", good_cond),
        ("Lambda is in the Lorentz group", good_def),
        ("det Lambda = +1 (proper, not a reflection)", good_det),
        ("Lambda is orthochronous", good_time),
        ("one matrix fits every particle", good_res),
        ("rotation about the y-axis", dom == 1),
        ("particle masses preserved", dm2 < 1e-6),
        ("Lorentz invariants preserved", worst < 1e-9),
        ("weights untouched", rel_w < 1e-12 and wdiff == 0),
    ]
    for name, flag in checks:
        print(f"  {ok(flag)}  {name}")

    fails = [n for n, f in checks if not f]
    if not fails:
        print("\n  \033[92mThe afterburner applied a proper, orthochronous Lorentz")
        print("  transformation to every particle. Invariants and weights are")
        print("  untouched. The file is good.\033[0m")
    else:
        print("\n  Not everything passed:")
        for n in fails:
            print(f"    - {n}")
        print("\n  Triage:")
        print("    det/orthochronous/invariants FAIL -> a real bug.")
        print("    'one matrix fits' FAIL alone      -> per-particle divergence")
        print("                                         or energy spread is on.")
        print("    'well conditioned' FAIL           -> raise --nfit, rerun.")


# ---------------------------------------------------------------------------
# MODE B : standalone
# ---------------------------------------------------------------------------
def mode_b(path, args):
    R = Reader(path)
    nmax = min(R.n, args.nmax) if args.nmax > 0 else R.n
    print(f"\nMODE B: standalone check of {path}  ({nmax:,} of {R.n:,} events)")

    ion_ang, ele_ang, ion_px = [], [], []
    vtx, Ebeam_i, Ebeam_e = [], [], []
    for i in range(nmax):
        ev = R.event(i)
        j = find_ion(ev["pid"], ev["st"])
        k = find(ev["pid"], ev["st"], 11, 4)
        if j >= 0:
            ion_px.append(ev["p"][j, 0])
            ion_ang.append(math.atan2(ev["p"][j, 0], ev["p"][j, 2]))
            Ebeam_i.append(ev["p"][j, 3])
        if k >= 0:
            ele_ang.append(math.atan2(ev["p"][k, 0], -ev["p"][k, 2]))
            Ebeam_e.append(ev["p"][k, 3])
        vtx.append(ev["pos"][:3])

    section("B1. CROSSING ANGLE")
    mpx = float(np.mean(ion_px))
    print(f"  <ion beam px>    : {mpx:+.6f} GeV   {ok(mpx < 0)}   (ePIC: negative)")
    print(f"  <ion angle>      : {np.mean(ion_ang)*1e3:+.5f} mrad")
    print(f"  <e- angle>       : {np.mean(ele_ang)*1e3:+.5f} mrad")
    tot = abs(np.mean(ion_ang) - np.mean(ele_ang)) * 1e3
    print(f"  beam-beam        : {tot:.5f} mrad   "
          f"{ok(abs(tot - args.expect_angle) < args.angle_tol)}")

    section("B2. BEAM ENERGY SPREAD")
    print(f"  ion  E : {np.mean(Ebeam_i):.6f} +/- {np.std(Ebeam_i):.3e} GeV")
    print(f"  e-   E : {np.mean(Ebeam_e):.6f} +/- {np.std(Ebeam_e):.3e} GeV")
    spread = np.std(Ebeam_i) > 1e-9 or np.std(Ebeam_e) > 1e-9
    print(f"  energy spread : {'ON' if spread else 'OFF'}")

    section("B3. VERTEX")
    V = np.array(vtx)
    print(f"  mean (mm) : ({V[:,0].mean():+.6f}, {V[:,1].mean():+.6f}, "
          f"{V[:,2].mean():+.6f})")
    print(f"  rms  (mm) : ({V[:,0].std():.6f}, {V[:,1].std():.6f}, "
          f"{V[:,2].std():.6f})")
    print(f"  smearing  : {'ON' if V.std(axis=0).max() > 1e-9 else 'OFF'}")

    print("\n  Without a --before file this cannot prove the transformation")
    print("  was a proper Lorentz transformation. Supply one.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Validate the EIC afterburner transformation.")
    ap.add_argument("after")
    ap.add_argument("--before", default=None)
    ap.add_argument("--nmax", type=int, default=20000)
    ap.add_argument("--nfit", type=int, default=2000,
                    help="events pooled into the global 4x4 fit")
    ap.add_argument("--expect-angle", type=float, default=25.0,
                    help="expected full crossing angle, mrad")
    ap.add_argument("--angle-tol", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="tolerance on |Lambda^T g Lambda - g|")
    ap.add_argument("--rtol", type=float, default=1e-10,
                    help="relative tolerance on the fit residual")
    args = ap.parse_args()

    if args.before:
        mode_a(args.after, args.before, args)
    else:
        mode_b(args.after, args)
    print("\nDone.")


if __name__ == "__main__":
    main()
