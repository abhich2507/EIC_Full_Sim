#!/usr/bin/env python3
"""
validate_eic_rotation.py
========================

Validation of the EIC beam-convention rotation (electron -z, ion +z) applied
to the HepMC3-in-ROOT output of the 3He tagged-DIS generator.

WHAT IT CHECKS
--------------
  1. Record sanity      : particle multiplicity, PDG codes, status codes
  2. Beam geometry      : e- beam pz < 0, ion beam pz > 0, px = py = 0
  3. Beam energies      : match the requested Ee and EpA * A
  4. Invariants         : Q2, xB, y, W recomputed FROM THE WRITTEN 4-VECTORS
  5. Weight vector      : layout {w, lambda, n_Lambda}, size distribution
  6. Cross section      : sum(w)/N  with MC uncertainty
  7. Spin asymmetries   : A_L (beam), A_n (target), A_LL (double) + SIGNS
  8. Spectator geometry : spectators must go forward, toward the ion (+z)

  Optional --compare <reference.root> : diffs 4, 6, 7 against a pre-rotation
  file generated with the SAME SEED.  Items 4, 6, 7 must be identical.
  This is the test that catches a mirror reflection (det = -1) masquerading
  as a proper rotation.

USAGE
-----
    python3 validate_eic_rotation.py FILE.root
    python3 validate_eic_rotation.py FILE.root --nmax 500000
    python3 validate_eic_rotation.py NEW.root --compare OLD.root
    python3 validate_eic_rotation.py FILE.root --Ee 5 --EpA 41 --A 3 --Z 2
    python3 validate_eic_rotation.py FILE.root --plots out.png

NOTES
-----
  * 4-momentum is deliberately NOT checked for conservation at the vertex.
    The generator writes the scattered electron, the virtual photon (status 2)
    and the spectator nucleons, but NOT the struck-nucleon remnant, so the
    vertex does not balance.  That is expected.
  * HepMC3 FourVector storage in this tree is m_v1..m_v4 = (px, py, pz, E).
"""

import argparse
import math
import sys

import ROOT


# ----------------------------------------------------------------------------
# C++ helpers, JIT-compiled once.  Doing the per-event work in C++ keeps this
# fast enough to run over all 1e7 events.
# ----------------------------------------------------------------------------
CPP = r"""
#include "ROOT/RVec.hxx"
#include <cmath>

using ROOT::VecOps::RVec;

// Index of the first particle matching (pid, status).  -1 if absent.
// pid == 0 is a wildcard.
int idxOf(const RVec<int>& pid, const RVec<int>& st, int wpid, int wst)
{
    for (size_t i = 0; i < pid.size(); ++i)
        if ((wpid == 0 || pid[i] == wpid) && st[i] == wst) return (int)i;
    return -1;
}

// Index of the beam ion: any status-4 particle that is not the electron.
int idxBeamIon(const RVec<int>& pid, const RVec<int>& st)
{
    for (size_t i = 0; i < pid.size(); ++i)
        if (st[i] == 4 && pid[i] != 11) return (int)i;
    return -1;
}

// Recompute the DIS invariants from the 4-vectors actually stored in the file.
// Returns {Q2, xB, y, W2, s, nu} ; all -1 if the record is malformed.
//
// These are Lorentz scalars, so they are blind to the pi-about-x rotation.
// If any of them shifts after the patch, the patch is wrong.
RVec<double> disKin(const RVec<int>& pid, const RVec<int>& st,
                    const RVec<double>& px, const RVec<double>& py,
                    const RVec<double>& pz, const RVec<double>& e)
{
    RVec<double> bad = {-1., -1., -1., -1., -1., -1.};

    int iki = idxOf(pid, st, 11, 4);   // beam electron
    int iko = idxOf(pid, st, 11, 1);   // scattered electron
    int ipi = idxBeamIon(pid, st);     // beam ion
    if (iki < 0 || iko < 0 || ipi < 0) return bad;

    // q = k_in - k_out
    double qx = px[iki] - px[iko];
    double qy = py[iki] - py[iko];
    double qz = pz[iki] - pz[iko];
    double qE = e [iki] - e [iko];

    double Q2 = -(qE*qE - qx*qx - qy*qy - qz*qz);

    // Per-NUCLEON target momentum: the generator's xB is defined w.r.t. a
    // single nucleon, so divide the nuclear 4-vector by A.  A is inferred
    // from the PDG code 10LZZZAAAI.
    int nuc = pid[ipi];
    double A = 1.0;
    if (nuc > 1000000000) A = (double)((nuc / 10) % 1000);

    double Px = px[ipi] / A, Py = py[ipi] / A;
    double Pz = pz[ipi] / A, PE = e [ipi] / A;

    double Pq = PE*qE - Px*qx - Py*qy - Pz*qz;
    double Pk = PE*e[iki] - Px*px[iki] - Py*py[iki] - Pz*pz[iki];

    double xB = (Pq != 0.) ? Q2 / (2.0 * Pq) : -1.;
    double y  = (Pk != 0.) ? Pq / Pk        : -1.;

    double wE = PE + qE, wx = Px + qx, wy = Py + qy, wz = Pz + qz;
    double W2 = wE*wE - wx*wx - wy*wy - wz*wz;

    double sE = PE + e[iki], sx = Px + px[iki],
           sy = Py + py[iki], sz = Pz + pz[iki];
    double s  = sE*sE - sx*sx - sy*sy - sz*sz;

    return RVec<double>{Q2, xB, y, W2, s, Pq};
}

// Momentum-weighted mean pz of the final-state spectators (everything with
// status 1 that is not the scattered electron).  Must be large and POSITIVE
// in the EIC convention.
double spectatorPz(const RVec<int>& pid, const RVec<int>& st,
                   const RVec<double>& pz)
{
    double sum = 0.; int n = 0;
    for (size_t i = 0; i < pid.size(); ++i)
        if (st[i] == 1 && pid[i] != 11) { sum += pz[i]; ++n; }
    return n ? sum / n : 0.;
}

double at(const RVec<double>& v, int i) { return (i >= 0 && i < (int)v.size()) ? v[i] : 0.; }
int    ati(const RVec<int>&    v, int i) { return (i >= 0 && i < (int)v.size()) ? v[i] : 0; }

// weights = {w, lambda, n_Lambda}; the zero-weight early-return paths push
// only ONE element, so guard every access.
double wgt(const RVec<double>& w)  { return w.size() > 0 ? w[0] : 0.; }
double lam(const RVec<double>& w)  { return w.size() > 1 ? w[1] : 0.; }
double nlam(const RVec<double>& w) { return w.size() > 2 ? w[2] : 0.; }
"""


def build_df(path, nmax=0):
    """Open the file, take the CURRENT cycle of the tree, define columns."""
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        sys.exit(f"ERROR: cannot open {path}")
    # ";*" would give the backup cycle too -- Get() returns the highest cycle.
    tree = f.Get("hepmc3_tree")
    if not tree:
        sys.exit(f"ERROR: no hepmc3_tree in {path}")

    df = ROOT.RDataFrame(tree)
    if nmax > 0:
        df = df.Range(nmax)

    P = "particles."
    df = (df
          .Alias("pid",  P + "pid")
          .Alias("stat", P + "status")
          .Alias("px",   P + "momentum.m_v1")
          .Alias("py",   P + "momentum.m_v2")
          .Alias("pz",   P + "momentum.m_v3")
          .Alias("en",   P + "momentum.m_v4")
          .Define("npart",  "(int)pid.size()")
          .Define("nw",     "(int)weights.size()")
          .Define("w",      "wgt(weights)")
          .Define("lambda_","lam(weights)")
          .Define("nLambda","nlam(weights)")
          .Define("i_ke",   "idxOf(pid, stat, 11, 4)")
          .Define("i_ks",   "idxOf(pid, stat, 11, 1)")
          .Define("i_ion",  "idxBeamIon(pid, stat)")
          .Define("beam_e_pz",  "at(pz, i_ke)")
          .Define("beam_e_E",   "at(en, i_ke)")
          .Define("beam_i_pz",  "at(pz, i_ion)")
          .Define("beam_i_E",   "at(en, i_ion)")
          .Define("beam_i_pdg", "ati(pid, i_ion)")
          .Define("beam_e_px",  "at(px, i_ke)")
          .Define("beam_e_py",  "at(py, i_ke)")
          .Define("beam_i_px",  "at(px, i_ion)")
          .Define("beam_i_py",  "at(py, i_ion)")
          .Define("escat_pz",   "at(pz, i_ks)")
          .Define("escat_E",    "at(en, i_ks)")
          .Define("spec_pz",    "spectatorPz(pid, stat, pz)")
          .Define("kin",  "disKin(pid, stat, px, py, pz, en)")
          .Define("Q2",   "kin[0]")
          .Define("xB",   "kin[1]")
          .Define("yD",   "kin[2]")
          .Define("W2",   "kin[3]")
          .Define("sHat", "kin[4]")
          .Define("W",    "W2 > 0 ? sqrt(W2) : -1.")
          .Define("logQ2","Q2 > 0 ? log10(Q2) : -99.")
          .Define("logxB","xB > 0 ? log10(xB) : -99.")
          )
    return f, df


def ok(flag):
    return "\033[92mPASS\033[0m" if flag else "\033[91mFAIL\033[0m"


def section(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def analyse(path, args):
    """Run every check on one file.  Returns a dict of headline numbers so
    two files can be diffed."""
    _f, df = build_df(path, args.nmax)

    # --- book everything first so RDataFrame makes a single pass ------------
    n_tot   = df.Count()

    h_npart = df.Histo1D(("npart", ";particles / event;", 12, -0.5, 11.5), "npart")
    h_nw    = df.Histo1D(("nw",    ";weights.size();",     6, -0.5,  5.5), "nw")

    m_bepz  = df.Mean("beam_e_pz");   m_beE  = df.Mean("beam_e_E")
    m_bipz  = df.Mean("beam_i_pz");   m_biE  = df.Mean("beam_i_E")
    mx_bepx = df.Max("beam_e_px");    mx_bepy = df.Max("beam_e_py")
    mx_bipx = df.Max("beam_i_px");    mx_bipy = df.Max("beam_i_py")
    min_bepz= df.Max("beam_e_pz")     # worst case: the LARGEST pz must be < 0
    min_bipz= df.Min("beam_i_pz")     # worst case: the SMALLEST pz must be > 0

    n_bad   = df.Filter("i_ke < 0 || i_ks < 0 || i_ion < 0").Count()
    ion_pdg = df.Max("beam_i_pdg")

    # weighted physics
    dfw   = df.Filter("w > 0", "nonzero weight")
    n_pos = dfw.Count()
    sum_w = dfw.Sum("w")
    sum_w2= dfw.Define("w2", "w*w").Sum("w2")
    max_w = dfw.Max("w")

    mean_spec = dfw.Mean("spec_pz")
    min_spec  = dfw.Min("spec_pz")

    # invariants, weighted
    m_Q2 = dfw.Mean("Q2");  m_xB = dfw.Mean("xB")
    m_y  = dfw.Mean("yD");  m_W  = dfw.Mean("W")
    m_s  = dfw.Mean("sHat")
    mn_Q2= dfw.Min("Q2");   mx_Q2= dfw.Max("Q2")
    mn_xB= dfw.Min("xB");   mx_xB= dfw.Max("xB")
    mn_y = dfw.Min("yD");   mx_y = dfw.Max("yD")

    # unphysical-region counters
    n_ybad  = dfw.Filter("yD < 0 || yD > 1").Count()
    n_xbad  = dfw.Filter("xB < 0 || xB > 1.0001").Count()
    n_W2bad = dfw.Filter("W2 < 0").Count()

    # spin sums (weighted)
    s_all  = dfw.Sum("w")
    s_lp   = dfw.Filter("lambda_  > 0").Sum("w")
    s_lm   = dfw.Filter("lambda_  < 0").Sum("w")
    s_np   = dfw.Filter("nLambda > 0").Sum("w")
    s_nm   = dfw.Filter("nLambda < 0").Sum("w")
    s_pp   = dfw.Filter("lambda_*nLambda > 0").Sum("w")
    s_pm   = dfw.Filter("lambda_*nLambda < 0").Sum("w")

    hists = {}
    if args.plots:
        hists["Q2"] = dfw.Histo1D(("hQ2", "log_{10} Q^{2};log_{10}(Q^{2}/GeV^{2});d#sigma",
                                   100, -1, 4), "logQ2", "w")
        hists["xB"] = dfw.Histo1D(("hxB", "log_{10} x_{B};log_{10} x_{B};d#sigma",
                                   100, -5, 0.2), "logxB", "w")
        hists["y"]  = dfw.Histo1D(("hy",  "y;y;d#sigma", 100, 0, 1), "yD", "w")
        hists["W"]  = dfw.Histo1D(("hW",  "W;W [GeV];d#sigma", 100, 0, 60), "W", "w")
        hists["epz"]= dfw.Histo1D(("hepz","scattered e^{-} p_{z};p_{z} [GeV];d#sigma",
                                   100, -30, 30), "escat_pz", "w")
        hists["spz"]= dfw.Histo1D(("hspz","spectator <p_{z}>;p_{z} [GeV];d#sigma",
                                   100, -20, 120), "spec_pz", "w")

    # ---- trigger the event loop -------------------------------------------
    N = n_tot.GetValue()
    print(f"\nProcessed {N:,} events from {path}")

    # =========================== 1. record sanity ==========================
    section("1. RECORD SANITY")
    print(f"  malformed events (missing beam or scattered e-) : {n_bad.GetValue():,}   "
          f"[{ok(n_bad.GetValue() == 0)}]")
    print(f"  beam ion PDG code                               : {ion_pdg.GetValue()}")
    print("  particles / event:")
    hn = h_npart.GetValue()
    for b in range(1, hn.GetNbinsX() + 1):
        c = hn.GetBinContent(b)
        if c:
            print(f"      {int(hn.GetBinCenter(b)):2d} particles : {int(c):>10,}")

    # =========================== 2. beam geometry ==========================
    section("2. BEAM GEOMETRY   (the actual point of the patch)")
    bepz, bipz = m_bepz.GetValue(), m_bipz.GetValue()
    e_neg  = min_bepz.GetValue() < 0.0
    i_pos  = min_bipz.GetValue() > 0.0
    transverse = max(abs(mx_bepx.GetValue()), abs(mx_bepy.GetValue()),
                     abs(mx_bipx.GetValue()), abs(mx_bipy.GetValue()))
    print(f"  <e- beam pz>   = {bepz:+12.5f} GeV   (must be NEGATIVE)  [{ok(e_neg)}]")
    print(f"  <ion beam pz>  = {bipz:+12.5f} GeV   (must be POSITIVE)  [{ok(i_pos)}]")
    print(f"  max |beam px|, |beam py| = {transverse:.3e} GeV")
    print(f"      -> must be 0: the crossing angle is the afterburner's job "
          f"[{ok(transverse < 1e-9)}]")

    # =========================== 3. beam energies ==========================
    section("3. BEAM ENERGIES")
    beE, biE = m_beE.GetValue(), m_biE.GetValue()
    print(f"  e- beam  E = {beE:10.4f} GeV")
    print(f"  ion beam E = {biE:10.4f} GeV   ->  E/A = {biE/args.A:8.4f} GeV/nucleon")
    if args.Ee > 0:
        good = abs(beE - args.Ee) < 1e-6
        print(f"      expected Ee  = {args.Ee}          [{ok(good)}]")
    if args.EpA > 0:
        good = abs(biE / args.A - args.EpA) < 1e-6
        print(f"      expected EpA = {args.EpA} GeV/nucleon  [{ok(good)}]")
    print(f"  sqrt(s) per nucleon = {math.sqrt(m_s.GetValue()):8.4f} GeV")

    # =========================== 4. invariants =============================
    section("4. DIS INVARIANTS   (recomputed from the WRITTEN 4-vectors)")
    print("  These are Lorentz scalars. A proper rotation cannot move them by")
    print("  even one ULP. Compare against the pre-rotation file with --compare.")
    print()
    print(f"  <Q2> = {m_Q2.GetValue():12.6f} GeV^2     range [{mn_Q2.GetValue():.4g}, {mx_Q2.GetValue():.4g}]")
    print(f"  <xB> = {m_xB.GetValue():12.6f}           range [{mn_xB.GetValue():.4g}, {mx_xB.GetValue():.4g}]")
    print(f"  <y>  = {m_y.GetValue():12.6f}           range [{mn_y.GetValue():.4g}, {mx_y.GetValue():.4g}]")
    print(f"  <W>  = {m_W.GetValue():12.6f} GeV")
    print()
    print(f"  events with y outside [0,1]  : {n_ybad.GetValue():,}  [{ok(n_ybad.GetValue()==0)}]")
    print(f"  events with xB outside [0,1] : {n_xbad.GetValue():,}  [{ok(n_xbad.GetValue()==0)}]")
    print(f"  events with W2 < 0           : {n_W2bad.GetValue():,}  [{ok(n_W2bad.GetValue()==0)}]")

    # =========================== 5. weight vector ==========================
    section("5. WEIGHT VECTOR LAYOUT")
    print("  Expected: {w, lambda, n_Lambda} -> size 3.")
    print("  Size 1 is the zero-weight early return (alpha < xB, or NaN).")
    hw = h_nw.GetValue()
    for b in range(1, hw.GetNbinsX() + 1):
        c = hw.GetBinContent(b)
        if c:
            print(f"      size {int(hw.GetBinCenter(b))} : {int(c):>10,}  "
                  f"({100.0*c/N:5.2f} %)")

    # =========================== 6. cross section ==========================
    section("6. CROSS SECTION")
    sw, sw2 = sum_w.GetValue(), sum_w2.GetValue()
    xsec  = sw / N
    dxsec = math.sqrt(sw2) / N
    print(f"  events with w > 0 : {n_pos.GetValue():,}  ({100.0*n_pos.GetValue()/N:.2f} %)")
    print(f"  sum(w)            : {sw:.10e}")
    print(f"  sigma = sum(w)/N  : {xsec:.8e}  +/- {dxsec:.3e}  nb"
          f"   ({100*dxsec/xsec if xsec else 0:.3f} %)")
    print(f"  max single weight : {max_w.GetValue():.6e}"
          f"   (= {max_w.GetValue()/sw*100 if sw else 0:.4f} % of the total)")
    if sw and max_w.GetValue() / sw > 0.01:
        print("      WARNING: one event carries >1% of the integral -> the")
        print("      sampling PDF is a poor match to the physics somewhere.")

    # =========================== 7. asymmetries ============================
    section("7. SPIN ASYMMETRIES   (the reflection test)")
    tot = s_all.GetValue()

    def asym(p, m):
        d = p + m
        if d == 0:
            return 0.0, 0.0
        a = (p - m) / d
        return a, math.sqrt(max(0.0, (1 - a * a) / max(n_pos.GetValue(), 1)))

    A_L,  dA_L  = asym(s_lp.GetValue(), s_lm.GetValue())
    A_n,  dA_n  = asym(s_np.GetValue(), s_nm.GetValue())
    A_LL, dA_LL = asym(s_pp.GetValue(), s_pm.GetValue())
    print(f"  A_L  (lambda  +/-)        = {A_L :+.6f} +/- {dA_L :.6f}")
    print(f"  A_n  (n_Lambda +/-)       = {A_n :+.6f} +/- {dA_n :.6f}")
    print(f"  A_LL (lambda * n_Lambda)  = {A_LL:+.6f} +/- {dA_LL:.6f}   <-- SIGN MATTERS")
    print()
    print("  A_L and A_n must be consistent with ZERO: the single-spin terms")
    print("  average out because both signs are thrown 50/50 and the")
    print("  unpolarized part dominates.")
    print(f"      A_L  compatible with 0 : [{ok(abs(A_L) < 5*max(dA_L, 1e-12))}]")
    print(f"      A_n  compatible with 0 : [{ok(abs(A_n) < 5*max(dA_n, 1e-12))}]")
    print("  A_LL is the physics. Its MAGNITUDE and its SIGN must both match")
    print("  the pre-rotation run. A flipped sign = you used a reflection")
    print("  (pz -> -pz) instead of a rotation (py, pz -> -py, -pz).")

    # =========================== 8. spectators =============================
    section("8. SPECTATOR GEOMETRY")
    msp = mean_spec.GetValue()
    print(f"  <spectator pz> = {msp:+10.4f} GeV   (must be LARGE and POSITIVE)"
          f"  [{ok(msp > 0)}]")
    print(f"  min over events = {min_spec.GetValue():+10.4f} GeV")
    print("  Spectators are nuclear fragments: they carry roughly the beam")
    print(f"  per-nucleon momentum, so <pz> should sit near {args.EpA:.0f} GeV.")

    if args.plots and hists:
        c = ROOT.TCanvas("c", "validation", 1500, 950)
        c.Divide(3, 2)
        for i, k in enumerate(["Q2", "xB", "y", "W", "epz", "spz"]):
            c.cd(i + 1)
            if k in ("Q2", "xB"):
                ROOT.gPad.SetLogy()
            hists[k].GetValue().Draw("hist")
        c.SaveAs(args.plots)
        print(f"\n  plots written to {args.plots}")

    return dict(N=N, sum_w=sw, xsec=xsec, dxsec=dxsec,
                Q2=m_Q2.GetValue(), xB=m_xB.GetValue(),
                y=m_y.GetValue(), W=m_W.GetValue(), s=m_s.GetValue(),
                A_L=A_L, A_n=A_n, A_LL=A_LL, dA_LL=dA_LL,
                bepz=bepz, bipz=bipz)


def dump_event(path, ievt=0):
    """Print one raw event so the geometry can be eyeballed."""
    f = ROOT.TFile.Open(path)
    t = f.Get("hepmc3_tree")
    t.GetEntry(ievt)
    section(f"RAW DUMP OF EVENT {ievt}")
    print(f"  {'#':>3} {'PDG':>12} {'st':>3} "
          f"{'px':>12} {'py':>12} {'pz':>14} {'E':>14}")
    print("  " + "-" * 76)
    n = len(t.particles.pid) if hasattr(t.particles, "pid") else 0
    for i in range(n):
        print(f"  {i:>3} {t.particles.pid[i]:>12} {t.particles.status[i]:>3} "
              f"{t.particles.momentum.m_v1[i]:>12.5f} "
              f"{t.particles.momentum.m_v2[i]:>12.5f} "
              f"{t.particles.momentum.m_v3[i]:>14.5f} "
              f"{t.particles.momentum.m_v4[i]:>14.5f}")
    print(f"\n  weights = {[t.weights[i] for i in range(len(t.weights))]}")
    print("            (index 0 = cross-section weight, 1 = lambda, 2 = n_Lambda)")
    f.Close()


def compare(a, b):
    section("COMPARISON: rotated vs reference")
    print("  Invariants and the cross section MUST be identical (same seed).")
    print("  A_LL must be identical INCLUDING ITS SIGN.\n")
    print(f"  {'quantity':<14} {'rotated':>18} {'reference':>18} {'rel. diff':>14}   verdict")
    print("  " + "-" * 84)
    for key, tol in [("sum_w", 1e-12), ("xsec", 1e-12), ("Q2", 1e-12),
                     ("xB", 1e-12), ("y", 1e-12), ("W", 1e-12),
                     ("s", 1e-12), ("A_LL", 1e-10)]:
        va, vb = a[key], b[key]
        rel = abs(va - vb) / abs(vb) if vb else abs(va - vb)
        print(f"  {key:<14} {va:>18.10g} {vb:>18.10g} {rel:>14.2e}   [{ok(rel < tol)}]")
    print()
    if a["A_LL"] * b["A_LL"] < 0:
        print("  \033[91m*** A_LL FLIPPED SIGN ***\033[0m")
        print("  You applied a MIRROR REFLECTION, not a rotation. Check that")
        print("  toEIC() negates BOTH py and pz:")
        print("      return PxPyPzEVector(v.Px(), -v.Py(), -v.Pz(), v.E());")
    else:
        print("  A_LL sign preserved -> the transformation is a proper rotation.")
    print()
    print(f"  beam pz, rotated   : e- {a['bepz']:+.4f}   ion {a['bipz']:+.4f}")
    print(f"  beam pz, reference : e- {b['bepz']:+.4f}   ion {b['bipz']:+.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--compare", default=None,
                   help="pre-rotation reference file, SAME SEED")
    p.add_argument("--nmax", type=int, default=0,
                   help="only process the first N events (disables MT)")
    p.add_argument("--Ee",  type=float, default=-1, help="expected electron energy")
    p.add_argument("--EpA", type=float, default=-1, help="expected GeV per nucleon")
    p.add_argument("--A", type=int, default=3)
    p.add_argument("--Z", type=int, default=2)
    p.add_argument("--plots", default=None, help="write control plots here, e.g. val.png")
    p.add_argument("--dump", type=int, default=0, help="also dump raw event N")
    args = p.parse_args()

    ROOT.gROOT.SetBatch(True)
    if args.nmax == 0:
        ROOT.EnableImplicitMT()
    ROOT.gInterpreter.Declare(CPP)

    res = analyse(args.file, args)

    if args.dump >= 0 and args.plots is None:
        dump_event(args.file, args.dump)

    if args.compare:
        print("\n\n" + "#" * 74)
        print("# REFERENCE FILE")
        print("#" * 74)
        ref = analyse(args.compare, args)
        compare(res, ref)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
