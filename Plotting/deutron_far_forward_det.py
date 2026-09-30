import os
import ROOT
import numpy as np
import awkward as ak
import uproot

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kViridis)
os.makedirs("./Plots", exist_ok=True)

DEUTERON = 1000010020
path = "/work/eic/users/aabhishe/EIC_3He_5x41_XRot_pi_bc.hepmc3.tree_sim_output_recon.root"
tree = uproot.open(f"{path}:events")

# ==================================================================
# MC truth
# ==================================================================
mc_pdg    = tree["MCParticles.PDG"].array()
mc_status = tree["MCParticles.generatorStatus"].array()

GEN_STATUS = 1          # taken from gen 
mc_is_gen_d = (mc_pdg == DEUTERON) & (mc_status == GEN_STATUS)

# Global flat index bookkeeping: lets us deduplicate hits belonging to the
# same MC particle and map "was hit" back onto the MCParticles collection.
n_mc       = ak.to_numpy(ak.num(mc_pdg))
offsets    = np.concatenate([[0], np.cumsum(n_mc)])[:-1]
n_mc_total = int(n_mc.sum())

flat_is_gen_d = ak.to_numpy(ak.flatten(mc_is_gen_d))
n_gen_d       = int(flat_is_gen_d.sum())

# ------------------------------------------------------------------
# generatorStatus audit.  EVERY acceptance below is divided by n_gen_d,
# so if the generator tags the spectator deuteron as beam remnant (4) or
# documentation (2/3) instead of final-state (1), every number is wrong by
# the same factor.  Look at this table once, then fix GEN_STATUS.
# ------------------------------------------------------------------
_d_status = ak.to_numpy(ak.flatten(mc_status[mc_pdg == DEUTERON]))
print("[status] generatorStatus of all PDG 1000010020 in MCParticles:")
if len(_d_status):
    for s, c in zip(*np.unique(_d_status, return_counts=True)):
        flag = "  <-- used as denominator" if s == GEN_STATUS else ""
        print(f"         status={int(s):<4d} n={int(c):<8d}{flag}")
else:
    print("         NONE FOUND - wrong PDG or wrong sample.")
print(f"[status] n_gen_d = {n_gen_d}\n")


# ==================================================================
# Association validation
#
# Three assumptions are made when we do mc_pdg[assoc_index]; none are
# guaranteed by the file format, so check them once per collection instead of
# silently producing wrong PDGs:
#
#   (1) the association collection is index-parallel to the hit collection
#       (one-to-one relation -> same number of entries per event),
#   (2) every index is either the -1 "no link" sentinel or inside
#       [0, n_MCParticles) *of that same event*,
#   (3) the resulting frame is self-consistent: positions of SimTrackerHits are
#       global-frame mm for every subdetector, so B0 / OMD / RP z-cuts are
#       directly comparable numbers.
# ==================================================================
def validate_assoc(hit_coll, assoc_coll, verbose=True):
    """Check hit<->MC association integrity. Returns (idx, valid, safe)."""
    idx    = tree[f"{assoc_coll}.index"].array()
    n_hits = ak.num(tree[f"{hit_coll}.position.x"].array())
    n_assc = ak.num(idx)

    # (1) per-event parallelism between hits and associations
    n_bad_len = int(ak.sum(n_hits != n_assc))
    if n_bad_len:
        raise RuntimeError(
            f"{assoc_coll}: {n_bad_len} events where the association count "
            f"differs from the hit count -> the relation is NOT index-parallel "
            f"to {hit_coll}. Boolean-masking the positions with a per-hit mask "
            f"is invalid for this collection."
        )

    # (2) index bounds, evaluated against the MCParticles count of the SAME event
    in_range = (idx >= 0) & (idx < n_mc)
    n_neg    = int(ak.sum(idx < 0))
    n_over   = int(ak.sum(idx >= n_mc))
    valid    = in_range
    safe     = ak.where(valid, idx, 0)   # clamp so the gather never wraps

    if verbose:
        tot = int(ak.sum(n_assc))
        print(f"[assoc] {hit_coll:<22} hits={tot:<8d} "
              f"unlinked(-1)={n_neg:<6d} out-of-range={n_over:<6d} "
              f"usable={int(ak.sum(valid)):d}")
        if n_over:
            print(f"        WARNING: {n_over} indices point outside MCParticles "
                  f"of their own event (cross-frame relation?). Dropped.")
    return idx, valid, safe


def hit_pdg_and_mask(hit_coll, assoc_coll):
    """Return (pdg per hit, deuteron mask per hit, valid-link mask, safe idx)."""
    _, valid, safe = validate_assoc(hit_coll, assoc_coll)
    pdg = ak.where(valid, mc_pdg[safe], 0)
    return pdg, (pdg == DEUTERON) & valid, valid, safe


def get_xyz(hit_coll, mask):
    x = tree[f"{hit_coll}.position.x"].array()[mask]
    y = tree[f"{hit_coll}.position.y"].array()[mask]
    z = tree[f"{hit_coll}.position.z"].array()[mask]
    return x, y, z


# ==================================================================
# Station definitions -- ONE source of truth, used by BOTH the acceptance
# and the plots, so the number printed on a canvas always refers to exactly
# the hits drawn above it.  (lo, hi); None = open ended.
# ==================================================================
STATIONS = {
    "RP": [("Station 1 (z #approx 32.5 m)", None,  32600.),
           ("Station 2 (z #approx 34.3 m)", 34200., None )],
    "OMD": [("Station 1 (z #approx 25.6 m)", None,  25600.),
            ("Station 2 (z #approx 26.9 m)", 26800., None )],
    "B0": [("Layer 1 (z < 6.00 m)",     None,  6000.),
           ("Layer 2 (6.10-6.30 m)",    6100., 6300.),
           ("Layer 3 (6.40-6.50 m)",    6400., 6500.),
           ("Layer 4 (z > 6.60 m)",     6600., None )],
}

# Coincidence requirement: N-of-M stations must fire.
#   RP / OMD 2-of-2 -> two space points, the minimum for a local track angle.
#   B0       3-of-4 -> curvature inside B0pf, tolerant of one dead layer.
N_REQUIRED = {"RP": 2, "OMD": 2, "B0": 3}


def zmask(z, lo, hi):
    if lo is None:
        return z < hi
    if hi is None:
        return z > lo
    return (z > lo) & (z < hi)


def particles_with_hit(safe, sel):
    """Global per-MC-particle bool: >=1 selected hit. Collapses layer multiplicity."""
    gidx = ak.to_numpy(ak.flatten(safe[sel] + offsets))
    out = np.zeros(n_mc_total, dtype=bool)
    if len(gidx):
        out[np.unique(gidx)] = True
    return out


def detector_acceptance(key, hit_coll, assoc_coll):
    """Per-station bits -> inclusive and N-of-M coincidence acceptance."""
    _, d_mask, _, safe = hit_pdg_and_mask(hit_coll, assoc_coll)
    z_raw = tree[f"{hit_coll}.position.z"].array()

    per_station, station_counts = [], []
    for _, lo, hi in STATIONS[key]:
        bits = particles_with_hit(safe, d_mask & zmask(z_raw, lo, hi))
        per_station.append(bits)
        station_counts.append(int((bits & flat_is_gen_d).sum()))

    n_fired   = np.sum(np.stack(per_station), axis=0)      # stations hit per particle
    incl_bits = (n_fired >= 1) & flat_is_gen_d
    coin_bits = (n_fired >= N_REQUIRED[key]) & flat_is_gen_d

    # hits landing in none of the windows -> would inflate an inclusive count
    in_any = zmask(z_raw, None, STATIONS[key][0][2])
    for _, lo, hi in STATIONS[key][1:]:
        in_any = in_any | zmask(z_raw, lo, hi)
    n_outside = int(ak.sum(d_mask & ~in_any))

    return {
        "station_counts": station_counts,
        "n_incl": int(incl_bits.sum()),
        "n_coin": int(coin_bits.sum()),
        "incl_bits": incl_bits,
        "coin_bits": coin_bits,
        "d_mask": d_mask,
        "n_outside": n_outside,
    }


def binom_err(k, n):
    if n == 0:
        return 0.0
    p = k / n
    return np.sqrt(max(p * (1 - p), 0.0) / n)


# ==================================================================
# Run it
# ==================================================================
COLLS = {
    "B0":  ("B0TrackerHits",          "_B0TrackerHits_particle"),
    "OMD": ("ForwardOffMTrackerHits", "_ForwardOffMTrackerHits_particle"),
    "RP":  ("ForwardRomanPotHits",    "_ForwardRomanPotHits_particle"),
}

RES = {}
for key, (hc, ac) in COLLS.items():
    RES[key] = detector_acceptance(key, hc, ac)
print()

print(f"Generated deuterons (status=={GEN_STATUS}) : {n_gen_d}\n")
for key in ("B0", "OMD", "RP"):
    r = RES[key]
    print(f"--- {key}  ({N_REQUIRED[key]}-of-{len(STATIONS[key])} coincidence) ---")
    for (lbl, _, _), c in zip(STATIONS[key], r["station_counts"]):
        a = c / n_gen_d if n_gen_d else 0.0
        print(f"    {lbl:<30} {c:>8d}   A = {a:.5f}")
    ai = r["n_incl"] / n_gen_d if n_gen_d else 0.0
    ac_ = r["n_coin"] / n_gen_d if n_gen_d else 0.0
    print(f"    {'inclusive (>=1 station)':<30} {r['n_incl']:>8d}   "
          f"A = {ai:.5f} +/- {binom_err(r['n_incl'], n_gen_d):.5f}")
    print(f"    {'coincidence':<30} {r['n_coin']:>8d}   "
          f"A = {ac_:.5f} +/- {binom_err(r['n_coin'], n_gen_d):.5f}")
    print(f"    reconstructable fraction A_coin/A_incl = "
          f"{(r['n_coin']/r['n_incl'] if r['n_incl'] else 0.0):.5f}")
    print(f"    deuteron hits outside all z-windows    = {r['n_outside']}")
    tot = int(ak.sum(ak.num(r["d_mask"])))
    sel = int(ak.sum(r["d_mask"]))
    print(f"    deuteron hit purity = {sel}/{tot} = "
          f"{(sel/tot if tot else 0.0):.3f}\n")

# ------------------------------------------------------------------
# Cross-detector: the coincidence bits are per-particle booleans on the same
# flat MCParticles axis, so they can be AND/OR-ed directly. This is why the
# three acceptances above must never be added together.
# ------------------------------------------------------------------
cb = {k: RES[k]["coin_bits"] for k in RES}
print("--- cross-detector (coincidence bits) ---")
for a, b in (("B0", "OMD"), ("B0", "RP"), ("OMD", "RP")):
    n = int((cb[a] & cb[b]).sum())
    print(f"    {a} AND {b:<4}: {n:>8d}   A = {(n/n_gen_d if n_gen_d else 0):.5f}")
n_any = int((cb["B0"] | cb["OMD"] | cb["RP"]).sum())
print(f"    ANY of the three : {n_any:>8d}   "
      f"A = {(n_any/n_gen_d if n_gen_d else 0):.5f}\n")


# ==================================================================
# XY occupancy maps (deuterons only), station windows identical to above
# ==================================================================
RANGES = {
    "RP":  [((100, -1300., -950.),  (100, -90., 90.)),
            ((100, -1400., -1050.), (100, -90., 90.))],
    "OMD": [((100, -1000., -850.),  (100, -100., 100.)),
            ((100, -1050., -940.),  (100, -100., 100.))],
    "B0":  None,       # auto-ranged, shared across the four layers
}


def fill_histogram(name, title, x_arr, y_arr, mask, x_bins, y_bins):
    x_vals = ak.to_numpy(ak.flatten(x_arr[mask])).astype(np.float64)
    y_vals = ak.to_numpy(ak.flatten(y_arr[mask])).astype(np.float64)

    hist = ROOT.TH2F(name, title,
                     x_bins[0], x_bins[1], x_bins[2],
                     y_bins[0], y_bins[1], y_bins[2])
    if len(x_vals):
        hist.FillN(len(x_vals), x_vals, y_vals,
                   np.ones(len(x_vals), dtype=np.float64))
        print(f"  {name}: {len(x_vals)} deuteron hits, "
              f"x=[{x_vals.min():.1f},{x_vals.max():.1f}] "
              f"y=[{y_vals.min():.1f},{y_vals.max():.1f}]")
    else:
        print(f"  {name}: empty")
    hist.GetXaxis().SetTitle("X Position [mm]")
    hist.GetYaxis().SetTitle("Y Position [mm]")
    hist.GetYaxis().SetTitleOffset(1.2)
    return hist


def auto_range(x_arr, y_arr, masks, pad=0.05, nb=100):
    xs = np.concatenate([ak.to_numpy(ak.flatten(x_arr[m])) for m in masks]) \
         if masks else np.array([0.])
    ys = np.concatenate([ak.to_numpy(ak.flatten(y_arr[m])) for m in masks]) \
         if masks else np.array([0.])
    if len(xs) == 0:
        return (nb, -1., 1.), (nb, -1., 1.)
    def span(v):
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-6:
            lo, hi = lo - 1., hi + 1.
        d = (hi - lo) * pad
        return (nb, lo - d, hi + d)
    return span(xs), span(ys)


def build_maps(key):
    hc, ac = COLLS[key]
    d_mask = RES[key]["d_mask"]
    x, y, z = get_xyz(hc, d_mask)
    masks = [zmask(z, lo, hi) for _, lo, hi in STATIONS[key]]

    rng = RANGES[key]
    if rng is None:
        xb, yb = auto_range(x, y, masks)
        rng = [(xb, yb)] * len(masks)

    hs = []
    for i, ((lbl, _, _), m) in enumerate(zip(STATIONS[key], masks)):
        hs.append(fill_histogram(f"h_{key.lower()}{i+1}",
                                 f"Deuterons: {key} {lbl}",
                                 x, y, m, rng[i][0], rng[i][1]))
    return hs


def draw_grid(cname, ctitle, hists, ncol, nrow, acc_incl, acc_coin,
              nreq, nstat, outfile, width, height):
    c = ROOT.TCanvas(cname, ctitle, width, height)
    c.Divide(ncol, nrow)
    for i, h in enumerate(hists, start=1):
        c.cd(i)
        ROOT.gPad.SetRightMargin(0.15)
        ROOT.gPad.SetLeftMargin(0.13)
        ROOT.gPad.SetBottomMargin(0.2)
        ROOT.gPad.SetTopMargin(0.10)
        h.Draw("COLZ")
    c.cd(0)
    lat = ROOT.TLatex()
    lat.SetNDC(); lat.SetTextSize(0.050 if nrow == 1 else 0.025)
    lat.DrawLatex(0.20, 0.065 if nrow == 1 else 0.015,
                  f"deuteron acceptance: inclusive = {acc_incl:.5f},  "
                  f"{nreq}-of-{nstat} coincidence = {acc_coin:.5f}  "
                  f"(N_{{gen}} = {n_gen_d})")
    c.SaveAs(outfile)
    return c


CANVAS = {
    "RP":  ("canvas_rp", "Roman Pot Deuteron Hits", 2, 1, 1400, 600,
            "./Plots/rompot_deuteron_hits_xy_both_stations.pdf"),
    "OMD": ("canvas_om", "OMD Deuteron Hits",       2, 1, 1400, 600,
            "./Plots/offm_deuteron_hits_xy_both_stations.pdf"),
    "B0":  ("canvas_b0", "B0 Tracker Deuteron Hits", 2, 2, 1400, 1200,
            "./Plots/b0_deuteron_hits_xy_all_layers.pdf"),
}

keep = []
for key in ("RP", "OMD", "B0"):
    hs = build_maps(key)
    cn, ct, nc, nr, w, h, out = CANVAS[key]
    keep.append(draw_grid(cn, ct, hs, nc, nr,
                          RES[key]["n_incl"] / n_gen_d if n_gen_d else 0.0,
                          RES[key]["n_coin"] / n_gen_d if n_gen_d else 0.0,
                          N_REQUIRED[key], len(STATIONS[key]), out, w, h))
    keep.extend(hs)
print()


# ==================================================================
# DIFFERENTIAL RP ACCEPTANCE vs momentum and theta
#
# incl_bits / coin_bits are per-MC-particle booleans on the flattened
# MCParticles axis, so they index the flattened momentum arrays directly:
# numerator and denominator are by construction the same particles, binned
# the same way.  No re-matching, no double counting.
# ==================================================================
px = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.x"].array())).astype(np.float64)
py = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.y"].array())).astype(np.float64)
pz = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.z"].array())).astype(np.float64)

# ------------------------------------------------------------------
# Frame. The sample is "XRot": the crossing angle is already applied, so the
# lab z-axis is NOT the hadron beam axis. Angles must be measured from the
# beam, otherwise the whole spectator peak sits at ~25 mrad and the RP
# acceptance window is smeared across it. Rotate about y to undo it.
# Flip ROT_SIGN if the beam-frame median below is not the one near zero.
# ------------------------------------------------------------------
CROSSING_ANGLE = 0.025      # rad
ROT_SIGN       = -1.0       # -1 undoes a +25 mrad rotation about +y

_a = ROT_SIGN * CROSSING_ANGLE
px_b =  px * np.cos(_a) + pz * np.sin(_a)
pz_b = -px * np.sin(_a) + pz * np.cos(_a)
py_b =  py

p_tot     = np.sqrt(px * px + py * py + pz * pz)
theta_lab = np.arctan2(np.hypot(px, py), pz) * 1e3                    # mrad
theta_bm  = np.arctan2(np.hypot(px_b, py_b), pz_b) * 1e3              # mrad
pt_bm     = np.hypot(px_b, py_b)

g = flat_is_gen_d
print("[frame] generated deuterons: "
      f"median theta_lab = {np.median(theta_lab[g]):.3f} mrad, "
      f"median theta_beam = {np.median(theta_bm[g]):.3f} mrad")
print("[frame] the beam-frame column should be the one peaking near 0; "
      "flip ROT_SIGN otherwise.")

# x_L relative to the nominal per-nucleon beam momentum times A=2
P_BEAM_PER_NUCLEON = 41.0
P_BEAM_D           = 2.0 * P_BEAM_PER_NUCLEON
xL = p_tot / P_BEAM_D

print(f"[frame] generated deuteron p: median = {np.median(p_tot[g]):.2f} GeV, "
      f"range = [{p_tot[g].min():.2f}, {p_tot[g].max():.2f}] GeV")
print(f"[frame] generated deuteron x_L (p / {P_BEAM_D:.0f} GeV): "
      f"median = {np.median(xL[g]):.4f}\n")

rp_incl = RES["RP"]["incl_bits"]
rp_coin = RES["RP"]["coin_bits"]


def _range(v, lo_q=0.0, hi_q=99.8, pad=0.05, floor=None):
    if len(v) == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, lo_q), np.percentile(v, hi_q)
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    d = (hi - lo) * pad
    lo = lo - d if floor is None else max(floor, lo - d)
    return float(lo), float(hi + d)


def fill1d(name, title, xtitle, vals, nb, lo, hi):
    h = ROOT.TH1D(name, f"{title};{xtitle};generated deuterons", nb, lo, hi)
    v = np.ascontiguousarray(vals, dtype=np.float64)
    if len(v):
        h.FillN(len(v), v, np.ones(len(v), dtype=np.float64))
    h.Sumw2()
    return h


def eff_plot(name, title, xtitle, var, nb, lo, hi):
    """Return (h_gen, h_incl, h_coin, TEfficiency incl, TEfficiency coin)."""
    h_gen  = fill1d(f"{name}_gen",  title, xtitle, var[g],          nb, lo, hi)
    h_inc  = fill1d(f"{name}_inc",  title, xtitle, var[rp_incl],    nb, lo, hi)
    h_coi  = fill1d(f"{name}_coi",  title, xtitle, var[rp_coin],    nb, lo, hi)

    # Clopper-Pearson: correct near A -> 0 or 1, where the Gaussian error is not.
    e_inc = ROOT.TEfficiency(h_inc, h_gen); e_inc.SetName(f"{name}_eff_inc")
    e_coi = ROOT.TEfficiency(h_coi, h_gen); e_coi.SetName(f"{name}_eff_coi")
    e_inc.SetStatisticOption(ROOT.TEfficiency.kFCP)
    e_coi.SetStatisticOption(ROOT.TEfficiency.kFCP)
    return h_gen, h_inc, h_coi, e_inc, e_coi


def style_spectra(h_gen, h_inc, h_coi):
    h_gen.SetLineColor(ROOT.kGray + 2); h_gen.SetLineWidth(2)
    h_gen.SetFillColorAlpha(ROOT.kGray, 0.35)
    h_inc.SetLineColor(ROOT.kAzure + 2); h_inc.SetLineWidth(2)
    h_coi.SetLineColor(ROOT.kRed + 1);   h_coi.SetLineWidth(2)
    for h in (h_gen, h_inc, h_coi):
        h.GetYaxis().SetTitle("deuterons")
        h.GetYaxis().SetTitleOffset(1.35)


def style_eff(e_inc, e_coi):
    e_inc.SetLineColor(ROOT.kAzure + 2); e_inc.SetMarkerColor(ROOT.kAzure + 2)
    e_inc.SetMarkerStyle(20); e_inc.SetMarkerSize(0.9); e_inc.SetLineWidth(2)
    e_coi.SetLineColor(ROOT.kRed + 1);   e_coi.SetMarkerColor(ROOT.kRed + 1)
    e_coi.SetMarkerStyle(21); e_coi.SetMarkerSize(0.9); e_coi.SetLineWidth(2)


def draw_diff(cname, outfile, blocks):
    """blocks = [(xtitle, h_gen, h_inc, h_coi, e_inc, e_coi), ...] one column each."""
    n = len(blocks)
    c = ROOT.TCanvas(cname, cname, 700 * n, 1000)
    c.Divide(n, 2)

    for i, (xt, hg, hi_, hc, ei, ec) in enumerate(blocks):
        # --- top row: spectra, generated vs accepted
        c.cd(i + 1)
        ROOT.gPad.SetRightMargin(0.06)
        ROOT.gPad.SetLeftMargin(0.15)
        ROOT.gPad.SetBottomMargin(0.14)
        ROOT.gPad.SetTopMargin(0.10)
        ROOT.gPad.SetLogy()
        hg.SetMaximum(hg.GetMaximum() * 5.0)
        hg.SetMinimum(0.5)
        hg.Draw("HIST")
        hi_.Draw("HIST SAME")
        hc.Draw("HIST SAME")
        leg = ROOT.TLegend(0.45, 0.72, 0.93, 0.89)
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.033)
        leg.AddEntry(hg,  "generated", "F")
        leg.AddEntry(hi_, "RP accepted (inclusive)", "L")
        leg.AddEntry(hc,  "RP accepted (2-of-2 coincidence)", "L")
        leg.Draw()
        keep.append(leg)

        # --- bottom row: acceptance
        c.cd(n + i + 1)
        ROOT.gPad.SetRightMargin(0.06)
        ROOT.gPad.SetLeftMargin(0.15)
        ROOT.gPad.SetBottomMargin(0.14)
        ROOT.gPad.SetTopMargin(0.10)
        ROOT.gPad.SetGridx(); ROOT.gPad.SetGridy()
        frame = ROOT.gPad.DrawFrame(hg.GetXaxis().GetXmin(), 0.0,
                                    hg.GetXaxis().GetXmax(), 1.05)
        frame.GetXaxis().SetTitle(xt)
        frame.GetYaxis().SetTitle("RP acceptance")
        frame.GetYaxis().SetTitleOffset(1.35)
        ei.Draw("P SAME")
        ec.Draw("P SAME")
        leg2 = ROOT.TLegend(0.45, 0.76, 0.93, 0.89)
        leg2.SetBorderSize(0); leg2.SetFillStyle(0); leg2.SetTextSize(0.033)
        leg2.AddEntry(ei, "inclusive (#geq1 station)", "LP")
        leg2.AddEntry(ec, "2-of-2 coincidence", "LP")
        leg2.Draw()
        keep.extend([frame, leg2])

    c.cd(0)
    lat = ROOT.TLatex(); lat.SetNDC(); lat.SetTextSize(0.022)
    lat.DrawLatex(0.15, 0.005,
                  f"Roman Pots, deuterons, N_{{gen}} = {n_gen_d};  "
                  f"errors = Clopper-Pearson 68% CL")
    c.SaveAs(outfile)
    return c


NB = 50
p_lo,  p_hi  = _range(p_tot[g],    0.2, 99.8, floor=0.0)
th_lo, th_hi = _range(theta_bm[g], 0.0, 99.5, floor=0.0)
xl_lo, xl_hi = _range(xL[g],       0.2, 99.8, floor=0.0)

blk_p  = ("p [GeV/c]",) + eff_plot("rp_p",  "Deuterons: RP acceptance vs momentum",
                                   "p [GeV/c]", p_tot, NB, p_lo, p_hi)
blk_th = ("#theta_{beam} [mrad]",) + eff_plot("rp_th", "Deuterons: RP acceptance vs #theta",
                                              "#theta_{beam} [mrad]", theta_bm,
                                              NB, th_lo, th_hi)
blk_xl = ("x_{L} = p / p_{beam}",) + eff_plot("rp_xl", "Deuterons: RP acceptance vs x_{L}",
                                              "x_{L}", xL, NB, xl_lo, xl_hi)

for b in (blk_p, blk_th, blk_xl):
    style_spectra(b[1], b[2], b[3])
    style_eff(b[4], b[5])
    keep.extend(b[1:])

keep.append(draw_diff("canvas_rp_diff",
                      "./Plots/rompot_deuteron_acceptance_vs_p_theta_xL.pdf",
                      [blk_p, blk_th, blk_xl]))

# ------------------------------------------------------------------
# 2D acceptance map: where in (theta, p) the coincidence requirement bites.
# Divide accepted by generated bin-by-bin; bins with no generated deuterons
# are blanked so an empty region is not drawn as zero acceptance.
# ------------------------------------------------------------------
def map2d(name, title, sel):
    h = ROOT.TH2D(name, title, 40, th_lo, th_hi, 40, p_lo, p_hi)
    xs = np.ascontiguousarray(theta_bm[sel], dtype=np.float64)
    ys = np.ascontiguousarray(p_tot[sel],    dtype=np.float64)
    if len(xs):
        h.FillN(len(xs), xs, ys, np.ones(len(xs), dtype=np.float64))
    return h


h2_gen  = map2d("h2_gen",  "generated",  g)
h2_inc  = map2d("h2_inc",  "inclusive",  rp_incl)
h2_coin = map2d("h2_coin", "coincidence", rp_coin)

c2 = ROOT.TCanvas("canvas_rp_map", "RP acceptance map", 1400, 600)
c2.Divide(2, 1)
for i, (hnum, lbl) in enumerate(((h2_inc, "inclusive (#geq1 station)"),
                                 (h2_coin, "2-of-2 coincidence")), start=1):
    hr = hnum.Clone(f"{hnum.GetName()}_ratio")
    hr.Divide(h2_gen)
    for bx in range(1, hr.GetNbinsX() + 1):
        for by in range(1, hr.GetNbinsY() + 1):
            if h2_gen.GetBinContent(bx, by) == 0:
                hr.SetBinContent(bx, by, -1.0)   # blank, distinct from true 0
    hr.SetTitle(f"Deuteron RP acceptance, {lbl}")
    hr.GetXaxis().SetTitle("#theta_{beam} [mrad]")
    hr.GetYaxis().SetTitle("p [GeV/c]")
    hr.GetYaxis().SetTitleOffset(1.2)
    hr.SetMinimum(0.0); hr.SetMaximum(1.0)
    c2.cd(i)
    ROOT.gPad.SetRightMargin(0.15)
    ROOT.gPad.SetLeftMargin(0.13)
    ROOT.gPad.SetBottomMargin(0.2)
    ROOT.gPad.SetTopMargin(0.10)
    hr.Draw("COLZ")
    keep.append(hr)
c2.cd(0)
lat = ROOT.TLatex(); lat.SetNDC(); lat.SetTextSize(0.050)
lat.DrawLatex(0.22, 0.065, f"empty (white) = no generated deuterons in bin")
c2.SaveAs("./Plots/rompot_deuteron_acceptance_map_p_vs_theta.pdf")
keep.extend([c2, h2_gen, h2_inc, h2_coin])

# ------------------------------------------------------------------
# Numerical summary in coarse theta slices -- easier to quote than a plot
# ------------------------------------------------------------------
print("RP acceptance in theta_beam slices (deuterons):")
print(f"  {'theta [mrad]':<18}{'N_gen':>9}{'A_incl':>10}{'A_coin':>10}{'coin/incl':>11}")
edges = np.linspace(th_lo, th_hi, 9)
for lo, hi in zip(edges[:-1], edges[1:]):
    sl = g & (theta_bm >= lo) & (theta_bm < hi)
    ng = int(sl.sum())
    if ng == 0:
        continue
    ni = int((sl & rp_incl).sum())
    nc = int((sl & rp_coin).sum())
    print(f"  {lo:6.2f} - {hi:6.2f}  {ng:>9d}{ni/ng:>10.4f}{nc/ng:>10.4f}"
          f"{(nc/ni if ni else 0.0):>11.4f}")
print()
