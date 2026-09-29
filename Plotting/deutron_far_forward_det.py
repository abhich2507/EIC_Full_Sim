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

# ------------------------------------------------------------------
# MC truth
# ------------------------------------------------------------------
mc_pdg    = tree["MCParticles.PDG"].array()
mc_status = tree["MCParticles.generatorStatus"].array()

# Generated (beam/primary) deuterons. generatorStatus==1 -> stable final-state
# particle from the generator. Drop the status cut if your sample tags the
# beam remnant differently.
mc_is_gen_d = (mc_pdg == DEUTERON) & (mc_status == 1)

# Global flat index bookkeeping: lets us deduplicate hits belonging to the
# same MC particle and map "was hit" back onto the MCParticles collection.
n_mc       = ak.to_numpy(ak.num(mc_pdg))
offsets    = np.concatenate([[0], np.cumsum(n_mc)])[:-1]
n_mc_total = int(n_mc.sum())

flat_is_gen_d = ak.to_numpy(ak.flatten(mc_is_gen_d))
n_gen_d       = int(flat_is_gen_d.sum())

# ------------------------------------------------------------------
# Association validation
#
# Three assumptions are made when we do mc_pdg[assoc_index]; none of them are
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
# ------------------------------------------------------------------
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


def accepted_fraction(hit_coll, assoc_coll, extra_hit_mask=None):
    """Per-particle acceptance: unique generated deuterons with >=1 hit."""
    _, d_mask, valid, safe = hit_pdg_and_mask(hit_coll, assoc_coll)
    sel = d_mask if extra_hit_mask is None else (d_mask & extra_hit_mask)

    global_idx = ak.to_numpy(ak.flatten(safe[sel] + offsets))
    was_hit = np.zeros(n_mc_total, dtype=bool)
    if len(global_idx):
        was_hit[np.unique(global_idx)] = True

    n_acc = int((was_hit & flat_is_gen_d).sum())
    return n_acc, was_hit


def get_xyz(hit_coll, mask):
    x = tree[f"{hit_coll}.position.x"].array()[mask]
    y = tree[f"{hit_coll}.position.y"].array()[mask]
    z = tree[f"{hit_coll}.position.z"].array()[mask]
    return x, y, z


# ------------------------------------------------------------------
# Roman Pots
# ------------------------------------------------------------------
rp_pdg, rp_d_mask, _, _ = hit_pdg_and_mask("ForwardRomanPotHits",
                                           "_ForwardRomanPotHits_particle")
rp_x, rp_y, rp_z = get_xyz("ForwardRomanPotHits", rp_d_mask)

# ------------------------------------------------------------------
# Off-Momentum Detector
# ------------------------------------------------------------------
om_pdg, om_d_mask, _, _ = hit_pdg_and_mask("ForwardOffMTrackerHits",
                                           "_ForwardOffMTrackerHits_particle")
om_x, om_y, om_z = get_xyz("ForwardOffMTrackerHits", om_d_mask)

# ------------------------------------------------------------------
# B0 Tracker
# ------------------------------------------------------------------
b0_pdg, b0_d_mask, _, _ = hit_pdg_and_mask("B0TrackerHits",
                                           "_B0TrackerHits_particle")
b0_x, b0_y, b0_z = get_xyz("B0TrackerHits", b0_d_mask)

# ------------------------------------------------------------------
# Acceptance
# ------------------------------------------------------------------
n_rp, _ = accepted_fraction("ForwardRomanPotHits",   "_ForwardRomanPotHits_particle")
n_om, _ = accepted_fraction("ForwardOffMTrackerHits", "_ForwardOffMTrackerHits_particle")
n_b0, _ = accepted_fraction("B0TrackerHits",          "_B0TrackerHits_particle")

def binom_err(k, n):
    if n == 0:
        return 0.0
    p = k / n
    return np.sqrt(max(p * (1 - p), 0.0) / n)

print()
print(f"Generated deuterons                : {n_gen_d}")
for label, k in (("B0 Tracker", n_b0),
                 ("Off-Momentum Detector", n_om),
                 ("Roman Pots", n_rp)):
    a = k / n_gen_d if n_gen_d else 0.0
    print(f"{label:<34}: {k:>8d}   A = {a:.5f} +/- {binom_err(k, n_gen_d):.5f}")

# Deuteron purity of the hit samples (sanity check on the PDG selection)
for label, dm in (("B0", b0_d_mask), ("OMD", om_d_mask), ("Roman Pots", rp_d_mask)):
    tot = int(ak.sum(ak.num(dm)))
    sel = int(ak.sum(dm))
    print(f"  {label} deuteron hit purity: {sel}/{tot}"
          f" = {sel/tot:.3f}" if tot else f"  {label}: no hits")

# ------------------------------------------------------------------
# Station splitting (deuteron hits only)
# ------------------------------------------------------------------
rp_z1_mask = rp_z < 32600
rp_z2_mask = rp_z > 34200

om_z1_mask = om_z < 25600
om_z2_mask = om_z > 26800

b0_z1_mask = (b0_z < 6000)
b0_z2_mask = (b0_z > 6100) & (b0_z < 6300)
b0_z3_mask = (b0_z > 6400) & (b0_z < 6500)
b0_z4_mask = (b0_z > 6600)


def auto_bins(x_arr, y_arr, masks, nbins=100, pad=0.08):
    """Common x/y binning across a set of station masks, padded around data."""
    xs, ys = [], []
    for m in masks:
        xs.append(ak.to_numpy(ak.flatten(x_arr[m])))
        ys.append(ak.to_numpy(ak.flatten(y_arr[m])))
    xs = np.concatenate(xs) if xs else np.array([])
    ys = np.concatenate(ys) if ys else np.array([])
    if len(xs) == 0:
        return (nbins, -1.0, 1.0), (nbins, -1.0, 1.0)

    def rng(v):
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
        d = (hi - lo) * pad
        return (nbins, lo - d, hi + d)
    return rng(xs), rng(ys)


def fill_histogram(name, title, x_arr, y_arr, mask, x_bins, y_bins):
    x_vals = ak.to_numpy(ak.flatten(x_arr[mask])).astype(np.float64)
    y_vals = ak.to_numpy(ak.flatten(y_arr[mask])).astype(np.float64)

    hist = ROOT.TH2F(name, title,
                     x_bins[0], x_bins[1], x_bins[2],
                     y_bins[0], y_bins[1], y_bins[2])
    if len(x_vals):
        hist.FillN(len(x_vals), x_vals, y_vals,
                   np.ones(len(x_vals), dtype=np.float64))
    hist.GetXaxis().SetTitle("X Position [mm]")
    hist.GetYaxis().SetTitle("Y Position [mm]")
    hist.GetYaxis().SetTitleOffset(1.2)
    print(f"  {name}: {len(x_vals)} deuteron hits, "
          f"x=[{x_vals.min():.1f},{x_vals.max():.1f}] "
          f"y=[{y_vals.min():.1f},{y_vals.max():.1f}]"
          if len(x_vals) else f"  {name}: empty")
    return hist


h_rp1 = fill_histogram("h_rp1", "Deuterons: Roman Pot Station 1 (z #approx 32.5 m)",
                       rp_x, rp_y, rp_z1_mask, (100, -1300, -950), (100, -90, 90))
h_rp2 = fill_histogram("h_rp2", "Deuterons: Roman Pot Station 2 (z #approx 34.3 m)",
                       rp_x, rp_y, rp_z2_mask, (100, -1400, -1050), (100, -90, 90))
h_om1 = fill_histogram("h_om1", "Deuterons: OMD Station 1 (z #approx 25.6 m)",
                       om_x, om_y, om_z1_mask, (100, -1000, -850), (100, -100, 100))
h_om2 = fill_histogram("h_om2", "Deuterons: OMD Station 2 (z #approx 26.9 m)",
                       om_x, om_y, om_z2_mask, (100, -1050, -940), (100, -100, 100))

# B0 sits on a different part of the rotated hadron beamline, so its x-offset is
# nothing like RP/OMD -> derive the axis ranges from the data itself.
b0_masks = (b0_z1_mask, b0_z2_mask, b0_z3_mask, b0_z4_mask)
b0_xb, b0_yb = auto_bins(b0_x, b0_y, b0_masks, nbins=100)

h_b0 = [
    fill_histogram(f"h_b0{i+1}",
                   f"Deuterons: B0 Layer {i+1} (z #approx {z:.2f} m)",
                   b0_x, b0_y, m, b0_xb, b0_yb)
    for i, (m, z) in enumerate(zip(b0_masks, (5.9, 6.2, 6.45, 6.7)))
]


def draw_grid(cname, ctitle, hists, ncol, nrow, acc, outfile, width=1400, height=600):
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
    lat.DrawLatex(0.34, 0.065 if nrow == 1 else 0.015,
                  f"deuteron acceptance = {acc:.5f}  (N_{{gen}} = {n_gen_d})")
    c.SaveAs(outfile)
    return c


c_rp = draw_grid("canvas_rp", "Roman Pot Deuteron Hits", [h_rp1, h_rp2], 2, 1,
                 n_rp / n_gen_d if n_gen_d else 0.0,
                 "./Plots/rompot_deuteron_hits_xy_both_stations.pdf")
c_om = draw_grid("canvas_om", "OMD Deuteron Hits", [h_om1, h_om2], 2, 1,
                 n_om / n_gen_d if n_gen_d else 0.0,
                 "./Plots/offm_deuteron_hits_xy_both_stations.pdf")
c_b0 = draw_grid("canvas_b0", "B0 Tracker Deuteron Hits", h_b0, 2, 2,
                 n_b0 / n_gen_d if n_gen_d else 0.0,
                 "./Plots/b0tracker_deuteron_hits_xy_all_layers.pdf",
                 width=1400, height=1200)
