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

# 1. Create an empty list to hold all the file paths
file_list = []

# 2. Loop from 0 to 9
for i in range(10):
    filename = f"/work/eic/users/aabhishe/sim_out_1M/EIC_3He_5x41_XRot_pi_bc_sim_chunk_{i}.root:events"
    file_list.append(filename)

print(f"Loading {len(file_list)} files...")

# 3. Load them all into one massive data object
tree = uproot.concatenate(file_list)

# ==================================================================
# ONE selection switch controls the denominator, the numerator AND the
# histograms, so a panel can never show hits the printed number excluded.
# ==================================================================
SELECTION_MODE = "generated"     # "generated" | "all_deuterons"
GEN_STATUS     = 1

# Beam / frame constants for the RP differential block
CROSSING_ANGLE = 0.025           # rad
ROT_SIGN       = 1.0             # flip to +1.0 if theta_beam comes out ~25 mrad
P_BEAM_D       = 82.0            # GeV, = 41 GeV/nucleon * A=2

# ==================================================================
# MC truth
# ==================================================================
mc_pdg    = tree["MCParticles.PDG"]
mc_status = tree["MCParticles.generatorStatus"]

if SELECTION_MODE == "generated":
    mc_is_sig = (mc_pdg == DEUTERON) & (mc_status == GEN_STATUS)
    SIG_LABEL = f"generated d (status=={GEN_STATUS})"
elif SELECTION_MODE == "all_deuterons":
    mc_is_sig = (mc_pdg == DEUTERON)
    SIG_LABEL = "all d (any status)"
else:
    raise ValueError(f"unknown SELECTION_MODE {SELECTION_MODE!r}")

n_mc       = ak.to_numpy(ak.num(mc_pdg))
offsets    = np.concatenate([[0], np.cumsum(n_mc)])[:-1]
n_mc_total = int(n_mc.sum())

flat_is_sig = ak.to_numpy(ak.flatten(mc_is_sig))
n_gen_d     = int(flat_is_sig.sum())

# ------------------------------------------------------------------
# generatorStatus audit -- every acceptance divides by n_gen_d
# ------------------------------------------------------------------
_d_status = ak.to_numpy(ak.flatten(mc_status[mc_pdg == DEUTERON]))
print(f"[status] generatorStatus of all PDG {DEUTERON} in MCParticles:")
if len(_d_status):
    for s, c in zip(*np.unique(_d_status, return_counts=True)):
        note = "  (Geant4-produced secondary)" if s == 0 else ""
        if SELECTION_MODE == "generated" and s == GEN_STATUS:
            note = "  <-- used as denominator"
        print(f"         status={int(s):<4d} n={int(c):<8d}{note}")
else:
    print("         NONE FOUND - wrong PDG or wrong sample.")
print(f"[status] selection = {SELECTION_MODE!r} -> N_gen = {n_gen_d}\n")

# Jagged mirror of flat_is_sig, aligned to MCParticles.
mc_is_sig_jag = ak.unflatten(flat_is_sig, n_mc)


# ==================================================================
# Association validation
# ==================================================================
def validate_assoc(hit_coll, assoc_coll, verbose=True):
    """Check hit<->MC association integrity. Returns (idx, valid, safe)."""
    idx    = tree[f"{assoc_coll}.index"]
    n_hits = ak.num(tree[f"{hit_coll}.position.x"])
    n_assc = ak.num(idx)

    n_bad_len = int(ak.sum(n_hits != n_assc))
    if n_bad_len:
        raise RuntimeError(
            f"{assoc_coll}: {n_bad_len} events where the association count "
            f"differs from the hit count -> the relation is NOT index-parallel "
            f"to {hit_coll}. Boolean-masking positions with a per-hit mask "
            f"is invalid for this collection."
        )

    in_range = (idx >= 0) & (idx < n_mc)
    n_neg    = int(ak.sum(idx < 0))
    n_over   = int(ak.sum(idx >= n_mc))
    safe     = ak.where(in_range, idx, 0)

    if verbose:
        tot = int(ak.sum(n_assc))
        print(f"[assoc] {hit_coll:<22} hits={tot:<8d} "
              f"unlinked(-1)={n_neg:<6d} out-of-range={n_over:<6d} "
              f"usable={int(ak.sum(in_range)):d}")
        if n_over:
            print(f"        WARNING: {n_over} indices point outside MCParticles "
                  f"of their own event (cross-frame relation?). Dropped.")
    return idx, in_range, safe


def hit_masks(hit_coll, assoc_coll):
    """Per-hit signal mask (PDG + status), used by BOTH acceptance and plots."""
    _, valid, safe = validate_assoc(hit_coll, assoc_coll)

    pdg      = ak.where(valid, mc_pdg[safe], 0)
    pdg_mask = (pdg == DEUTERON) & valid
    sig_mask = mc_is_sig_jag[safe] & valid

    n_pdg, n_sig = int(ak.sum(pdg_mask)), int(ak.sum(sig_mask))
    print(f"        deuteron hits: PDG-only={n_pdg:<6d} "
          f"passing '{SIG_LABEL}'={n_sig:<6d} "
          f"removed by status cut={n_pdg - n_sig:d}")
    if n_pdg and not n_sig:
        print(f"        *** all {n_pdg} deuteron hits here are linked to "
              f"particles OUTSIDE the denominator -> acceptance is exactly 0 "
              f"and the XY panels will be EMPTY (consistent, by design).")

    return sig_mask, pdg_mask, safe


def get_xyz(hit_coll, mask):
    x = tree[f"{hit_coll}.position.x"][mask]
    y = tree[f"{hit_coll}.position.y"][mask]
    z = tree[f"{hit_coll}.position.z"][mask]
    return x, y, z


# ==================================================================
# Station definitions -- ONE source of truth for the z-window AND the
# drawing range.  Entry = (label, z_lo, z_hi, xbins, ybins)
#   z_lo / z_hi : None = open ended
#   xbins/ybins : (nbins, lo, hi), or None = auto-range from the data
# ==================================================================
STATIONS = {
    "RP": [
        ("Station 1 (z #approx 32.5 m)", None,  32600.,
         (100, -1300., -950.), (100, -90., 90.)),
        ("Station 2 (z #approx 34.3 m)", 34200., None,
         (100, -1400., -1050.), (100, -90., 90.)),
    ],
    "OMD": [
        ("Station 1 (z #approx 25.6 m)", None,  25600.,
         (100, -1000., -850.), (100, -100., 100.)),
        ("Station 2 (z #approx 26.9 m)", 26800., None,
         (100, -1050., -940.), (100, -100., 100.)),
    ],
    # B0 ranges were never specified -> auto-range, shared across the 4 layers
    # so the panels stay comparable to one another.
    "B0": [
        ("Layer 1 (z < 6.00 m)",  None,  6000., None, None),
        ("Layer 2 (6.10-6.30 m)", 6100., 6300., None, None),
        ("Layer 3 (6.40-6.50 m)", 6400., 6500., None, None),
        ("Layer 4 (z > 6.60 m)",  6600., None,  None, None),
    ],
}

# Detectors whose auto-ranged panels should share ONE common range.
SHARED_AUTO_RANGE = {"B0"}

N_REQUIRED = {"RP": 2, "OMD": 2, "B0": 3}

COLLS = {
    "B0":  ("B0TrackerHits",          "_B0TrackerHits_particle"),
    "OMD": ("ForwardOffMTrackerHits", "_ForwardOffMTrackerHits_particle"),
    "RP":  ("ForwardRomanPotHits",    "_ForwardRomanPotHits_particle"),
}


def zmask(z, lo, hi):
    if lo is None:
        return z < hi
    if hi is None:
        return z > lo
    return (z > lo) & (z < hi)


def particles_with_hit(safe, sel):
    """Flat bool over MCParticles: particle produced >=1 selected hit."""
    g = ak.to_numpy(ak.flatten(safe[sel] + offsets))
    out = np.zeros(n_mc_total, dtype=bool)
    if len(g):
        out[np.unique(g)] = True
    return out


# ==================================================================
# Per-detector acceptance with in-detector coincidence
# ==================================================================
def detector_acceptance(key):
    hit_coll, assoc_coll = COLLS[key]
    sig_mask, _, safe = hit_masks(hit_coll, assoc_coll)
    z = tree[f"{hit_coll}.position.z"]

    bits, in_any = [], None
    for (label, lo, hi, _xb, _yb) in STATIONS[key]:
        zm  = zmask(z, lo, hi)
        sel = sig_mask & zm
        bits.append(particles_with_hit(safe, sel))
        in_any = zm if in_any is None else (in_any | zm)

    n_outside = int(ak.sum(sig_mask & ~in_any))

    n_fired    = np.sum(np.stack(bits), axis=0)
    incl_bits  = (n_fired >= 1) & flat_is_sig
    coin_bits  = (n_fired >= N_REQUIRED[key]) & flat_is_sig

    return {
        "bits": bits, "incl": incl_bits, "coin": coin_bits,
        "n_outside": n_outside, "sig_mask": sig_mask, "safe": safe,
    }


def binom_err(k, n):
    if n == 0:
        return 0.0
    p = k / n
    return np.sqrt(max(p * (1 - p), 0.0) / n)


print()
RES = {}
for key in ("B0", "OMD", "RP"):
    RES[key] = detector_acceptance(key)

print()
for key in ("B0", "OMD", "RP"):
    r = RES[key]
    n_i, n_c = int(r["incl"].sum()), int(r["coin"].sum())
    a_i = n_i / n_gen_d if n_gen_d else 0.0
    a_c = n_c / n_gen_d if n_gen_d else 0.0
    print(f"--- {key} ---  (N_gen = {n_gen_d}, '{SIG_LABEL}')")
    for (label, *_), b in zip(STATIONS[key], r["bits"]):
        n_s = int((b & flat_is_sig).sum())
        print(f"    {label:<32} n={n_s:<7d} A={n_s/n_gen_d if n_gen_d else 0:.5f}")
    print(f"    inclusive (>=1 station)          n={n_i:<7d} "
          f"A={a_i:.5f} +/- {binom_err(n_i, n_gen_d):.5f}")
    print(f"    coincidence ({N_REQUIRED[key]}-of-{len(STATIONS[key])})"
          f"{'':<15} n={n_c:<7d} A={a_c:.5f} +/- {binom_err(n_c, n_gen_d):.5f}")
    print(f"    coin/incl = {n_c/n_i if n_i else 0.0:.4f}   "
          f"deuteron hits outside all z-windows = {r['n_outside']}\n")

# Cross-detector correlations (never add acceptances -- AND/OR the bits)
print("--- cross-detector (coincidence bits) ---")
for a, b in (("B0", "OMD"), ("B0", "RP"), ("OMD", "RP")):
    n_and = int((RES[a]["coin"] & RES[b]["coin"]).sum())
    print(f"    {a} AND {b:<4} n={n_and:<7d} "
          f"A={n_and/n_gen_d if n_gen_d else 0:.5f}")
n_or = int((RES["B0"]["coin"] | RES["OMD"]["coin"] | RES["RP"]["coin"]).sum())
print(f"    ANY of the three  n={n_or:<7d} "
      f"A={n_or/n_gen_d if n_gen_d else 0:.5f}\n")


# ==================================================================
# XY maps -- same sig_mask as the acceptance, per-station axis ranges
# ==================================================================
def auto_range(vals, nbins=100, pad_frac=0.10, fallback=(-1.0, 1.0)):
    if not len(vals):
        return (nbins, fallback[0], fallback[1])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-6:
        lo, hi = lo - 1.0, hi + 1.0
    pad = pad_frac * (hi - lo)
    return (nbins, lo - pad, hi + pad)


def station_hists(key):
    hit_coll, _ = COLLS[key]
    r = RES[key]
    x, y, z = get_xyz(hit_coll, r["sig_mask"])

    # collect values first so auto-ranging can see the data
    vals = []
    for (label, lo, hi, xb, yb) in STATIONS[key]:
        m  = zmask(z, lo, hi)
        xv = ak.to_numpy(ak.flatten(x[m])).astype(np.float64)
        yv = ak.to_numpy(ak.flatten(y[m])).astype(np.float64)
        vals.append((label, xv, yv, xb, yb))

    shared_x = shared_y = None
    if key in SHARED_AUTO_RANGE:
        ax = np.concatenate([v[1] for v in vals]) if vals else np.array([])
        ay = np.concatenate([v[2] for v in vals]) if vals else np.array([])
        shared_x, shared_y = auto_range(ax), auto_range(ay)

    hists = []
    for i, (label, xv, yv, xb, yb) in enumerate(vals):
        # priority: explicit per-station range -> shared auto -> per-station auto
        xr = xb if xb is not None else (shared_x or auto_range(xv))
        yr = yb if yb is not None else (shared_y or auto_range(yv))

        name = f"h_{key.lower()}{i+1}"
        h = ROOT.TH2F(name, f"Deuterons: {key} {label}",
                      xr[0], xr[1], xr[2], yr[0], yr[1], yr[2])
        if len(xv):
            h.FillN(len(xv), xv, yv, np.ones(len(xv), dtype=np.float64))

        h.GetXaxis().SetTitle("X Position [mm]")
        h.GetYaxis().SetTitle("Y Position [mm]")
        h.GetYaxis().SetTitleOffset(1.2)
        # stop the tick labels from colliding on the narrow x-ranges
        h.GetXaxis().SetNdivisions(505)
        h.GetXaxis().SetLabelSize(0.035)
        h.GetYaxis().SetLabelSize(0.035)

        if len(xv):
            print(f"  {name}: {len(xv)} hits  "
                  f"x=[{xv.min():.1f},{xv.max():.1f}] "
                  f"y=[{yv.min():.1f},{yv.max():.1f}]  "
                  f"axis x=[{xr[1]:.0f},{xr[2]:.0f}]")
        else:
            print(f"  {name}: empty")
        hists.append(h)
    return hists


def draw_grid(key, hists, ncol, nrow, size, outfile):
    r = RES[key]
    n_i, n_c = int(r["incl"].sum()), int(r["coin"].sum())
    a_i = n_i / n_gen_d if n_gen_d else 0.0
    a_c = n_c / n_gen_d if n_gen_d else 0.0

    c = ROOT.TCanvas(f"canvas_{key.lower()}", f"{key} Deuteron Hits", *size)
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
    lat.SetNDC()
    lat.SetTextSize(0.050 if nrow == 1 else 0.025)
    lat.DrawLatex(0.20, 0.035,
                  f"deuteron A_{{{N_REQUIRED[key]}/{len(hists)}}} = {a_c:.5f}"
                  f"   (incl. {a_i:.5f}, N_{{gen}} = {n_gen_d})")
    c.SaveAs(outfile)
    return c


print("\n[plots]")
c_rp = draw_grid("RP",  station_hists("RP"),  2, 1, (1400, 600),
                 "./Plots/rompot_deuteron_hits_xy_both_stations.pdf")
c_om = draw_grid("OMD", station_hists("OMD"), 2, 1, (1400, 600),
                 "./Plots/offm_deuteron_hits_xy_both_stations.pdf")
c_b0 = draw_grid("B0",  station_hists("B0"),  2, 2, (1400, 1200),
                 "./Plots/b0_deuteron_hits_xy_all_layers.pdf")


# ==================================================================
# Deuteron Endpoints (2D XY and 1D Z)
# ==================================================================
print("\n[plots] Generating deuteron endpoint distributions...")

# Extract endpoints and apply the signal mask immediately (REMOVED .array())
end_x = ak.to_numpy(ak.flatten(tree["MCParticles.endpoint.x"]))[flat_is_sig].astype(np.float64)
end_y = ak.to_numpy(ak.flatten(tree["MCParticles.endpoint.y"]))[flat_is_sig].astype(np.float64)
end_z = ak.to_numpy(ak.flatten(tree["MCParticles.endpoint.z"]))[flat_is_sig].astype(np.float64)

c_end = ROOT.TCanvas("c_end", "Deuteron Endpoints", 1400, 600)
c_end.Divide(2, 1)

# 1. XY 2D Endpoint (Left)
c_end.cd(1)
ROOT.gPad.SetRightMargin(0.15)  # Make room for the Z-axis color bar
h_end_xy = ROOT.TH2F("h_end_xy", "Deuteron Endpoint XY;x [mm];y [mm]", 
                     100, -1300.0, -900.0, 
                     100, -100.0, 100.0)
if len(end_x) and len(end_y):
    h_end_xy.FillN(len(end_x), end_x, end_y, np.ones(len(end_x), dtype=np.float64))
h_end_xy.Draw("COLZ")

# 2. Z Endpoint (Right)
c_end.cd(2)
ROOT.gPad.SetLogy()
h_end_z = ROOT.TH1F("h_end_z", "Deuteron Endpoint Z;z [mm];Count", 100, 25000.0, 40000.0)
h_end_z.SetLineColor(ROOT.kAzure + 2)
h_end_z.SetFillColorAlpha(ROOT.kAzure + 2, 0.35)
if len(end_z):
    h_end_z.FillN(len(end_z), end_z, np.ones(len(end_z), dtype=np.float64))
h_end_z.Draw("HIST")

# Force ROOT to draw the canvas in memory to calculate the Y-axis limits
ROOT.gPad.Update()
y_min = ROOT.gPad.GetUymin()
y_max = ROOT.gPad.GetUymax()

# Convert back from log scale
if ROOT.gPad.GetLogy():
    y_min = 10**y_min
    y_max = 10**y_max

# Draw the vertical line at z = 32546.2
z_line = ROOT.TLine(32546.2, y_min, 32546.2, y_max)
z_line.SetLineColor(ROOT.kRed)
z_line.SetLineStyle(2)
z_line.SetLineWidth(2)
z_line.Draw("SAME")

c_end.SaveAs("./Plots/deuteron_endpoints_XY_Z.pdf")


# ==================================================================
# RP differential acceptance vs p, theta, xL
# ==================================================================
# REMOVED .array()
px = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.x"]))
py = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.y"]))
pz = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.z"]))

p_tot   = np.sqrt(px**2 + py**2 + pz**2)
th_lab  = np.arctan2(np.hypot(px, py), pz) * 1e3        # mrad

ca, sa  = np.cos(ROT_SIGN * CROSSING_ANGLE), np.sin(ROT_SIGN * CROSSING_ANGLE)
pxb     =  ca * px + sa * pz
pzb     = -sa * px + ca * pz
th_beam = np.arctan2(np.hypot(pxb, py), pzb) * 1e3      # mrad
xL      = p_tot / P_BEAM_D

_g = flat_is_sig
if _g.sum():
    print(f"\n[frame] median theta_lab = {np.median(th_lab[_g]):.2f} mrad, "
          f"median theta_beam = {np.median(th_beam[_g]):.2f} mrad "
          f"(the beam-frame one should be the SMALL one; else flip ROT_SIGN)")
    print(f"[frame] median p = {np.median(p_tot[_g]):.2f} GeV, "
          f"median xL = {np.median(xL[_g]):.3f}")

rp_incl, rp_coin = RES["RP"]["incl"], RES["RP"]["coin"]

VARS = [
    ("p",     p_tot,   "p [GeV]",               60, 0.0, 100.0),
    ("theta", th_beam, "#theta_{beam} [mrad]",  60, 0.0,  5.0),
    ("xL",    xL,      "x_{L} = p / p_{beam}",  60, 0.0,   1.3),
]

c_diff = ROOT.TCanvas("c_diff", "RP deuteron acceptance", 1800, 1000)
c_diff.Divide(3, 2)
keep = []

for i, (vname, arr, axt, nb, lo, hi) in enumerate(VARS):
    h_gen = ROOT.TH1F(f"hg_{vname}", f";{axt};deuterons", nb, lo, hi)
    h_inc = ROOT.TH1F(f"hi_{vname}", f";{axt};deuterons", nb, lo, hi)
    h_coi = ROOT.TH1F(f"hc_{vname}", f";{axt};deuterons", nb, lo, hi)
    for h, m in ((h_gen, _g), (h_inc, rp_incl), (h_coi, rp_coin)):
        v = arr[m].astype(np.float64)
        if len(v):
            h.FillN(len(v), v, np.ones(len(v), dtype=np.float64))

    h_gen.SetFillColorAlpha(ROOT.kGray + 1, 0.35)
    h_gen.SetLineColor(ROOT.kGray + 2)
    h_inc.SetLineColor(ROOT.kAzure + 2);  h_inc.SetLineWidth(2)
    h_coi.SetLineColor(ROOT.kRed + 1);    h_coi.SetLineWidth(2)

    c_diff.cd(i + 1)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLeftMargin(0.13)
    h_gen.SetMaximum(max(h_gen.GetMaximum() * 5.0, 10.0))
    h_gen.SetMinimum(0.5)
    h_gen.Draw("HIST")
    h_inc.Draw("HIST SAME")
    h_coi.Draw("HIST SAME")

    leg = ROOT.TLegend(0.55, 0.70, 0.88, 0.88)
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.035)
    leg.AddEntry(h_gen, "generated", "f")
    leg.AddEntry(h_inc, "RP inclusive", "l")
    leg.AddEntry(h_coi, "RP 2-of-2", "l")
    leg.Draw()

    c_diff.cd(i + 4)
    ROOT.gPad.SetLeftMargin(0.13)
    frame = ROOT.TH1F(f"fr_{vname}", f";{axt};acceptance", nb, lo, hi)
    frame.SetMinimum(0.0); frame.SetMaximum(1.15)
    frame.Draw()
    eff_i = ROOT.TEfficiency(h_inc, h_gen)
    eff_c = ROOT.TEfficiency(h_coi, h_gen)
    eff_i.SetLineColor(ROOT.kAzure + 2); eff_i.SetMarkerColor(ROOT.kAzure + 2)
    eff_i.SetMarkerStyle(20); eff_i.SetMarkerSize(0.7)
    eff_c.SetLineColor(ROOT.kRed + 1);   eff_c.SetMarkerColor(ROOT.kRed + 1)
    eff_c.SetMarkerStyle(21); eff_c.SetMarkerSize(0.7)
    eff_i.Draw("SAME P")
    eff_c.Draw("SAME P")

    leg2 = ROOT.TLegend(0.55, 0.20, 0.88, 0.38)
    leg2.SetBorderSize(0); leg2.SetFillStyle(0); leg2.SetTextSize(0.035)
    leg2.AddEntry(eff_i, "inclusive", "lp")
    leg2.AddEntry(eff_c, "2-of-2", "lp")
    leg2.Draw()

    keep += [h_gen, h_inc, h_coi, frame, eff_i, eff_c, leg, leg2]

c_diff.SaveAs("./Plots/rompot_deuteron_acceptance_vs_p_theta_xL.pdf")


# 2D acceptance map: theta vs p
def acc_map(name, title, num_mask):
    hn = ROOT.TH2F(f"n_{name}", title, 40, -1.5, 5.0, 40, 70.0, 95.0)
    hd = ROOT.TH2F(f"d_{name}", title, 40, -1.5, 5.0, 40, 70.0, 95.0)
    for h, m in ((hn, num_mask), (hd, _g)):
        tv, pv = th_beam[m].astype(np.float64), p_tot[m].astype(np.float64)
        if len(tv):
            h.FillN(len(tv), tv, pv, np.ones(len(tv), dtype=np.float64))
    hn.Divide(hd)
    # empty denominator -> white, so "never tested" != "acceptance 0"
    for bx in range(1, hn.GetNbinsX() + 1):
        for by in range(1, hn.GetNbinsY() + 1):
            if hd.GetBinContent(bx, by) == 0:
                hn.SetBinContent(bx, by, -1.0)
    hn.SetMinimum(-0.001); hn.SetMaximum(1.0)
    hn.GetXaxis().SetTitle("#theta_{beam} [mrad]")
    hn.GetYaxis().SetTitle("p [GeV]")
    hn.GetYaxis().SetTitleOffset(1.3)
    return hn, hd


c_map = ROOT.TCanvas("c_map", "RP acceptance map", 1400, 600)
c_map.Divide(2, 1)
m_i, d_i = acc_map("incl", "RP deuteron acceptance (inclusive)", rp_incl)
m_c, d_c = acc_map("coin", "RP deuteron acceptance (2-of-2)",    rp_coin)
for j, h in enumerate((m_i, m_c), start=1):
    c_map.cd(j)
    ROOT.gPad.SetRightMargin(0.16)
    ROOT.gPad.SetLeftMargin(0.14)
    h.Draw("COLZ")
c_map.SaveAs("./Plots/rompot_deuteron_acceptance_map_p_vs_theta.pdf")

# Tabulated acceptance in theta slices
print("\n[RP acceptance vs theta_beam]")
print(f"  {'theta [mrad]':<16}{'N_gen':>8}{'incl':>10}{'2-of-2':>10}"
      f"{'A_incl':>10}{'A_coin':>10}{'coin/incl':>11}")
edges = np.linspace(0.0, 10.0, 9)
for k in range(len(edges) - 1):
    sl  = _g & (th_beam >= edges[k]) & (th_beam < edges[k + 1])
    ng  = int(sl.sum())
    ni  = int((sl & rp_incl).sum())
    nc  = int((sl & rp_coin).sum())
    if not ng:
        continue
    print(f"  {edges[k]:5.1f} {edges[k+1]:5.1f}{ng:>8d}{ni:>10d}{nc:>10d}"
          f"{ni/ng:>10.4f}{nc/ng:>10.4f}{(nc/ni if ni else 0.0):>11.4f}")


# ==================================================================
# Missing Deuterons Investigation
# ==================================================================
print("\n[investigation] Analyzing missing deuterons...")

# Isolate the Missing Particles (generated but failed RP coincidence)
missing_mask = _g & ~rp_coin
n_missing = int(missing_mask.sum())
print(f"  Investigating {n_missing} missing deuterons...")

if n_missing > 0:
    # Extract variables for missing deuterons (REMOVED .array())
    missing_end_z = ak.to_numpy(ak.flatten(tree["MCParticles.endpoint.z"]))[missing_mask].astype(np.float64)
    missing_end_x = ak.to_numpy(ak.flatten(tree["MCParticles.endpoint.x"]))[missing_mask].astype(np.float64)
    missing_th    = th_beam[missing_mask].astype(np.float64)
    missing_p     = p_tot[missing_mask].astype(np.float64)

    # 1. 1D Endpoint Z Plot
    h_missing_z = ROOT.TH1F("h_missing_z", "Endpoint Z of Missing Deuterons;z [mm];Count", 200, 0.0, 40000.0)
    h_missing_z.SetLineColor(ROOT.kRed)
    h_missing_z.SetFillColorAlpha(ROOT.kRed, 0.35)
    
    if len(missing_end_z):
        h_missing_z.FillN(len(missing_end_z), missing_end_z, np.ones(len(missing_end_z), dtype=np.float64))

    c_miss_z = ROOT.TCanvas("c_miss_z", "Missing Z", 800, 600)
    ROOT.gPad.SetLogy()
    h_missing_z.Draw("HIST")
    
    ROOT.gPad.Update()
    y_min_miss = ROOT.gPad.GetUymin()
    y_max_miss = ROOT.gPad.GetUymax()
    if ROOT.gPad.GetLogy():
        y_min_miss = 10**y_min_miss
        y_max_miss = 10**y_max_miss
        
    z_line_miss = ROOT.TLine(32546.2, y_min_miss, 32546.2, y_max_miss)
    z_line_miss.SetLineColor(ROOT.kBlack)
    z_line_miss.SetLineStyle(2)
    z_line_miss.SetLineWidth(2)
    z_line_miss.Draw("SAME")

    c_miss_z.SaveAs("./Plots/missing_deuterons_z.pdf")

    # 2. Correlate Loss with Kinematics (Momentum vs Angle)
    h_missing_kin = ROOT.TH2F("h_missing_kin", "Kinematics of Missing Deuterons;#theta_{beam} [mrad];p [GeV]", 
                              50, 0.0, 6.0, 50, 40.0, 100.0)
    if len(missing_th):
        h_missing_kin.FillN(len(missing_th), missing_th, missing_p, np.ones(len(missing_th), dtype=np.float64))

    c_miss_kin = ROOT.TCanvas("c_miss_kin", "Missing Kinematics", 800, 600)
    ROOT.gPad.SetRightMargin(0.15)
    h_missing_kin.Draw("COLZ")
    c_miss_kin.SaveAs("./Plots/missing_deuterons_kinematics.pdf")

    # 3. NEW: Missing Theta vs Missing Endpoint Z
    h_miss_th_z = ROOT.TH2F("h_miss_th_z", "Missing Deuterons: #theta_{beam} vs Endpoint Z;Endpoint Z [mm];#theta_{beam} [mrad]", 
                            50, 25000.0, 40000.0, 50, 0.0, 5.0)
    if len(missing_end_z) and len(missing_th):
        h_miss_th_z.FillN(len(missing_end_z), missing_end_z, missing_th, np.ones(len(missing_end_z), dtype=np.float64))

    c_miss_th_z = ROOT.TCanvas("c_miss_th_z", "Missing Theta vs Z", 800, 600)
    ROOT.gPad.SetRightMargin(0.15)
    h_miss_th_z.Draw("COLZ")
    c_miss_th_z.SaveAs("./Plots/missing_deuterons_theta_vs_z.pdf")

    # 4. NEW: Missing Endpoint X vs Missing Endpoint Z
    h_miss_x_z = ROOT.TH2F("h_miss_x_z", "Missing Deuterons: Endpoint X vs Endpoint Z;Endpoint Z [mm];Endpoint X [mm]", 
                           50, 25000.0, 40000.0, 50, -1500.0, -500.0)
    if len(missing_end_z) and len(missing_end_x):
        h_miss_x_z.FillN(len(missing_end_z), missing_end_z, missing_end_x, np.ones(len(missing_end_z), dtype=np.float64))

    c_miss_x_z = ROOT.TCanvas("c_miss_x_z", "Missing X vs Z", 800, 600)
    ROOT.gPad.SetRightMargin(0.15)
    ROOT.gPad.SetLeftMargin(0.13)
    h_miss_x_z.Draw("COLZ")
    c_miss_x_z.SaveAs("./Plots/missing_deuterons_x_vs_z.pdf")

else:
    print("  No missing deuterons found! Acceptance is 100%.")