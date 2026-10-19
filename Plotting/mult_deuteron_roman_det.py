import os
import ROOT
import numpy as np
import awkward as ak
import uproot
from pathlib import Path

# ==================================================================
# Setup & Constants
# ==================================================================
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kViridis)
ROOT.TH1.AddDirectory(False)  # CRITICAL: Prevents ROOT from deleting hists when files close
os.makedirs("./Plots", exist_ok=True)

DEUTERON = 1000010020
SELECTION_MODE = "generated"     # "generated" | "all_deuterons"
GEN_STATUS     = 1

CROSSING_ANGLE = 0.025           # rad
ROT_SIGN       = 1.0             
P_BEAM_D       = 82.0            # GeV

if SELECTION_MODE == "generated":
    SIG_LABEL = f"generated d (status=={GEN_STATUS})"
elif SELECTION_MODE == "all_deuterons":
    SIG_LABEL = "all d (any status)"
else:
    raise ValueError(f"unknown SELECTION_MODE {SELECTION_MODE!r}")

# Files to process
dir_path = Path("/work/eic/users/aabhishe/Recon_out_1M")
file_count = len(list(dir_path.glob("recon_chunk_*.root")))
file_list = [f"/work/eic/users/aabhishe/Recon_out_1M/recon_chunk_{i}.root" for i in range(file_count)]
print(f"Found {len(file_list)} files to process in {dir_path}.")

# ==================================================================
# Master Dictionaries & Histograms (Accumulators for all files)
# ==================================================================
STATIONS = {
    "RP": [
        ("Station 1 (z #approx 32.5 m)", None,  32600., (100, -1300., -950.), (100, -90., 90.)),
        ("Station 2 (z #approx 34.3 m)", 34200., None,  (100, -1400., -1050.), (100, -90., 90.)),
    ]
}

N_REQUIRED = {"RP": 2}
COLLS = {
    "RP":  ("ForwardRomanPotHits", "_ForwardRomanPotHits_particle")
}

# Stats accumulators
total_gen_d = 0
total_miss = 0
status_counts = {}

acc_stats = {
    "RP":  {"incl": 0, "coin": 0, "outside": 0, "bits": [0, 0]}
}

edges = np.linspace(0.0, 10.0, 9)
theta_table = {k: {"ng": 0, "ni": 0, "nc": 0} for k in range(len(edges) - 1)}

# --- Initialize ROOT Master Histograms ---
h_xy = {"RP": []}
for key in STATIONS:
    for i, (label, lo, hi, xb, yb) in enumerate(STATIONS[key]):
        h = ROOT.TH2F(f"h_master_{key.lower()}{i+1}", f"Deuterons: {key} {label}", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
        h.GetXaxis().SetTitle("X Position [mm]")
        h.GetYaxis().SetTitle("Y Position [mm]")
        h.GetYaxis().SetTitleOffset(1.2)
        h.GetXaxis().SetNdivisions(505)
        h.GetXaxis().SetLabelSize(0.035)
        h.GetYaxis().SetLabelSize(0.035)
        h_xy[key].append(h)

VARS_DEF = [
    ("p",     "p [GeV]",               60, 0.0, 100.0),
    ("theta", "#theta_{beam} [mrad]",  60, 0.0,  5.0),
    ("xL",    "x_{L} = p / p_{beam}",  60, 0.0,   1.3),
]
h_diff = {}
for vname, axt, nb, lo, hi in VARS_DEF:
    h_diff[vname] = {
        "gen": ROOT.TH1F(f"hg_{vname}", f";{axt};deuterons", nb, lo, hi),
        "inc": ROOT.TH1F(f"hi_{vname}", f";{axt};deuterons", nb, lo, hi),
        "coi": ROOT.TH1F(f"hc_{vname}", f";{axt};deuterons", nb, lo, hi),
    }

h_map_n_incl = ROOT.TH2F("n_incl", "RP deuteron acceptance (inclusive)", 40, -1.5, 5.0, 40, 70.0, 95.0)
h_map_d_incl = ROOT.TH2F("d_incl", "RP deuteron acceptance (inclusive)", 40, -1.5, 5.0, 40, 70.0, 95.0)
h_map_n_coin = ROOT.TH2F("n_coin", "RP deuteron acceptance (2-of-2)", 40, -1.5, 5.0, 40, 70.0, 95.0)
h_map_d_coin = ROOT.TH2F("d_coin", "RP deuteron acceptance (2-of-2)", 40, -1.5, 5.0, 40, 70.0, 95.0)

h_miss_z    = ROOT.TH1F("h_missing_z", "Endpoint Z of Missing Deuterons;z [mm];Count", 200, 0.0, 40000.0)
h_miss_kin  = ROOT.TH2F("h_missing_kin", "Kinematics of Missing Deuterons;#theta_{beam} [mrad];p [GeV]", 50, 0.0, 6.0, 50, 40.0, 100.0)
h_miss_th_z = ROOT.TH2F("h_miss_th_z", "Missing Deuterons: #theta_{beam} vs Endpoint Z;Endpoint Z [mm];#theta_{beam} [mrad]", 50, 25000.0, 40000.0, 50, 0.0, 5.0)
h_miss_x_z  = ROOT.TH2F("h_miss_x_z", "Missing Deuterons: Endpoint X vs Endpoint Z;Endpoint Z [mm];Endpoint X [mm]", 50, 25000.0, 40000.0, 50, -1500.0, -500.0)


# ==================================================================
# Helper Functions (Memory Safe - Passing arrays explicitly)
# ==================================================================
def validate_assoc(tree, hit_coll, assoc_coll, n_mc):
    idx    = tree[f"{assoc_coll}.index"].array()
    n_hits = ak.num(tree[f"{hit_coll}.position.x"].array())
    n_assc = ak.num(idx)
    n_bad_len = int(ak.sum(n_hits != n_assc))
    if n_bad_len:
        raise RuntimeError(f"{assoc_coll}: mismatched association length.")
    in_range = (idx >= 0) & (idx < n_mc)
    safe     = ak.where(in_range, idx, 0)
    return idx, in_range, safe

def hit_masks(tree, hit_coll, assoc_coll, n_mc, mc_pdg, mc_is_sig_jag):
    _, valid, safe = validate_assoc(tree, hit_coll, assoc_coll, n_mc)
    pdg      = ak.where(valid, mc_pdg[safe], 0)
    pdg_mask = (pdg == DEUTERON) & valid
    sig_mask = mc_is_sig_jag[safe] & valid
    return sig_mask, pdg_mask, safe

def get_xyz(tree, hit_coll, mask):
    x = tree[f"{hit_coll}.position.x"].array()[mask]
    y = tree[f"{hit_coll}.position.y"].array()[mask]
    z = tree[f"{hit_coll}.position.z"].array()[mask]
    return x, y, z

def zmask(z, lo, hi):
    if lo is None: return z < hi
    if hi is None: return z > lo
    return (z > lo) & (z < hi)

def particles_with_hit(safe, sel, offsets, n_mc_total):
    g = ak.to_numpy(ak.flatten(safe[sel] + offsets))
    out = np.zeros(n_mc_total, dtype=bool)
    if len(g):
        out[np.unique(g)] = True
    return out

def binom_err(k, n):
    if n == 0: return 0.0
    p = k / n
    return np.sqrt(max(p * (1 - p), 0.0) / n)


# ==================================================================
# MAIN LOOP (File by File execution)
# ==================================================================
print(f"Starting chunked processing for {len(file_list)} files...\n")

for f_idx, filename in enumerate(file_list):
    if not os.path.exists(filename):
        print(f"WARNING: File {filename} not found. Skipping.")
        continue
        
    print(f"Processing File {f_idx + 1}/{len(file_list)}...")
    tree = uproot.open(f"{filename}:events")

    # 1. MC Truth for this chunk
    mc_pdg    = tree["MCParticles.PDG"].array()
    mc_status = tree["MCParticles.generatorStatus"].array()

    if SELECTION_MODE == "generated":
        mc_is_sig = (mc_pdg == DEUTERON) & (mc_status == GEN_STATUS)
    else:
        mc_is_sig = (mc_pdg == DEUTERON)

    n_mc       = ak.to_numpy(ak.num(mc_pdg))
    offsets    = np.concatenate([[0], np.cumsum(n_mc)])[:-1]
    n_mc_total = int(n_mc.sum())
    
    flat_is_sig = ak.to_numpy(ak.flatten(mc_is_sig))
    n_gen_d     = int(flat_is_sig.sum())
    total_gen_d += n_gen_d
    
    mc_is_sig_jag = ak.unflatten(flat_is_sig, n_mc)

    # Accumulate generator statuses
    _d_status = ak.to_numpy(ak.flatten(mc_status[mc_pdg == DEUTERON]))
    if len(_d_status):
        unique_s, counts = np.unique(_d_status, return_counts=True)
        for s, c in zip(unique_s, counts):
            status_counts[s] = status_counts.get(s, 0) + c

    # 2. Extract Hit Masks and Acceptances for this chunk
    RES = {}
    for key in ["RP"]:
        hit_coll, assoc_coll = COLLS[key]
        sig_mask, _, safe = hit_masks(tree, hit_coll, assoc_coll, n_mc, mc_pdg, mc_is_sig_jag)
        z = tree[f"{hit_coll}.position.z"].array()

        bits, in_any = [], None
        for (label, lo, hi, _xb, _yb) in STATIONS[key]:
            zm  = zmask(z, lo, hi)
            sel = sig_mask & zm
            bits.append(particles_with_hit(safe, sel, offsets, n_mc_total))
            in_any = zm if in_any is None else (in_any | zm)

        n_outside = int(ak.sum(sig_mask & ~in_any))
        n_fired    = np.sum(np.stack(bits), axis=0)
        incl_bits  = (n_fired >= 1) & flat_is_sig
        coin_bits  = (n_fired >= N_REQUIRED[key]) & flat_is_sig

        RES[key] = {"bits": bits, "incl": incl_bits, "coin": coin_bits, "n_outside": n_outside, "sig_mask": sig_mask}

        # Update Master Stats
        acc_stats[key]["incl"] += int(incl_bits.sum())
        acc_stats[key]["coin"] += int(coin_bits.sum())
        acc_stats[key]["outside"] += n_outside
        for i, b in enumerate(bits):
            acc_stats[key]["bits"][i] += int((b & flat_is_sig).sum())

    # 3. Fill XY Station Histograms
    for key in ["RP"]:
        x, y, z = get_xyz(tree, COLLS[key][0], RES[key]["sig_mask"])
        for i, (label, lo, hi, xb, yb) in enumerate(STATIONS[key]):
            m = zmask(z, lo, hi)
            xv = ak.to_numpy(ak.flatten(x[m])).astype(np.float64)
            yv = ak.to_numpy(ak.flatten(y[m])).astype(np.float64)
            if len(xv):
                h_xy[key][i].FillN(len(xv), xv, yv, np.ones(len(xv), dtype=np.float64))

    # 4. Fill Differential Acceptance Histograms
    px = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.x"].array()))
    py = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.y"].array()))
    pz = ak.to_numpy(ak.flatten(tree["MCParticles.momentum.z"].array()))

    p_tot   = np.sqrt(px**2 + py**2 + pz**2)
    ca, sa  = np.cos(ROT_SIGN * CROSSING_ANGLE), np.sin(ROT_SIGN * CROSSING_ANGLE)
    pxb     =  ca * px + sa * pz
    pzb     = -sa * px + ca * pz
    th_beam = np.arctan2(np.hypot(pxb, py), pzb) * 1e3
    xL      = p_tot / P_BEAM_D

    rp_incl = RES["RP"]["incl"]
    rp_coin = RES["RP"]["coin"]
    
    kinematics = {"p": p_tot, "theta": th_beam, "xL": xL}
    for vname, arr in kinematics.items():
        for h_type, mask in (("gen", flat_is_sig), ("inc", rp_incl), ("coi", rp_coin)):
            v = arr[mask].astype(np.float64)
            if len(v):
                h_diff[vname][h_type].FillN(len(v), v, np.ones(len(v), dtype=np.float64))

    # 5. Fill 2D Acceptance Maps
    for dest, mask in ((h_map_n_incl, rp_incl), (h_map_n_coin, rp_coin)):
        tv, pv = th_beam[mask].astype(np.float64), p_tot[mask].astype(np.float64)
        if len(tv): dest.FillN(len(tv), tv, pv, np.ones(len(tv), dtype=np.float64))

    tv, pv = th_beam[flat_is_sig].astype(np.float64), p_tot[flat_is_sig].astype(np.float64)
    if len(tv):
        h_map_d_incl.FillN(len(tv), tv, pv, np.ones(len(tv), dtype=np.float64))
        h_map_d_coin.FillN(len(tv), tv, pv, np.ones(len(tv), dtype=np.float64))

    # 6. Accumulate Theta Table
    for k in range(len(edges) - 1):
        sl = flat_is_sig & (th_beam >= edges[k]) & (th_beam < edges[k + 1])
        theta_table[k]["ng"] += int(sl.sum())
        theta_table[k]["ni"] += int((sl & rp_incl).sum())
        theta_table[k]["nc"] += int((sl & rp_coin).sum())

    # 7. Fill Missing Deuteron Histograms
    missing_mask = flat_is_sig & ~rp_coin
    n_miss = int(missing_mask.sum())
    total_miss += n_miss
    
    if n_miss > 0:
        missing_end_z = ak.to_numpy(ak.flatten(tree["MCParticles.endpoint.z"].array()))[missing_mask].astype(np.float64)
        missing_end_x = ak.to_numpy(ak.flatten(tree["MCParticles.endpoint.x"].array()))[missing_mask].astype(np.float64)
        missing_th    = th_beam[missing_mask].astype(np.float64)
        missing_p     = p_tot[missing_mask].astype(np.float64)

        if len(missing_end_z):
            h_miss_z.FillN(len(missing_end_z), missing_end_z, np.ones(len(missing_end_z), dtype=np.float64))
            h_miss_kin.FillN(len(missing_th), missing_th, missing_p, np.ones(len(missing_th), dtype=np.float64))
            h_miss_th_z.FillN(len(missing_end_z), missing_end_z, missing_th, np.ones(len(missing_end_z), dtype=np.float64))
            h_miss_x_z.FillN(len(missing_end_z), missing_end_z, missing_end_x, np.ones(len(missing_end_z), dtype=np.float64))
            
    # File is automatically closed when tree is overwritten in the next loop, memory is freed.


# ==================================================================
# PRINT FINAL CONSOLIDATED STATISTICS
# ==================================================================
print("\n" + "="*50)
print("FINAL STATISTICS OVER ALL FILES")
print("="*50)
print(f"[status] generatorStatus of all PDG {DEUTERON} in MCParticles:")
for s, c in sorted(status_counts.items()):
    note = "  (Geant4-produced secondary)" if s == 0 else ""
    if SELECTION_MODE == "generated" and s == GEN_STATUS:
        note = "  <-- used as denominator"
    print(f"         status={int(s):<4d} n={int(c):<8d}{note}")
print(f"[status] selection = {SELECTION_MODE!r} -> N_gen = {total_gen_d}\n")

for key in ["RP"]:
    r = acc_stats[key]
    a_i = r["incl"] / total_gen_d if total_gen_d else 0.0
    a_c = r["coin"] / total_gen_d if total_gen_d else 0.0
    print(f"--- {key} ---")
    for i, (label, *_) in enumerate(STATIONS[key]):
        n_s = r["bits"][i]
        print(f"    {label:<32} n={n_s:<7d} A={n_s/total_gen_d if total_gen_d else 0:.5f}")
    print(f"    inclusive (>=1 station)          n={r['incl']:<7d} A={a_i:.5f} +/- {binom_err(r['incl'], total_gen_d):.5f}")
    print(f"    coincidence ({N_REQUIRED[key]}-of-{len(STATIONS[key])}){'':<15} n={r['coin']:<7d} A={a_c:.5f} +/- {binom_err(r['coin'], total_gen_d):.5f}")
    print(f"    coin/incl = {r['coin']/r['incl'] if r['incl'] else 0.0:.4f}   deuteron hits outside all z-windows = {r['outside']}\n")


print("[RP acceptance vs theta_beam]")
print(f"  {'theta [mrad]':<16}{'N_gen':>8}{'incl':>10}{'2-of-2':>10}{'A_incl':>10}{'A_coin':>10}{'coin/incl':>11}")
for k in range(len(edges) - 1):
    d = theta_table[k]
    if d["ng"] == 0: continue
    print(f"  {edges[k]:5.1f} {edges[k+1]:5.1f}{d['ng']:>8d}{d['ni']:>10d}{d['nc']:>10d}"
          f"{d['ni']/d['ng']:>10.4f}{d['nc']/d['ng']:>10.4f}{(d['nc']/d['ni'] if d['ni'] else 0.0):>11.4f}")


# ==================================================================
# FINAL DRAWING AND SAVING PLOTS
# ==================================================================
print("\n[plots] Generating master canvases...")

def draw_grid_master(key, hists, ncol, nrow, size, outfile, acc_dict):
    a_i = acc_dict["incl"] / total_gen_d if total_gen_d else 0.0
    a_c = acc_dict["coin"] / total_gen_d if total_gen_d else 0.0
    c = ROOT.TCanvas(f"canvas_{key.lower()}", f"{key} Deuteron Hits", *size)
    c.Divide(ncol, nrow)
    for i, h in enumerate(hists, start=1):
        c.cd(i)
        ROOT.gPad.SetRightMargin(0.15); ROOT.gPad.SetLeftMargin(0.13)
        ROOT.gPad.SetBottomMargin(0.2); ROOT.gPad.SetTopMargin(0.10)
        h.Draw("COLZ")
    c.cd(0)
    lat = ROOT.TLatex()
    lat.SetNDC(); lat.SetTextSize(0.050 if nrow == 1 else 0.025)
    lat.DrawLatex(0.20, 0.035, f"deuteron A_{{{N_REQUIRED[key]}/{len(hists)}}} = {a_c:.5f}   (incl. {a_i:.5f}, N_{{gen}} = {total_gen_d})")
    c.SaveAs(outfile)

draw_grid_master("RP",  h_xy["RP"],  2, 1, (1400, 600),  "./Plots/rompot_deuteron_hits_xy_both_stations.pdf", acc_stats["RP"])

# Diff Canvas
c_diff = ROOT.TCanvas("c_diff", "RP deuteron acceptance", 1800, 1000)
c_diff.Divide(3, 2)
keep = []
for i, (vname, *_rest) in enumerate(VARS_DEF):
    h_g = h_diff[vname]["gen"]; h_g.SetFillColorAlpha(ROOT.kGray + 1, 0.35); h_g.SetLineColor(ROOT.kGray + 2)
    h_i = h_diff[vname]["inc"]; h_i.SetLineColor(ROOT.kAzure + 2); h_i.SetLineWidth(2)
    h_c = h_diff[vname]["coi"]; h_c.SetLineColor(ROOT.kRed + 1); h_c.SetLineWidth(2)

    c_diff.cd(i + 1)
    ROOT.gPad.SetLogy(); ROOT.gPad.SetLeftMargin(0.13)
    h_g.SetMaximum(max(h_g.GetMaximum() * 5.0, 10.0)); h_g.SetMinimum(0.5)
    h_g.Draw("HIST"); h_i.Draw("HIST SAME"); h_c.Draw("HIST SAME")
    
    leg = ROOT.TLegend(0.55, 0.70, 0.88, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.035)
    leg.AddEntry(h_g, "generated", "f"); leg.AddEntry(h_i, "RP inclusive", "l"); leg.AddEntry(h_c, "RP 2-of-2", "l")
    leg.Draw()

    c_diff.cd(i + 4)
    ROOT.gPad.SetLeftMargin(0.13)
    frame = ROOT.TH1F(f"fr_{vname}", h_g.GetTitle().replace("deuterons", "acceptance"), h_g.GetNbinsX(), h_g.GetXaxis().GetXmin(), h_g.GetXaxis().GetXmax())
    frame.SetMinimum(0.0); frame.SetMaximum(1.15)
    frame.Draw()
    eff_i = ROOT.TEfficiency(h_i, h_g); eff_i.SetLineColor(ROOT.kAzure + 2); eff_i.SetMarkerColor(ROOT.kAzure + 2); eff_i.SetMarkerStyle(20); eff_i.SetMarkerSize(0.7)
    eff_c = ROOT.TEfficiency(h_c, h_g); eff_c.SetLineColor(ROOT.kRed + 1); eff_c.SetMarkerColor(ROOT.kRed + 1); eff_c.SetMarkerStyle(21); eff_c.SetMarkerSize(0.7)
    eff_i.Draw("SAME P"); eff_c.Draw("SAME P")
    keep.extend([h_g, h_i, h_c, frame, eff_i, eff_c, leg])

c_diff.SaveAs("./Plots/rompot_deuteron_acceptance_vs_p_theta_xL.pdf")

# Map Canvas
for h_n, h_d in ((h_map_n_incl, h_map_d_incl), (h_map_n_coin, h_map_d_coin)):
    h_n.Divide(h_d)
    for bx in range(1, h_n.GetNbinsX() + 1):
        for by in range(1, h_n.GetNbinsY() + 1):
            if h_d.GetBinContent(bx, by) == 0: h_n.SetBinContent(bx, by, -1.0)
    h_n.SetMinimum(-0.001); h_n.SetMaximum(1.0); h_n.GetYaxis().SetTitleOffset(1.3)

c_map = ROOT.TCanvas("c_map", "RP acceptance map", 1400, 600)
c_map.Divide(2, 1)
c_map.cd(1); ROOT.gPad.SetRightMargin(0.16); ROOT.gPad.SetLeftMargin(0.14); h_map_n_incl.Draw("COLZ")
c_map.cd(2); ROOT.gPad.SetRightMargin(0.16); ROOT.gPad.SetLeftMargin(0.14); h_map_n_coin.Draw("COLZ")
c_map.SaveAs("./Plots/rompot_deuteron_acceptance_map_p_vs_theta.pdf")

# Missing Canvases
if total_miss > 0:
    h_miss_z.SetLineColor(ROOT.kRed); h_miss_z.SetFillColorAlpha(ROOT.kRed, 0.35)
    c_miss_z = ROOT.TCanvas("c_miss_z", "Missing Z", 800, 600)
    ROOT.gPad.SetLogy()
    h_miss_z.Draw("HIST")
    ROOT.gPad.Update()
    y_min_miss, y_max_miss = ROOT.gPad.GetUymin(), ROOT.gPad.GetUymax()
    if ROOT.gPad.GetLogy(): y_min_miss, y_max_miss = 10**y_min_miss, 10**y_max_miss
    z_line_miss = ROOT.TLine(32546.2, y_min_miss, 32546.2, y_max_miss); z_line_miss.SetLineColor(ROOT.kBlack); z_line_miss.SetLineStyle(2); z_line_miss.SetLineWidth(2)
    z_line_miss.Draw("SAME")
    c_miss_z.SaveAs("./Plots/missing_deuterons_z.pdf")

    c_miss_kin = ROOT.TCanvas("c_miss_kin", "Missing Kinematics", 800, 600); ROOT.gPad.SetRightMargin(0.15); h_miss_kin.Draw("COLZ")
    c_miss_kin.SaveAs("./Plots/missing_deuterons_kinematics.pdf")

    c_miss_th_z = ROOT.TCanvas("c_miss_th_z", "Missing Theta vs Z", 800, 600); ROOT.gPad.SetRightMargin(0.15); h_miss_th_z.Draw("COLZ")
    c_miss_th_z.SaveAs("./Plots/missing_deuterons_theta_vs_z.pdf")

    c_miss_x_z = ROOT.TCanvas("c_miss_x_z", "Missing X vs Z", 800, 600); ROOT.gPad.SetRightMargin(0.15); ROOT.gPad.SetLeftMargin(0.13); h_miss_x_z.Draw("COLZ")
    c_miss_x_z.SaveAs("./Plots/missing_deuterons_x_vs_z.pdf")