import os
import ROOT
import numpy as np
import awkward as ak
import uproot
from pathlib import Path

# ==================================================================
# Setup & Constants
# ==================================================================
max_events = -1
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kViridis)
ROOT.TH1.AddDirectory(False)
os.makedirs("./Plots", exist_ok=True)

DEUTERON = 1000010020
SELECTION_MODE = "generated"     
GEN_STATUS     = 1

CROSSING_ANGLE = 0.025           
ROT_SIGN       = 1.0             
P_BEAM_D       = 82.0            

if SELECTION_MODE == "generated":
    SIG_LABEL = f"generated d (status=={GEN_STATUS})"
elif SELECTION_MODE == "all_deuterons":
    SIG_LABEL = "all d (any status)"
else:
    raise ValueError(f"unknown SELECTION_MODE {SELECTION_MODE!r}")

dir_path = Path("/work/eic/users/aabhishe/Recon_out_1M")
file_count = len(list(dir_path.glob("recon_chunk_*.root")))
file_list = [f"/work/eic/users/aabhishe/Recon_out_1M/recon_chunk_{i}.root" for i in range(file_count)]
print(f"Found {len(file_list)} files to process in {dir_path}.")

# ==================================================================
# Master Dictionaries & Histograms 
# ==================================================================
STATIONS = {
    "RP": [
        ("Station 1 (z #approx 32.5 m)", None,  32600., (100, -1300., -950.), (100, -90., 90.)),
        ("Station 2 (z #approx 34.3 m)", 34200., None,  (100, -1400., -1050.), (100, -90., 90.)),
    ]
}

N_REQUIRED = {"RP": 2}
COLLS = {"RP": ("ForwardRomanPotHits", "_ForwardRomanPotHits_particle")}

total_gen_d = 0
status_counts = {}
acc_stats = {"RP": {"incl": 0, "coin": 0, "outside": 0, "bits": [0, 0]}}

edges = np.linspace(0.0, 7.0, 8) 
theta_table = {k: {"ng": 0, "n1": 0, "n2": 0, "ni": 0, "nc": 0} for k in range(len(edges) - 1)}

h_xy = {"RP": []}
for key in STATIONS:
    for i, (label, lo, hi, xb, yb) in enumerate(STATIONS[key]):
        h = ROOT.TH2F(f"h_master_{key.lower()}{i+1}", f"Deuterons: {key} {label}", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
        h.GetXaxis().SetTitle("X Position [mm]")
        h.GetYaxis().SetTitle("Y Position [mm]")
        h.GetXaxis().SetTitleOffset(1.2)
        h.GetYaxis().SetTitleOffset(1.3)
        h.GetXaxis().SetNdivisions(505)
        h.GetXaxis().SetTitleSize(0.05)
        h.GetYaxis().SetTitleSize(0.05)
        h.GetXaxis().SetLabelSize(0.045)
        h.GetYaxis().SetLabelSize(0.045)
        h.GetXaxis().CenterTitle(True)
        h.GetYaxis().CenterTitle(True)
        h_xy[key].append(h)

VARS_DEF = [
    ("p",     "p [GeV]",               60, 55.0, 105.0),
    ("theta", "#theta_{beam} [mrad]",  60, 0.0,  6.0),
]
h_diff = {}
for vname, axt, nb, lo, hi in VARS_DEF:
    h_diff[vname] = {
        # Capitalized Deuterons here
        "gen": ROOT.TH1F(f"hg_{vname}", f";{axt};Deuterons", nb, lo, hi),
        "inc": ROOT.TH1F(f"hi_{vname}", f";{axt};Deuterons", nb, lo, hi),
        "coi": ROOT.TH1F(f"hc_{vname}", f";{axt};Deuterons", nb, lo, hi),
    }

h_map_n_incl = ROOT.TH2F("n_incl", "RP deuteron acceptance (inclusive)", 40, 0.0, 7.0, 40, 40.0, 130.0)
h_map_d_incl = ROOT.TH2F("d_incl", "RP deuteron acceptance (inclusive)", 40, 0.0, 7.0, 40, 40.0, 130.0)
h_map_n_coin = ROOT.TH2F("n_coin", "RP deuteron acceptance (2-of-2)", 40, 0.0, 7.0, 40, 40.0, 130.0)
h_map_d_coin = ROOT.TH2F("d_coin", "RP deuteron acceptance (2-of-2)", 40, 0.0, 7.0, 40, 40.0, 130.0)

# ==================================================================
# Helper Functions
# ==================================================================
def validate_assoc(tree, hit_coll, assoc_coll, n_mc):
    idx    = tree[f"{assoc_coll}.index"].array()
    n_hits = ak.num(tree[f"{hit_coll}.position.x"].array())
    n_assc = ak.num(idx)
    if int(ak.sum(n_hits != n_assc)):
        raise RuntimeError(f"{assoc_coll}: mismatched association length.")
    in_range = (idx >= 0) & (idx < n_mc)
    return idx, in_range, ak.where(in_range, idx, 0)

def hit_masks(tree, hit_coll, assoc_coll, n_mc, mc_pdg, mc_is_sig_jag):
    _, valid, safe = validate_assoc(tree, hit_coll, assoc_coll, n_mc)
    pdg_mask = (ak.where(valid, mc_pdg[safe], 0) == DEUTERON) & valid
    return mc_is_sig_jag[safe] & valid, pdg_mask, safe

def get_xyz(tree, hit_coll, mask):
    return tree[f"{hit_coll}.position.x"].array()[mask], tree[f"{hit_coll}.position.y"].array()[mask], tree[f"{hit_coll}.position.z"].array()[mask]

def zmask(z, lo, hi):
    if lo is None: return z < hi
    if hi is None: return z > lo
    return (z > lo) & (z < hi)

def particles_with_hit(safe, sel, offsets, n_mc_total):
    g = ak.to_numpy(ak.flatten(safe[sel] + offsets))
    out = np.zeros(n_mc_total, dtype=bool)
    if len(g): out[np.unique(g)] = True
    return out

# ==================================================================
# MAIN LOOP
# ==================================================================
print(f"Starting chunked processing for {len(file_list)} files...\n")

for f_idx, filename in enumerate(file_list):
    if f_idx == max_events: break
    if not os.path.exists(filename): continue
    print(f"Processing File {f_idx + 1}/{len(file_list)}...")
    tree = uproot.open(f"{filename}:events")

    mc_pdg    = tree["MCParticles.PDG"].array()
    mc_status = tree["MCParticles.generatorStatus"].array()
    mc_is_sig = (mc_pdg == DEUTERON) & (mc_status == GEN_STATUS) if SELECTION_MODE == "generated" else (mc_pdg == DEUTERON)

    n_mc       = ak.to_numpy(ak.num(mc_pdg))
    offsets    = np.concatenate([[0], np.cumsum(n_mc)])[:-1]
    n_mc_total = int(n_mc.sum())
    
    flat_is_sig = ak.to_numpy(ak.flatten(mc_is_sig))
    total_gen_d += int(flat_is_sig.sum())
    mc_is_sig_jag = ak.unflatten(flat_is_sig, n_mc)

    _d_status = ak.to_numpy(ak.flatten(mc_status[mc_pdg == DEUTERON]))
    if len(_d_status):
        for s, c in zip(*np.unique(_d_status, return_counts=True)):
            status_counts[s] = status_counts.get(s, 0) + c

    RES = {}
    for key in ["RP"]:
        sig_mask, _, safe = hit_masks(tree, *COLLS[key], n_mc, mc_pdg, mc_is_sig_jag)
        z = tree[f"{COLLS[key][0]}.position.z"].array()

        bits, in_any = [], None
        for (label, lo, hi, _xb, _yb) in STATIONS[key]:
            zm = zmask(z, lo, hi)
            bits.append(particles_with_hit(safe, sig_mask & zm, offsets, n_mc_total))
            in_any = zm if in_any is None else (in_any | zm)

        n_outside = int(ak.sum(sig_mask & ~in_any))
        n_fired = np.sum(np.stack(bits), axis=0)
        
        RES[key] = {
            "bits": bits, "incl": (n_fired >= 1) & flat_is_sig, 
            "coin": (n_fired >= N_REQUIRED[key]) & flat_is_sig, "sig_mask": sig_mask
        }

        acc_stats[key]["incl"] += int(RES[key]["incl"].sum())
        acc_stats[key]["coin"] += int(RES[key]["coin"].sum())
        acc_stats[key]["outside"] += n_outside
        for i, b in enumerate(bits): acc_stats[key]["bits"][i] += int((b & flat_is_sig).sum())

    for key in ["RP"]:
        x, y, z = get_xyz(tree, COLLS[key][0], RES[key]["sig_mask"])
        for i, (_, lo, hi, _, _) in enumerate(STATIONS[key]):
            m = zmask(z, lo, hi)
            xv, yv = ak.to_numpy(ak.flatten(x[m])).astype(np.float64), ak.to_numpy(ak.flatten(y[m])).astype(np.float64)
            if len(xv): h_xy[key][i].FillN(len(xv), xv, yv, np.ones(len(xv), dtype=np.float64))

    px, py, pz = [ak.to_numpy(ak.flatten(tree[f"MCParticles.momentum.{v}"].array())) for v in ["x", "y", "z"]]
    p_tot   = np.sqrt(px**2 + py**2 + pz**2)
    ca, sa  = np.cos(ROT_SIGN * CROSSING_ANGLE), np.sin(ROT_SIGN * CROSSING_ANGLE)
    th_beam = np.arctan2(np.hypot(ca * px + sa * pz, py), -sa * px + ca * pz) * 1e3
    
    rp_incl, rp_coin = RES["RP"]["incl"], RES["RP"]["coin"]
    
    for vname, arr in {"p": p_tot, "theta": th_beam}.items():
        for h_type, mask in (("gen", flat_is_sig), ("inc", rp_incl), ("coi", rp_coin)):
            v = arr[mask].astype(np.float64)
            if len(v): h_diff[vname][h_type].FillN(len(v), v, np.ones(len(v), dtype=np.float64))

    for dest, mask in ((h_map_n_incl, rp_incl), (h_map_n_coin, rp_coin)):
        tv, pv = th_beam[mask].astype(np.float64), p_tot[mask].astype(np.float64)
        if len(tv): dest.FillN(len(tv), tv, pv, np.ones(len(tv), dtype=np.float64))

    tv, pv = th_beam[flat_is_sig].astype(np.float64), p_tot[flat_is_sig].astype(np.float64)
    if len(tv):
        h_map_d_incl.FillN(len(tv), tv, pv, np.ones(len(tv), dtype=np.float64))
        h_map_d_coin.FillN(len(tv), tv, pv, np.ones(len(tv), dtype=np.float64))

    s1_bits, s2_bits = RES["RP"]["bits"][0] & flat_is_sig, RES["RP"]["bits"][1] & flat_is_sig
    for k in range(len(edges) - 1):
        sl = flat_is_sig & (th_beam >= edges[k]) & (th_beam < edges[k + 1])
        theta_table[k]["ng"] += int(sl.sum())
        theta_table[k]["n1"] += int((sl & s1_bits).sum())
        theta_table[k]["n2"] += int((sl & s2_bits).sum())
        theta_table[k]["ni"] += int((sl & rp_incl).sum())
        theta_table[k]["nc"] += int((sl & rp_coin).sum())

# ==================================================================
# PRINT & PLOT
# ==================================================================
print("\n" + "="*50 + "\nFINAL STATISTICS OVER ALL FILES\n" + "="*50)
for s, c in sorted(status_counts.items()):
    print(f"         status={int(s):<4d} n={int(c):<8d}")
print(f"\n[status] selection = {SELECTION_MODE!r} -> N_gen = {total_gen_d}\n")

r = acc_stats["RP"]
print("--- RP ---")
for i, (label, *_) in enumerate(STATIONS["RP"]):
    print(f"    {label:<32} n={r['bits'][i]:<7d} A={r['bits'][i]/total_gen_d if total_gen_d else 0:.5f}")
print(f"    inclusive (>=1 station)          n={r['incl']:<7d} A={r['incl']/total_gen_d if total_gen_d else 0:.5f}")
print(f"    coincidence (2-of-2)             n={r['coin']:<7d} A={r['coin']/total_gen_d if total_gen_d else 0:.5f}\n")

print("[RP acceptance vs theta_beam]")
print(f"  {'theta [mrad]':<16}{'N_gen':>8}{'Stat 1':>10}{'Stat 2':>10}{'incl':>10}{'2-of-2':>10}{'A_1':>10}{'A_2':>10}{'A_incl':>10}{'A_coin':>10}{'coin/incl':>11}")
for k in range(len(edges) - 1):
    d = theta_table[k]
    if d["ng"]:
        print(f"  {edges[k]:5.1f} {edges[k+1]:5.1f}{d['ng']:>8d}{d['n1']:>10d}{d['n2']:>10d}{d['ni']:>10d}{d['nc']:>10d}"
              f"{d['n1']/d['ng']:>10.4f}{d['n2']/d['ng']:>10.4f}{d['ni']/d['ng']:>10.4f}{d['nc']/d['ng']:>10.4f}{(d['nc']/d['ni'] if d['ni'] else 0.0):>11.4f}")

# Plots
c1 = ROOT.TCanvas("c1", "RP Hits", 1400, 600); c1.Divide(2, 1)
for i, h in enumerate(h_xy["RP"], 1):
    c1.cd(i); ROOT.gPad.SetRightMargin(0.15); ROOT.gPad.SetLeftMargin(0.13); ROOT.gPad.SetBottomMargin(0.2); h.Draw("COLZ")
c1.SaveAs("./Plots/rompot_deuteron_hits_xy_both_stations.pdf")

# Set up a 2x2 grid for p and theta (Top row: Hists, Bottom row: Efficiency)
c_diff = ROOT.TCanvas("c_diff", "RP deuteron acceptance", 1400, 1000)
c_diff.Divide(2, 2)
keep = []

for i, (vname, axt, *_) in enumerate(VARS_DEF):
    h_g, h_i, h_c = h_diff[vname]["gen"], h_diff[vname]["inc"], h_diff[vname]["coi"]
    h_g.SetFillColorAlpha(ROOT.kGray + 1, 0.35); h_g.SetLineColor(ROOT.kGray + 2)
    h_i.SetLineColor(ROOT.kAzure + 2); h_i.SetLineWidth(2)
    h_c.SetLineColor(ROOT.kRed + 1); h_c.SetLineWidth(2)
    
    # Global cosmetics for these histograms
    for h in [h_g, h_i, h_c]:
        h.GetXaxis().SetTitleOffset(1.1)
        h.GetYaxis().SetTitleOffset(1.3)
        h.GetXaxis().SetTitleSize(0.055)
        h.GetYaxis().SetTitleSize(0.055)
        h.GetXaxis().SetLabelSize(0.05)
        h.GetYaxis().SetLabelSize(0.05)
        h.GetXaxis().CenterTitle(True)
        h.GetYaxis().CenterTitle(True)

    # Top Row: Histograms
    c_diff.cd(i + 1)
    ROOT.gPad.SetLogy(); ROOT.gPad.SetLeftMargin(0.16); ROOT.gPad.SetRightMargin(0.05); ROOT.gPad.SetBottomMargin(0.15)
    h_g.SetMaximum(max(h_g.GetMaximum() * 5.0, 10.0)); h_g.SetMinimum(0.5)
    h_g.Draw("HIST"); h_i.Draw("HIST SAME"); h_c.Draw("HIST SAME")
    
    if i == 0:
        leg = ROOT.TLegend(0.45, 0.70, 0.90, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.045)
        leg.AddEntry(h_g, "generated", "f"); leg.AddEntry(h_i, "RP inclusive", "l"); leg.AddEntry(h_c, "RP 2-of-2", "l")
        leg.Draw(); keep.append(leg)

    # Bottom Row: Efficiency / Acceptance Fractions
    c_diff.cd(i + 3)
    ROOT.gPad.SetLeftMargin(0.16); ROOT.gPad.SetRightMargin(0.05); ROOT.gPad.SetBottomMargin(0.15); ROOT.gPad.SetGridy()
    
    # Ensure X axis title is matched perfectly, Y axis is Detected / Generated
    frame = ROOT.TH1F(f"fr_{vname}", f";{axt};Detected / Generated", h_g.GetNbinsX(), h_g.GetXaxis().GetXmin(), h_g.GetXaxis().GetXmax())
    frame.SetMinimum(0.0); frame.SetMaximum(1.15)
    frame.GetXaxis().SetTitleSize(0.055); frame.GetYaxis().SetTitleSize(0.055)
    frame.GetXaxis().SetLabelSize(0.05); frame.GetYaxis().SetLabelSize(0.05)
    frame.GetXaxis().SetTitleOffset(1.1); frame.GetYaxis().SetTitleOffset(1.3)
    frame.GetXaxis().CenterTitle(True); frame.GetYaxis().CenterTitle(True)
    frame.Draw()
    
    eff_i = ROOT.TEfficiency(h_i, h_g); eff_i.SetLineColor(ROOT.kAzure + 2); eff_i.SetMarkerColor(ROOT.kAzure + 2); eff_i.SetMarkerStyle(20)
    eff_c = ROOT.TEfficiency(h_c, h_g); eff_c.SetLineColor(ROOT.kRed + 1); eff_c.SetMarkerColor(ROOT.kRed + 1); eff_c.SetMarkerStyle(21)
    eff_i.Draw("SAME P"); eff_c.Draw("SAME P"); keep.extend([h_g, frame, eff_i, eff_c])

c_diff.SaveAs("./Plots/rompot_deuteron_acceptance_vs_p_theta.pdf")

for h_n, h_d in ((h_map_n_incl, h_map_d_incl), (h_map_n_coin, h_map_d_coin)):
    h_n.Divide(h_d)
    for bx in range(1, h_n.GetNbinsX() + 1):
        for by in range(1, h_n.GetNbinsY() + 1):
            if h_d.GetBinContent(bx, by) == 0: h_n.SetBinContent(bx, by, -1.0)
    h_n.SetMinimum(-0.001); h_n.SetMaximum(1.0); h_n.GetYaxis().SetTitleOffset(1.3)

c_map = ROOT.TCanvas("c_map", "RP map", 1400, 600); c_map.Divide(2, 1)
c_map.cd(1); ROOT.gPad.SetRightMargin(0.16); ROOT.gPad.SetLeftMargin(0.14); h_map_n_incl.Draw("COLZ")
c_map.cd(2); ROOT.gPad.SetRightMargin(0.16); ROOT.gPad.SetLeftMargin(0.14); h_map_n_coin.Draw("COLZ")
c_map.SaveAs("./Plots/rompot_deuteron_acceptance_map_p_vs_theta.pdf")