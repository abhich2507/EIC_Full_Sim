import os
import ROOT
import numpy as np
import awkward as ak
import uproot
from pathlib import Path

# ==================================================================
# Setup & Constants
# ==================================================================
max_events = 50
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

dir_path = Path("/work/eic/users/aabhishe/Recon_out_1M")
file_count = len(list(dir_path.glob("recon_chunk_*.root")))
file_list = [f"/work/eic/users/aabhishe/Recon_out_1M/recon_chunk_{i}.root" for i in range(file_count)]
print(f"Found {len(file_list)} files to process in {dir_path}.")

STATIONS = {
    "RP": [
        ("Station 1 (z #approx 32.5 m)", None,  32600., (100, -1300., -950.), (100, -90., 90.)),
        ("Station 2 (z #approx 34.3 m)", 34200., None,  (100, -1400., -1050.), (100, -90., 90.)),
    ]
}

N_REQUIRED = {"RP": 2}
COLLS = {"RP": ("ForwardRomanPotHits", "_ForwardRomanPotHits_particle")}

total_miss = 0

# Initialize ROOT Master Histograms (Missing Deuterons Only)
# Updated ranges: p = 40 to 130, theta = 0 to 7
h_miss_z    = ROOT.TH1F("h_missing_z", "Endpoint Z of Missing Deuterons;z [mm];Count", 200, 0.0, 40000.0)
h_miss_kin  = ROOT.TH2F("h_missing_kin", "Kinematics of Missing Deuterons;#theta_{beam} [mrad];p [GeV]", 50, 0.0, 7.0, 50, 40.0, 130.0)
h_miss_th_z = ROOT.TH2F("h_miss_th_z", "Missing Deuterons: #theta_{beam} vs Endpoint Z;Endpoint Z [mm];#theta_{beam} [mrad]", 50, 25000.0, 40000.0, 50, 0.0, 7.0)
h_miss_x_z  = ROOT.TH2F("h_miss_x_z", "Missing Deuterons: Endpoint X vs Endpoint Z;Endpoint Z [mm];Endpoint X [mm]", 50, 25000.0, 40000.0, 50, -1500.0, -500.0)

# ==================================================================
# Helper Functions
# ==================================================================
def validate_assoc(tree, hit_coll, assoc_coll, n_mc):
    idx = tree[f"{assoc_coll}.index"].array()
    in_range = (idx >= 0) & (idx < n_mc)
    return idx, in_range, ak.where(in_range, idx, 0)

def hit_masks(tree, hit_coll, assoc_coll, n_mc, mc_pdg, mc_is_sig_jag):
    _, valid, safe = validate_assoc(tree, hit_coll, assoc_coll, n_mc)
    return mc_is_sig_jag[safe] & valid, safe

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
    if f_idx == max_events: 
        break
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
    mc_is_sig_jag = ak.unflatten(flat_is_sig, n_mc)

    # Reconstruct RP Coincidence logic to identify the missing ones
    sig_mask, safe = hit_masks(tree, *COLLS["RP"], n_mc, mc_pdg, mc_is_sig_jag)
    z = tree[f"{COLLS['RP'][0]}.position.z"].array()

    bits = []
    for (_, lo, hi, _, _) in STATIONS["RP"]:
        bits.append(particles_with_hit(safe, sig_mask & zmask(z, lo, hi), offsets, n_mc_total))

    rp_coin = (np.sum(np.stack(bits), axis=0) >= N_REQUIRED["RP"]) & flat_is_sig
    missing_mask = flat_is_sig & ~rp_coin
    
    n_miss = int(missing_mask.sum())
    total_miss += n_miss
    
    if n_miss > 0:
        px, py, pz = [ak.to_numpy(ak.flatten(tree[f"MCParticles.momentum.{v}"].array())) for v in ["x", "y", "z"]]
        ca, sa = np.cos(ROT_SIGN * CROSSING_ANGLE), np.sin(ROT_SIGN * CROSSING_ANGLE)
        
        missing_end_z = ak.to_numpy(ak.flatten(tree["MCParticles.endpoint.z"].array()))[missing_mask].astype(np.float64)
        missing_end_x = ak.to_numpy(ak.flatten(tree["MCParticles.endpoint.x"].array()))[missing_mask].astype(np.float64)
        missing_th    = (np.arctan2(np.hypot(ca * px + sa * pz, py), -sa * px + ca * pz) * 1e3)[missing_mask].astype(np.float64)
        missing_p     = np.sqrt(px**2 + py**2 + pz**2)[missing_mask].astype(np.float64)

        if len(missing_end_z):
            h_miss_z.FillN(len(missing_end_z), missing_end_z, np.ones(len(missing_end_z), dtype=np.float64))
            h_miss_kin.FillN(len(missing_th), missing_th, missing_p, np.ones(len(missing_th), dtype=np.float64))
            h_miss_th_z.FillN(len(missing_end_z), missing_end_z, missing_th, np.ones(len(missing_end_z), dtype=np.float64))
            h_miss_x_z.FillN(len(missing_end_z), missing_end_z, missing_end_x, np.ones(len(missing_end_z), dtype=np.float64))

print(f"\n[Done] Total missing deuterons analyzed: {total_miss}")

# ==================================================================
# DRAWING AND SAVING PLOTS
# ==================================================================
if total_miss > 0:
    h_miss_z.SetLineColor(ROOT.kRed); h_miss_z.SetFillColorAlpha(ROOT.kRed, 0.35)
    c_miss_z = ROOT.TCanvas("c_miss_z", "Missing Z", 800, 600)
    ROOT.gPad.SetLogy(); h_miss_z.Draw("HIST"); ROOT.gPad.Update()
    
    y_min, y_max = ROOT.gPad.GetUymin(), ROOT.gPad.GetUymax()
    if ROOT.gPad.GetLogy(): y_min, y_max = 10**y_min, 10**y_max
    
    z_line = ROOT.TLine(32546.2, y_min, 32546.2, y_max)
    z_line.SetLineColor(ROOT.kBlack); z_line.SetLineStyle(2); z_line.SetLineWidth(2); z_line.Draw("SAME")
    c_miss_z.SaveAs("./Plots/missing_deuterons_z.pdf")

    c_miss_kin = ROOT.TCanvas("c_miss_kin", "Missing Kinematics", 800, 600)
    ROOT.gPad.SetRightMargin(0.15); h_miss_kin.Draw("COLZ")
    c_miss_kin.SaveAs("./Plots/missing_deuterons_kinematics.pdf")

    c_miss_th_z = ROOT.TCanvas("c_miss_th_z", "Missing Theta vs Z", 800, 600)
    ROOT.gPad.SetRightMargin(0.15); h_miss_th_z.Draw("COLZ")
    c_miss_th_z.SaveAs("./Plots/missing_deuterons_theta_vs_z.pdf")

    c_miss_x_z = ROOT.TCanvas("c_miss_x_z", "Missing X vs Z", 800, 600)
    ROOT.gPad.SetRightMargin(0.15); ROOT.gPad.SetLeftMargin(0.13); h_miss_x_z.Draw("COLZ")
    c_miss_x_z.SaveAs("./Plots/missing_deuterons_x_vs_z.pdf")