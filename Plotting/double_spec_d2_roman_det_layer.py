import os
import ROOT
import numpy as np
import awkward as ak
import uproot
from pathlib import Path
MAX_EVENTS = -1
# Setup & Constants
# ==================================================================
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)

# NEW OUTPUT DIRECTORY
OUT_DIR = "/home/aabhishe/eic/EIC_Full_Sim/Plots/RP_Protons_D2_layer"
os.makedirs(OUT_DIR, exist_ok=True)

DEUTERON = 1000010020
PROTON   = 2212
GEN_STATUS = 1

# Files to process
dir_path = Path("/work/eic/users/aabhishe/Recon_out_1M")
file_count = len(list(dir_path.glob("recon_chunk_*.root")))
file_list = [f"/work/eic/users/aabhishe/Recon_out_1M/recon_chunk_{i}.root" for i in range(file_count)]
print(f"Found {len(file_list)} files to process in {dir_path}.")
print(f"Plots will be saved to: {OUT_DIR}")

# ==================================================================
# Layer Definitions & Histograms
# ==================================================================
sensor_window_z = 5 #mm
L1_z, L2_z, L3_z, L4_z = 32547.3, 32567.3, 34245.5, 34265.5
STATIONS = {
    "RP": [
        (f"P2 S1 Layer 1 (z={L1_z} mm)", L1_z - sensor_window_z, L1_z + sensor_window_z, (100, -1300., -950.), (100, -90., 90.)),
        (f"P2 S1 Layer 2 (z={L2_z} mm)", L2_z - sensor_window_z, L2_z + sensor_window_z, (100, -1300., -950.), (100, -90., 90.)),
        (f"P2 S2 Layer 1 (z={L3_z} mm)", L3_z - sensor_window_z, L3_z + sensor_window_z, (100, -1400., -1050.), (100, -90., 90.)),
        (f"P2 S2 Layer 2 (z={L4_z} mm)", L4_z - sensor_window_z, L4_z + sensor_window_z, (100, -1400., -1050.), (100, -90., 90.)),
    ]
}

h_xy_d = []
h_xy_p1 = []
h_xy_p2 = []

# NEW: Counters for acceptance
total_double_tag_events = 0
p1_acc_count = [0, 0, 0, 0]
p2_acc_count = [0, 0, 0, 0]

for i, (label, lo, hi, xb, yb) in enumerate(STATIONS["RP"]):
    # Deuteron Histograms (Red)
    hd = ROOT.TH2F(f"h_d_{i}", f"RP {label};X Position [mm];Y Position [mm]", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
    hd.SetMarkerColor(ROOT.kRed)
    hd.SetMarkerStyle(20)
    hd.SetMarkerSize(0.4)
    h_xy_d.append(hd)
    
    # Proton 1 Histograms (Blue)
    hp1 = ROOT.TH2F(f"h_p1_{i}", f"RP {label};X Position [mm];Y Position [mm]", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
    hp1.SetMarkerColor(ROOT.kBlue)
    hp1.SetMarkerStyle(20)
    hp1.SetMarkerSize(0.4)
    h_xy_p1.append(hp1)

    # Proton 2 Histograms (Green)
    hp2 = ROOT.TH2F(f"h_p2_{i}", f"RP {label};X Position [mm];Y Position [mm]", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
    hp2.SetMarkerColor(ROOT.kGreen + 2)
    hp2.SetMarkerStyle(20)
    hp2.SetMarkerSize(0.4)
    h_xy_p2.append(hp2)

# ==================================================================
# MAIN LOOP
# ==================================================================
print(f"Starting chunked processing for {len(file_list)} files...\n")

for f_idx, filename in enumerate(file_list):
    if f_idx == MAX_EVENTS:
        break
    if not os.path.exists(filename):
        continue
        
    print(f"Processing File {f_idx + 1}/{len(file_list)}...")
    tree = uproot.open(f"{filename}:events")

    # MC Truth Arrays
    mc_pdg    = tree["MCParticles.PDG"].array()
    mc_status = tree["MCParticles.generatorStatus"].array()

    # Hit Arrays
    idx   = tree["_ForwardRomanPotHits_particle.index"].array()
    hit_x = tree["ForwardRomanPotHits.position.x"].array()
    hit_y = tree["ForwardRomanPotHits.position.y"].array()
    hit_z = tree["ForwardRomanPotHits.position.z"].array()

    # Ensure index is in range to prevent awkward array crashes
    valid_idx = (idx >= 0) & (idx < ak.num(mc_pdg))

    hit_mc_pdg = ak.where(valid_idx, mc_pdg[idx], 0)
    hit_mc_sts = ak.where(valid_idx, mc_status[idx], 0)

    # 1. Mask for Deuteron Hits
    hit_is_d = valid_idx & (hit_mc_pdg == DEUTERON) & (hit_mc_sts == GEN_STATUS)

    # 2. Logic for Double Proton events
    is_p = (mc_pdg == PROTON) & (mc_status == GEN_STATUS)
    mc_indices = ak.local_index(mc_pdg)
    
    # Extract just the indices of the protons in each event
    p_indices = mc_indices[is_p]

    has_two_protons = ak.num(p_indices) >= 2
    
    # NEW: Accumulate total double tagging events
    total_double_tag_events += int(ak.sum(has_two_protons))
    
    # Pad the list so every event has at least 2 slots (fills missing with None)
    p_indices_2 = ak.pad_none(p_indices, 2, axis=1)
    
    # Extract the target MC indices for Proton 1 and Proton 2
    idx_p1 = p_indices_2[:, 0]  
    idx_p2 = p_indices_2[:, 1]

    # A hit belongs to p1/p2 if its associated index matches the p1/p2 index for that event
    hit_is_p1 = valid_idx & ak.fill_none(idx == idx_p1, False) & has_two_protons
    hit_is_p2 = valid_idx & ak.fill_none(idx == idx_p2, False) & has_two_protons

    # 3. Fill the Histograms
    for i, (label, lo, hi, xb, yb) in enumerate(STATIONS["RP"]):
        zm = (hit_z > lo) & (hit_z < hi)

        # Fill Deuteron
        m_d = hit_is_d & zm
        x_d = ak.to_numpy(ak.flatten(hit_x[m_d])).astype(np.float64)
        y_d = ak.to_numpy(ak.flatten(hit_y[m_d])).astype(np.float64)
        if len(x_d): h_xy_d[i].FillN(len(x_d), x_d, y_d, np.ones(len(x_d), dtype=np.float64))

        # Fill Proton 1
        m_p1 = hit_is_p1 & zm
        
        # NEW: Accumulate accepted P1 particles (unique events where P1 hit this layer)
        p1_acc_count[i] += int(ak.sum(ak.any(m_p1, axis=1)))
        
        x_p1 = ak.to_numpy(ak.flatten(hit_x[m_p1])).astype(np.float64)
        y_p1 = ak.to_numpy(ak.flatten(hit_y[m_p1])).astype(np.float64)
        # x_p1 += 100.
        if len(x_p1): h_xy_p1[i].FillN(len(x_p1), x_p1, y_p1, np.ones(len(x_p1), dtype=np.float64))

        # Fill Proton 2
        m_p2 = hit_is_p2 & zm
        p2_acc_count[i] += int(ak.sum(ak.any(m_p2, axis=1)))
        x_p2 = ak.to_numpy(ak.flatten(hit_x[m_p2])).astype(np.float64)
        y_p2 = ak.to_numpy(ak.flatten(hit_y[m_p2])).astype(np.float64)
        if len(x_p2): h_xy_p2[i].FillN(len(x_p2), x_p2, y_p2, np.ones(len(x_p2), dtype=np.float64))


# ==================================================================
# DRAWING AND SAVING PLOT
# ==================================================================
print("\n[plots] Generating canvas...")

c_master = ROOT.TCanvas("c_master", "Roman Pot Hits", 1400, 1000)
c_master.Divide(2, 2)
legends = []    
for i in range(4):
    c_master.cd(i + 1)
    ROOT.gPad.SetLeftMargin(0.13)
    ROOT.gPad.SetRightMargin(0.08)
    ROOT.gPad.SetBottomMargin(0.12)
    
    # Use h_xy_d as the frame for the axes
    # h_xy_d[i].GetYaxis().SetTitleOffset(1.3)
    # h_xy_d[i].SetLineColor(ROOT.kRed)
    # h_xy_d[i].SetFillColor(ROOT.kRed)
    # h_xy_d[i].SetLineWidth(2)

    # h_xy_p1[i].SetLineColor(ROOT.kBlue)
    # h_xy_p1[i].SetFillColor(ROOT.kBlue)
    # h_xy_p1[i].SetLineWidth(2)

    # h_xy_p2[i].SetLineColor(ROOT.kGreen)
    # h_xy_p2[i].SetFillColor(ROOT.kGreen)
    # h_xy_p2[i].SetLineWidth(2)

    # Draw Deuteron, Proton 1, and Proton 2 on top of each other using P (points)
    #h_xy_d[i].Draw("BOX")
    h_xy_p1[i].Draw("COLZ")
    # h_xy_p2[i].Draw("BOX SAME")
    
    # NEW: Calculate acceptance for Proton 1
    acc_p1 = p1_acc_count[i] / total_double_tag_events if total_double_tag_events > 0 else 0.0
    acc_p2 = p2_acc_count[i] / total_double_tag_events if total_double_tag_events > 0 else 0.0
    
    # Legend
    leg = ROOT.TLegend(0.50, 0.75, 0.90, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    #leg.AddEntry(h_xy_d[i], f"Deuteron (N={int(h_xy_d[i].GetEntries())})", "p")
    #leg.AddEntry(h_xy_p1[i], f"Proton 1 (N={int(h_xy_p1[i].GetEntries())}, A={acc_p1:.4f})", "p")
    leg.AddEntry(h_xy_p2[i], f"Proton 2 (N={int(h_xy_p2[i].GetEntries())}, A={acc_p2:.4f})", "p")
    leg.Draw()
    legends.append(leg)
    

c_master.SaveAs(f"{OUT_DIR}/rompot_2D_hits_protons2.pdf")
print(f"Total Double Tagging Events: {total_double_tag_events}")
print("Done!")