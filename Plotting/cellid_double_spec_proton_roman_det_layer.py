import os
import ROOT
import numpy as np
import awkward as ak
import uproot
from pathlib import Path

# ==================================================================
# Setup & Constants
# ==================================================================
MAX_EVENTS = -1
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.TH1.AddDirectory(False)

# NEW OUTPUT DIRECTORY
# OUT_DIR = "/home/aabhishe/eic/EIC_Full_Sim/Plots/RP_Protons_D2_layer"
OUT_DIR = "/home/aabhishe/eic/EIC_Full_Sim/Plots/Test_RP_Protons_D2_layer"
os.makedirs(OUT_DIR, exist_ok=True)

DEUTERON   = 1000010020
PROTON     = 2212
GEN_STATUS = 1
PZ_MIN     = 20.0  # GeV/c cut for true beam-momentum spectator protons
VZ_MAX = 500.

#dir_path = Path("/work/eic/users/aabhishe/Recon_out_1M")
dir_path = Path("/work/eic/users/aabhishe/Test_run_pencil")
file_count = len(list(dir_path.glob("*.root")))
#file_list = [str(dir_path / f"recon_chunk_{i}.root") for i in range(file_count)]
file_list = [str(f) for f in dir_path.glob("*.root")]
print(f"Found {len(file_list)} files to process in {dir_path}.")
print(f"Plots will be saved to: {OUT_DIR}")
#/work/eic/users/aabhishe/Test_run_pencil/test_proton_gun.edm4hep.root
# ==================================================================
# Roman Pot cellID Bitfield Decoder (Matched to XML)
# <id>system:8,assembly:3,layer:4,module:8,sensor:8,x:32:-16,y:-16</id>
# ==================================================================
def decode_rp_cellid(cellID):
    """Vectorized extraction of RP bitfields based on ePIC XML readout definition."""
    sys_id   = (cellID >> 0) & 0xFF
    assembly = (cellID >> 8) & 0x7
    layer    = (cellID >> 11) & 0xF
    module   = (cellID >> 15) & 0xFF
    sensor   = (cellID >> 23) & 0xFF
    return sys_id, assembly, layer, module, sensor

def discover_rp_geometry(files):
    """
    Scans files to auto-discover the (system, layer) IDs for the 4 RP stations.
    The XML states the sensitive layer is the 4th component (index 4 in 1-based indexing, 
    often recorded as sensor=4 or mapped directly by GEANT4 depending on tracking flags).
    """
    for fpath in files:
        if not os.path.exists(fpath): continue
        try:
            tree = uproot.open(f"{fpath}:events")
            if "ForwardRomanPotHits.cellID" not in tree: continue
            
            cids = ak.flatten(tree["ForwardRomanPotHits.cellID"].array())
            zs   = ak.flatten(tree["ForwardRomanPotHits.position.z"].array())
            if len(cids) == 0: continue
            
            sys_ids, _, layers, _, sensors = decode_rp_cellid(cids)
            
            sys_ids = ak.to_numpy(sys_ids)
            layers  = ak.to_numpy(layers)
            sensors = ak.to_numpy(sensors)
            z_vals  = ak.to_numpy(zs)
            
            # Identify the sensor ID that records the vast majority of tracking hits 
            # (This corresponds to the LGAD module_component sensitive="true")
            active_sensor = int(np.bincount(sensors).argmax())
            
            unique_pairs = np.unique(np.column_stack((sys_ids, layers)), axis=0)
            
            pairs_with_z = []
            for (s, l) in unique_pairs:
                mask = (sys_ids == s) & (layers == l) & (sensors == active_sensor)
                if np.any(mask):
                    mean_z = np.mean(z_vals[mask])
                    pairs_with_z.append(((s, l), mean_z))
                    
            # Sort by Z-position (Station 1 Layer 1 -> Station 2 Layer 2)
            pairs_with_z.sort(key=lambda x: x[1])
            
            if len(pairs_with_z) >= 4:
                return [p[0] for p in pairs_with_z[:4]], active_sensor
                
        except Exception:
            pass
            
    raise RuntimeError("Could not auto-discover RP geometry from the input files.")

RP_STATION_IDS, ACTIVE_SENSOR = discover_rp_geometry(file_list)
print(f"Auto-Discovered RP Geometry -> Active Sensor Field: {ACTIVE_SENSOR}")
for i, (s, l) in enumerate(RP_STATION_IDS):
    print(f"  Station Layer {i+1}: system={s}, layer={l}")


STATIONS = [
    (f"Proton S1 Layer 1", RP_STATION_IDS[0], (100, -1300., -950.), (100, -90., 90.)),
    (f"Proton S1 Layer 2", RP_STATION_IDS[1], (100, -1300., -950.), (100, -90., 90.)),
    (f"Proton S2 Layer 1", RP_STATION_IDS[2], (100, -1400., -1050.), (100, -90., 90.)),
    (f"Proton S2 Layer 2", RP_STATION_IDS[3], (100, -1400., -1050.), (100, -90., 90.)),
]

h_xy_both = []
h_xy_p1 = []
h_xy_p2 = []

total_double_tag_events = 0
p1_acc_count = [0, 0, 0, 0]
p2_acc_count = [0, 0, 0, 0]
double_det_count = [0, 0, 0, 0]

for i, (label, (sys_id, lyr_id), xb, yb) in enumerate(STATIONS):
    h_both = ROOT.TH2F(f"h_both_{i}", f"RP {label};X Position [mm];Y Position [mm]", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
    h_xy_both.append(h_both)
    
    hp1 = ROOT.TH2F(f"h_p1_{i}", f"RP {label};X Position [mm];Y Position [mm]", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
    hp1.SetMarkerColor(ROOT.kBlue)
    hp1.SetMarkerStyle(20)
    hp1.SetMarkerSize(0.4)
    h_xy_p1.append(hp1)

    hp2 = ROOT.TH2F(f"h_p2_{i}", f"RP {label};X Position [mm];Y Position [mm]", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
    hp2.SetMarkerColor(ROOT.kGreen + 2)
    hp2.SetMarkerStyle(20)
    hp2.SetMarkerSize(0.4)
    h_xy_p2.append(hp2)

# ==================================================================
# MAIN LOOP
# ==================================================================
print(f"\nStarting chunked processing for {len(file_list)} files...\n")

for f_idx, filename in enumerate(file_list):
    if f_idx == MAX_EVENTS:
        break
    if not os.path.exists(filename):
        continue
        
    print(f"Processing File {f_idx + 1}/{len(file_list)}...")
    tree = uproot.open(f"{filename}:events")

    if "ForwardRomanPotHits.cellID" not in tree:
        continue

    # MC Truth Arrays
    mc_pdg    = tree["MCParticles.PDG"].array()
    mc_status = tree["MCParticles.generatorStatus"].array()
    mc_pz     = tree["MCParticles.momentum.z"].array()

    mc_vz     = tree["MCParticles.vertex.z"].array()

    # Hit Arrays
    idx        = tree["_ForwardRomanPotHits_particle.index"].array()
    hit_cellid = tree["ForwardRomanPotHits.cellID"].array()
    hit_x      = tree["ForwardRomanPotHits.position.x"].array()
    hit_y      = tree["ForwardRomanPotHits.position.y"].array()

    hit_sys, _, hit_layer, _, hit_sensor = decode_rp_cellid(hit_cellid)

    # Ensure index is in range and hit is strictly in the active silicon sensor
    valid_idx = (idx >= 0) & (idx < ak.num(mc_pdg))
    valid_active_hit = valid_idx & (hit_sensor == ACTIVE_SENSOR)

    hit_mc_pdg = ak.where(valid_active_hit, mc_pdg[idx], 0)
    hit_mc_sts = ak.where(valid_active_hit, mc_status[idx], 0)
    hit_mc_vz  = ak.where(valid_active_hit, mc_vz[idx], 0.0)

    # Generated Double Spectator Proton logic (with kinematic cut > 35 GeV/c)
    is_p = (mc_pdg == PROTON) & (mc_status == GEN_STATUS) & (mc_pz > PZ_MIN) &(abs(mc_vz) < VZ_MAX)
    mc_indices = ak.local_index(mc_pdg)
    
    p_indices = mc_indices[is_p]
    has_two_protons = ak.num(p_indices) >= 0
    total_double_tag_events += int(ak.sum(has_two_protons))
    
    if int(ak.sum(has_two_protons)) == 0:
        continue

    p_indices_2 = ak.pad_none(p_indices, 2, axis=1)
    idx_p1 = p_indices_2[:, 0]  
    idx_p2 = p_indices_2[:, 1]

    hit_is_p1 = valid_active_hit & ak.fill_none(idx == idx_p1, False) & has_two_protons
    hit_is_p2 = valid_active_hit & ak.fill_none(idx == idx_p2, False) & has_two_protons

    # Fill Histograms Using True cellID Logic
    for i, (label, (st_sys, st_lyr), xb, yb) in enumerate(STATIONS):
        
        # Logical mask for this exact Roman Pot station layer
        station_mask = (hit_sys == st_sys) & (hit_layer == st_lyr)

        # Fill Proton 1
        m_p1 = hit_is_p1 & station_mask
        ev_has_p1 = ak.any(m_p1, axis=1)
        p1_acc_count[i] += int(ak.sum(ev_has_p1))
        
        x_p1 = ak.to_numpy(ak.flatten(hit_x[m_p1])).astype(np.float64)
        y_p1 = ak.to_numpy(ak.flatten(hit_y[m_p1])).astype(np.float64)
        if len(x_p1): 
            h_xy_p1[i].FillN(len(x_p1), x_p1, y_p1, np.ones(len(x_p1), dtype=np.float64))
            h_xy_both[i].FillN(len(x_p1), x_p1, y_p1, np.ones(len(x_p1), dtype=np.float64))

        # Fill Proton 2
        m_p2 = hit_is_p2 & station_mask
        ev_has_p2 = ak.any(m_p2, axis=1)
        p2_acc_count[i] += int(ak.sum(ev_has_p2))
        
        x_p2 = ak.to_numpy(ak.flatten(hit_x[m_p2])).astype(np.float64)
        y_p2 = ak.to_numpy(ak.flatten(hit_y[m_p2])).astype(np.float64)
        if len(x_p2): 
            h_xy_p2[i].FillN(len(x_p2), x_p2, y_p2, np.ones(len(x_p2), dtype=np.float64))
            h_xy_both[i].FillN(len(x_p2), x_p2, y_p2, np.ones(len(x_p2), dtype=np.float64))
            
        double_det_count[i] += int(ak.sum(ev_has_p1 & ev_has_p2))

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
    ROOT.gPad.SetRightMargin(0.15)
    ROOT.gPad.SetBottomMargin(0.12)
    
    # Draw Combined Hits in COLZ
    h_xy_both[i].GetYaxis().SetTitleOffset(1.3)
    h_xy_both[i].Draw("COLZ")
    
    # Calculate acceptance 
    acc_p1 = p1_acc_count[i] / total_double_tag_events if total_double_tag_events > 0 else 0.0
    acc_p2 = p2_acc_count[i] / total_double_tag_events if total_double_tag_events > 0 else 0.0
    acc_both = double_det_count[i] / total_double_tag_events if total_double_tag_events > 0 else 0.0
    
    # Legend
    leg = ROOT.TLegend(0.15, 0.72, 0.70, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(1001)
    leg.SetFillColorAlpha(ROOT.kWhite, 0.75)
    leg.SetTextSize(0.035)
    leg.SetTextFont(62)
    leg.SetTextColor(ROOT.kRed + 1)
    leg.AddEntry(0, f"P1 Hits: {int(h_xy_p1[i].GetEntries())} (Eff: {acc_p1*100:.2f}%)", "")
    leg.AddEntry(0, f"P2 Hits: {int(h_xy_p2[i].GetEntries())} (Eff: {acc_p2*100:.2f}%)", "")
    leg.AddEntry(0, f"Double Sepctator Eff: {acc_both*100:.2f}% ({double_det_count[i]}/{total_double_tag_events})", "")
    leg.Draw()
    legends.append(leg)
    
c_master.SaveAs(f"{OUT_DIR}/rompot_2D_hits_protons2_cellID.pdf")
print(f"Total Double Tagging Events: {total_double_tag_events}")
print("Done!")