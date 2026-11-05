import os
import ROOT
import numpy as np
import awkward as ak
import uproot
from pathlib import Path

MAX_FILES = 50 
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kBird)
ROOT.TH1.AddDirectory(False)

dir_path = Path("/work/eic/users/aabhishe/Recon_out_1M_omd_b0")
OUT_DIR = "/home/aabhishe/eic/EIC_Full_Sim/Plots/OMD_Protons_layer"
os.makedirs(OUT_DIR, exist_ok=True)

PROTON     = 2212
GEN_STATUS = 1
PZ_MIN     = 35.0
OFFSET_X   = 20.0

file_indices = sorted(
    int(p.stem.split("_")[-1]) for p in dir_path.glob("recon_chunk_*.root")
    if p.stem.split("_")[-1].isdigit()
)
file_list = [str(dir_path / f"recon_chunk_{i}.root") for i in file_indices]
print(f"Found {len(file_list)} files to process in {dir_path}.")
print(f"Plots will be saved to: {OUT_DIR}")

def decode_omd_cellid(cellID):
    """Vectorized extraction of OMD bitfields from uint64 cellID array."""
    sys_id = (cellID >> 0) & 0xFF
    layer  = (cellID >> 8) & 0x1F
    module = (cellID >> 13) & 0x1F
    slc    = (cellID >> 18) & 0xF
    return sys_id, layer, module, slc

def discover_omd_station_system_ids(files):
    """
    Finds the 4 unique OMD 'system' IDs from the first file with hits
    and orders them from Station 1 to Station 4.
    """
    for fpath in files:
        if not os.path.exists(fpath):
            continue
        tree = uproot.open(f"{fpath}:events")
        if "ForwardOffMTrackerHits.cellID" not in tree:
            continue
        cids = ak.flatten(tree["ForwardOffMTrackerHits.cellID"].array())
        zs   = ak.flatten(tree["ForwardOffMTrackerHits.position.z"].array())
        if len(cids) == 0:
            continue
        sys_ids = ak.to_numpy((cids >> 0) & 0xFF)
        z_vals  = ak.to_numpy(zs)
        unique_sys = np.unique(sys_ids)
        sys_sorted_by_z = sorted(unique_sys, key=lambda s: np.mean(z_vals[sys_ids == s]))
        if len(sys_sorted_by_z) >= 4:
            return sys_sorted_by_z[:4]
    raise RuntimeError("Could not auto-discover 4 OMD station system IDs from the input files.")

STATION_SYS_IDS = discover_omd_station_system_ids(file_list)
print(f"Discovered OMD Station system IDs (Stations 1-4): {STATION_SYS_IDS}")

x12_c = -941.0 + OFFSET_X
x34_c = -1032.0 + OFFSET_X

STATIONS = [
    (f"OMD Layer 1 ( z=25500 mm)", STATION_SYS_IDS[0], x12_c, (160, x12_c - 80., x12_c + 80.), (240, -120., 120.)),
    (f"OMD Layer 2 ( z=25520 mm)", STATION_SYS_IDS[1], x12_c, (160, x12_c - 80., x12_c + 80.), (240, -120., 120.)),
    (f"OMD Layer 3 ( z=27000 mm)", STATION_SYS_IDS[2], x34_c, (160, x34_c - 80., x34_c + 80.), (240, -120., 120.)),
    (f"OMD Layer 4 ( z=27020 mm)", STATION_SYS_IDS[3], x34_c, (160, x34_c - 80., x34_c + 80.), (240, -120., 120.)),
]

h_xy_both = []
h_xy_p1   = []
h_xy_p2   = []

total_double_spectator_events = 0
p1_acc_count        = [0, 0, 0, 0]
p2_acc_count        = [0, 0, 0, 0]
double_det_count    = [0, 0, 0, 0]
double_coin_all4    = 0
double_any_station  = 0

for i, (label, sys_id, xc, xb, yb) in enumerate(STATIONS):
    h_both = ROOT.TH2F(f"h_both_{i}", f"{label};X[mm];Y[mm]", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
    h_xy_both.append(h_both)

    hp1 = ROOT.TH2F(f"h_p1_{i}", f"{label};X[mm];Y[mm]", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
    hp1.SetMarkerColor(ROOT.kBlue)
    hp1.SetMarkerStyle(20)
    hp1.SetMarkerSize(0.4)
    h_xy_p1.append(hp1)

    hp2 = ROOT.TH2F(f"h_p2_{i}", f"{label};X[mm];Y[mm]", xb[0], xb[1], xb[2], yb[0], yb[1], yb[2])
    hp2.SetMarkerColor(ROOT.kRed)
    hp2.SetMarkerStyle(20)
    hp2.SetMarkerSize(0.4)
    h_xy_p2.append(hp2)

print(f"\nStarting chunked processing for {len(file_list)} files...\n")

for f_idx, filename in enumerate(file_list):
    if MAX_FILES > 0 and f_idx >= MAX_FILES:
        break
    if not os.path.exists(filename):
        continue

    print(f"Processing File {f_idx + 1}/{len(file_list)}: {os.path.basename(filename)}...")
    try:
        tree = uproot.open(f"{filename}:events")
    except Exception as e:
        print(f"  -> Error opening {filename}: {e}")
        continue

    mc_pdg    = tree["MCParticles.PDG"].array()
    mc_status = tree["MCParticles.generatorStatus"].array()
    mc_pz     = tree["MCParticles.momentum.z"].array()

    is_spectator_p = (mc_pdg == PROTON) & (mc_status == GEN_STATUS) & (mc_pz > PZ_MIN)
    mc_indices     = ak.local_index(mc_pdg)
    p_indices      = mc_indices[is_spectator_p]

    has_two_protons = ak.num(p_indices) >= 2
    n_double_gen    = int(ak.sum(has_two_protons))
    total_double_spectator_events += n_double_gen

    if n_double_gen == 0 or "ForwardOffMTrackerHits.cellID" not in tree:
        continue

    p_indices_2 = ak.pad_none(p_indices, 2, axis=1)
    idx_p1      = p_indices_2[:, 0]
    idx_p2      = p_indices_2[:, 1]

    idx        = tree["_ForwardOffMTrackerHits_particle.index"].array()
    hit_cellid = tree["ForwardOffMTrackerHits.cellID"].array()
    hit_x      = tree["ForwardOffMTrackerHits.position.x"].array()
    hit_y      = tree["ForwardOffMTrackerHits.position.y"].array()

    hit_sys, hit_layer, hit_module, hit_slice = decode_omd_cellid(hit_cellid)

    valid_idx = (idx >= 0) & (idx < ak.num(mc_pdg))
    valid_lgad_hit = valid_idx & (hit_layer == 1) & (hit_module == 1) & (hit_slice == 1)

    hit_is_p1 = valid_lgad_hit & ak.fill_none(idx == idx_p1, False) & has_two_protons
    hit_is_p2 = valid_lgad_hit & ak.fill_none(idx == idx_p2, False) & has_two_protons

    both_detected_per_station = []
    p1_any_station = ak.zeros_like(has_two_protons, dtype=bool)
    p2_any_station = ak.zeros_like(has_two_protons, dtype=bool)

    for i, (label, sys_id, xc, xb, yb) in enumerate(STATIONS):
        st_mask = (hit_sys == sys_id)

        m_p1 = hit_is_p1 & st_mask
        m_p2 = hit_is_p2 & st_mask

        ev_has_p1 = ak.any(m_p1, axis=1)
        ev_has_p2 = ak.any(m_p2, axis=1)
        ev_has_both = ev_has_p1 & ev_has_p2

        p1_acc_count[i]     += int(ak.sum(ev_has_p1))
        p2_acc_count[i]     += int(ak.sum(ev_has_p2))
        double_det_count[i] += int(ak.sum(ev_has_both))

        both_detected_per_station.append(ev_has_both)
        p1_any_station = p1_any_station | ev_has_p1
        p2_any_station = p2_any_station | ev_has_p2

        x_p1 = ak.to_numpy(ak.flatten(hit_x[m_p1])).astype(np.float64)
        y_p1 = ak.to_numpy(ak.flatten(hit_y[m_p1])).astype(np.float64)
        if len(x_p1):
            w1 = np.ones(len(x_p1), dtype=np.float64)
            h_xy_p1[i].FillN(len(x_p1), x_p1, y_p1, w1)
            h_xy_both[i].FillN(len(x_p1), x_p1, y_p1, w1)

        x_p2 = ak.to_numpy(ak.flatten(hit_x[m_p2])).astype(np.float64)
        y_p2 = ak.to_numpy(ak.flatten(hit_y[m_p2])).astype(np.float64)
        if len(x_p2):
            w2 = np.ones(len(x_p2), dtype=np.float64)
            h_xy_p2[i].FillN(len(x_p2), x_p2, y_p2, w2)
            h_xy_both[i].FillN(len(x_p2), x_p2, y_p2, w2)

    ev_coin_all4 = (
        both_detected_per_station[0] &
        both_detected_per_station[1] &
        both_detected_per_station[2] &
        both_detected_per_station[3]
    )
    double_coin_all4   += int(ak.sum(ev_coin_all4))
    double_any_station += int(ak.sum(p1_any_station & p2_any_station))

print(f"\n{'='*70}")
print(" OMD DOUBLE SPECTATOR PROTON EFFICIENCY SUMMARY")
print(f"{'='*70}")
print(f"Total Generated Double Spectator Events (Pz > {PZ_MIN} GeV/c) : {total_double_spectator_events}\n")

if total_double_spectator_events > 0:
    for i, (label, sys_id, _, _, _) in enumerate(STATIONS):
        eff_p1   = (p1_acc_count[i] / total_double_spectator_events) * 100.0
        eff_p2   = (p2_acc_count[i] / total_double_spectator_events) * 100.0
        eff_both = (double_det_count[i] / total_double_spectator_events) * 100.0
        print(f"Station {i+1} (sys={sys_id}) | P1 Acc: {eff_p1:6.2f}% | P2 Acc: {eff_p2:6.2f}% | "
              f"Double Spectator Eff: {eff_both:6.2f}%")

    eff_coin4 = (double_coin_all4 / total_double_spectator_events) * 100.0
    eff_any   = (double_any_station / total_double_spectator_events) * 100.0
    print("-" * 70)
    print(f"Coincidence of All 4 Stations (Both P1 & P2 in S1-S4) : {double_coin_all4:5d} / {total_double_spectator_events} ({eff_coin4:6.2f}%)")
    print(f"Inclusive OMD (Both P1 & P2 in >=1 Station)           : {double_any_station:5d} / {total_double_spectator_events} ({eff_any:6.2f}%)")
else:
    print("No generated double spectator events found!")
print(f"{'='*70}\n")

print("[plots] Generating canvas...")

c_master = ROOT.TCanvas("c_master", "OMD Double Spectator Proton Hits", 1400, 1100)
c_master.Divide(2, 2)
legends = []
sensor_boxes = []

for i, (label, sys_id, xc, xb, yb) in enumerate(STATIONS):
    pad = c_master.cd(i + 1)
    pad.SetLeftMargin(0.12)
    pad.SetRightMargin(0.14)
    pad.SetBottomMargin(0.12)
    h_xy_both[i].GetYaxis().SetTitleOffset(1.3)
    h_xy_both[i].Draw("COLZ")

    box = ROOT.TBox(xc - 50.0, -100.0, xc + 50.0, 100.0)
    box.SetFillStyle(0)
    box.SetLineColor(ROOT.kRed)
    box.SetLineStyle(2)
    box.SetLineWidth(1)
    box.Draw("SAME")
    sensor_boxes.append(box)

    eff_p1   = (p1_acc_count[i] / total_double_spectator_events) * 100.0 if total_double_spectator_events > 0 else 0.0
    eff_p2   = (p2_acc_count[i] / total_double_spectator_events) * 100.0 if total_double_spectator_events > 0 else 0.0
    eff_both = (double_det_count[i] / total_double_spectator_events) * 100.0 if total_double_spectator_events > 0 else 0.0

    leg = ROOT.TLegend(0.3, 0.74, 0.75, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(1001)
    leg.SetFillColorAlpha(ROOT.kWhite, 0.85)
    leg.SetTextSize(0.032)
    leg.SetTextFont(62)
    leg.SetTextColor(ROOT.kRed)
    leg.AddEntry(0, f"P1 Hits: {int(h_xy_p1[i].GetEntries())} (Acc: {eff_p1:.2f}%)", "")
    leg.AddEntry(0, f"P2 Hits: {int(h_xy_p2[i].GetEntries())} (Acc: {eff_p2:.2f}%)", "")
    leg.AddEntry(0, f"Double Spectator Eff: {eff_both:.2f}%", "")
    leg.Draw()
    legends.append(leg)

out_pdf = f"{OUT_DIR}/omd_2D_hits_double_spectators.pdf"
out_png = f"{OUT_DIR}/omd_2D_hits_double_spectators.png"
c_master.SaveAs(out_pdf)
c_master.SaveAs(out_png)
print(f"Saved plots to:\n  - {out_pdf}\n  - {out_png}")