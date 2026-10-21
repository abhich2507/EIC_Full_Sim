import uproot
import awkward as ak
import numpy as np
import vector
import ROOT
from pathlib import Path

vector.register_awkward()
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Constants
DEUTERON_PDG = 1000010020
HE3_PDG      = 1000020030
ELECTRON_PDG = 11

REC_GEN_STATUS  = 1  # Final-state MC particles
BEAM_GEN_STATUS = 4  # Beam MC particles

ELECTRON_MASS = 0.000511  # GeV
HE3_MASS      = 2.8084    # GeV

# Define bin edges
Q2_bin_edges = np.logspace(0, 2, 101)
x_bin_edges  = np.logspace(-3, 0, 101)
theta_bin_edges = np.linspace(0, 180, 101)

# Initialize 1D Histograms
h_p_with_d  = ROOT.TH1F("h_p_with_d",  "Electron Momentum;p_{e} [GeV];Events", 60, 0, 10)
h_p_no_d    = ROOT.TH1F("h_p_no_d",    "Electron Momentum;p_{e} [GeV];Events", 60, 0, 10)

h_Q2_with_d = ROOT.TH1F("h_Q2_with_d", "Q^{2} Distribution;Q^{2} [GeV^{2}];Events", len(Q2_bin_edges) - 1, Q2_bin_edges)
h_Q2_no_d   = ROOT.TH1F("h_Q2_no_d",   "Q^{2} Distribution;Q^{2} [GeV^{2}];Events", len(Q2_bin_edges) - 1, Q2_bin_edges)

h_x_with_d  = ROOT.TH1F("h_x_with_d",  "Bjorken x Distribution;x_{B};Events", len(x_bin_edges) - 1, x_bin_edges)
h_x_no_d    = ROOT.TH1F("h_x_no_d",    "Bjorken x Distribution;x_{B};Events", len(x_bin_edges) - 1, x_bin_edges)

h_theta_with_d  = ROOT.TH1F("h_theta_with_d",  "Electron Polar Angle;#theta [deg];Events", len(theta_bin_edges) - 1, theta_bin_edges)
h_theta_no_d    = ROOT.TH1F("h_theta_no_d",    "Electron Polar Angle;#theta [deg];Events", len(theta_bin_edges) - 1, theta_bin_edges)

# Initialize 2D Histograms
h2_x_Q2_with_d = ROOT.TH2F("h2_x_Q2_with_d", "x_{B} vs Q^{2} (e- With RP deuteron coin);Q^{2} [GeV^{2}];x_{B}", len(Q2_bin_edges) - 1, Q2_bin_edges, len(x_bin_edges) - 1, x_bin_edges)
h2_x_Q2_no_d   = ROOT.TH2F("h2_x_Q2_no_d",   "x_{B} vs Q^{2} (e- Without RP deuteron coin);Q^{2} [GeV^{2}];x_{B}", len(Q2_bin_edges) - 1, Q2_bin_edges, len(x_bin_edges) - 1, x_bin_edges)

for h in (h_p_with_d, h_Q2_with_d, h_x_with_d, h_theta_with_d):
    h.SetLineColor(ROOT.kRed)
    h.SetLineWidth(2)

for h in (h_p_no_d, h_Q2_no_d, h_x_no_d, h_theta_no_d):
    h.SetLineColor(ROOT.kBlue)
    h.SetLineWidth(2)

# File list setup
dir_path  = Path("/work/eic/users/aabhishe/Recon_out_1M")
file_list = list(dir_path.glob("recon_chunk_*.root"))

total_events = 0
total_e_with = 0
total_e_no   = 0

for f_dx, filepath in enumerate(file_list):
    # if f_dx == 10:  # Remove this break to run on all files
    #     break
    filename = str(filepath)
    print(f"Processing file {f_dx+1}/{len(file_list)}: {filename}")
    
    with uproot.open(filename) as file:
        tree = file["events"]

        # 1. Roman Pot Deuteron Coincidence
        mc_pdg    = tree["MCParticles.PDG"].array()
        mc_status = tree["MCParticles.generatorStatus"].array()
        rp_mc_idx = tree["_ForwardRomanPotHits_particle.index"].array()
        rp_z      = tree["ForwardRomanPotHits.position.z"].array()

        hit_pdg    = mc_pdg[rp_mc_idx]
        hit_status = mc_status[rp_mc_idx]
        is_d_hit   = (hit_pdg == DEUTERON_PDG) & (hit_status == REC_GEN_STATUS)

        has_station_1 = ak.any(is_d_hit & (rp_z < 32600.0), axis=1)
        has_station_2 = ak.any(is_d_hit & (rp_z > 34200.0), axis=1)
        event_has_rp_deuteron = has_station_1 & has_station_2

        # 2. Beam Vectors
        mc_px = tree["MCParticles.momentum.x"].array()
        mc_py = tree["MCParticles.momentum.y"].array()
        mc_pz = tree["MCParticles.momentum.z"].array()

        e_beam_mask = (mc_pdg == ELECTRON_PDG) & (mc_status == BEAM_GEN_STATUS)
        e_beam_E    = np.sqrt(mc_px[e_beam_mask]**2 + mc_py[e_beam_mask]**2 + mc_pz[e_beam_mask]**2 + ELECTRON_MASS**2)
        e_beam      = vector.zip({"px": mc_px[e_beam_mask], "py": mc_py[e_beam_mask], "pz": mc_pz[e_beam_mask], "E": e_beam_E})

        p_beam_mask = (mc_pdg == HE3_PDG) & (mc_status == BEAM_GEN_STATUS)
        p_beam_E    = np.sqrt(mc_px[p_beam_mask]**2 + mc_py[p_beam_mask]**2 + mc_pz[p_beam_mask]**2 + HE3_MASS**2)
        p_beam      = vector.zip({"px": mc_px[p_beam_mask], "py": mc_py[p_beam_mask], "pz": mc_pz[p_beam_mask], "E": p_beam_E})

        # 3. Reconstructed Electrons
        rec_pdg      = tree["ReconstructedChargedParticles.PDG"].array()
        e_prime_mask = (rec_pdg == ELECTRON_PDG)

        e_prime_px = tree["ReconstructedChargedParticles.momentum.x"].array()[e_prime_mask]
        e_prime_py = tree["ReconstructedChargedParticles.momentum.y"].array()[e_prime_mask]
        e_prime_pz = tree["ReconstructedChargedParticles.momentum.z"].array()[e_prime_mask]
        e_prime_E  = tree["ReconstructedChargedParticles.energy"].array()[e_prime_mask]
        e_prime    = vector.zip({"px": e_prime_px, "py": e_prime_py, "pz": e_prime_pz, "E": e_prime_E})
        
        # 4. Kinematics for Leading Electron
        e_prime_sorted  = e_prime[ak.argsort(e_prime.E, ascending=False)]
        e_prime_leading = ak.firsts(e_prime_sorted)
        e_beam_single   = ak.firsts(e_beam)
        p_beam_single   = ak.firsts(p_beam)

        # Calculate everything based ONLY on the leading electron
        elec_p     = e_prime_leading.p
        elec_pt    = np.sqrt(e_prime_leading.px**2 + e_prime_leading.py**2)
        elec_theta = 180.0 - np.degrees(np.arctan2(elec_pt, e_prime_leading.pz))
        
        q      = e_beam_single - e_prime_leading
        Q2     = -q.mass2
        x_B    = 3.0 * Q2 / (2.0 * p_beam_single.dot(q))

        # 5. Split and Fill (No flattening needed since everything is exactly 1-per-event now)
        p_with_d  = ak.to_numpy(ak.drop_none(elec_p[event_has_rp_deuteron])).astype(np.float64)
        p_no_d    = ak.to_numpy(ak.drop_none(elec_p[~event_has_rp_deuteron])).astype(np.float64)

        Q2_with_d = ak.to_numpy(ak.drop_none(Q2[event_has_rp_deuteron])).astype(np.float64)
        Q2_no_d   = ak.to_numpy(ak.drop_none(Q2[~event_has_rp_deuteron])).astype(np.float64)

        x_with_d  = ak.to_numpy(ak.drop_none(x_B[event_has_rp_deuteron])).astype(np.float64)
        x_no_d    = ak.to_numpy(ak.drop_none(x_B[~event_has_rp_deuteron])).astype(np.float64)

        theta_with_d  = ak.to_numpy(ak.drop_none(elec_theta[event_has_rp_deuteron])).astype(np.float64)
        theta_no_d    = ak.to_numpy(ak.drop_none(elec_theta[~event_has_rp_deuteron])).astype(np.float64)

        if len(p_with_d) > 0:  h_p_with_d.FillN(len(p_with_d), p_with_d, np.ones(len(p_with_d)))
        if len(p_no_d) > 0:    h_p_no_d.FillN(len(p_no_d), p_no_d, np.ones(len(p_no_d)))
        
        if len(Q2_with_d) > 0: h_Q2_with_d.FillN(len(Q2_with_d), Q2_with_d, np.ones(len(Q2_with_d)))
        if len(Q2_no_d) > 0:   h_Q2_no_d.FillN(len(Q2_no_d), Q2_no_d, np.ones(len(Q2_no_d)))
        
        if len(x_with_d) > 0:  h_x_with_d.FillN(len(x_with_d), x_with_d, np.ones(len(x_with_d)))
        if len(x_no_d) > 0:    h_x_no_d.FillN(len(x_no_d), x_no_d, np.ones(len(x_no_d)))
        
        if len(theta_with_d) > 0:  h_theta_with_d.FillN(len(theta_with_d), theta_with_d, np.ones(len(theta_with_d)))
        if len(theta_no_d) > 0:    h_theta_no_d.FillN(len(theta_no_d), theta_no_d, np.ones(len(theta_no_d)))
        
        if len(x_with_d) > 0 and len(Q2_with_d) > 0: 
            h2_x_Q2_with_d.FillN(len(x_with_d), Q2_with_d, x_with_d, np.ones(len(x_with_d)))
        if len(x_no_d) > 0 and len(Q2_no_d) > 0: 
            h2_x_Q2_no_d.FillN(len(x_no_d), Q2_no_d, x_no_d, np.ones(len(x_no_d)))

        total_events += len(event_has_rp_deuteron)
        total_e_with += len(p_with_d)
        total_e_no   += len(p_no_d)

print(f"\nTotal events processed: {total_events}")
print(f"Leading Electrons WITH RP deuteron coincidence:    {total_e_with}")
print(f"Leading Electrons WITHOUT RP deuteron coincidence: {total_e_no}")
if (total_e_no + total_e_with) > 0:
    print(f"Fraction of electrons with RP deuteron coincidence: {total_e_with/(total_e_no+total_e_with):.4f}")



for h in (h_p_with_d, h_p_no_d, h_Q2_with_d, h_Q2_no_d, h_x_with_d, h_x_no_d, h_theta_with_d, h_theta_no_d):
    h.GetXaxis().SetTitleSize(0.05)
    h.GetYaxis().SetTitleSize(0.05)
   # h.GetXaxis().SetTitleOffset(1.1)  # Pushes the X-axis title down slightly
    #h.GetYaxis().SetTitleOffset(1.3)  # Pushes the Y-axis title left slightly
# ==========================================
# Canvas 1: 1D Kinematics
# ==========================================
c1 = ROOT.TCanvas("c1", "Kinematics with/without RP Deuteron", 1500, 1500)

# 1. Create a sub-pad that stops at y=0.92, leaving the top 8% of c1 blank
main_pad = ROOT.TPad("main_pad", "main_pad", 0.0, 0.0, 1.0, 0.92)
main_pad.Draw()
main_pad.Divide(2, 2)

# 2. Draw in the sub-pads (use main_pad.cd instead of c1.cd)
main_pad.cd(1); ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15); ROOT.gPad.SetRightMargin(0.05); ROOT.gPad.SetBottomMargin(0.15)
h_p_no_d.Draw("HIST")
h_p_with_d.Draw("HIST SAME")


main_pad.cd(2); ROOT.gPad.SetLogx(); ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15); ROOT.gPad.SetRightMargin(0.05); ROOT.gPad.SetBottomMargin(0.15)
h_Q2_no_d.Draw("HIST")
h_Q2_with_d.Draw("HIST SAME")

main_pad.cd(3); ROOT.gPad.SetLogx(); ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15); ROOT.gPad.SetRightMargin(0.05); ROOT.gPad.SetBottomMargin(0.15)
h_x_no_d.Draw("HIST")
h_x_with_d.Draw("HIST SAME")

main_pad.cd(4); ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15); ROOT.gPad.SetRightMargin(0.05); ROOT.gPad.SetBottomMargin(0.15)
h_theta_no_d.Draw("HIST")
h_theta_with_d.Draw("HIST SAME")

# 3. Draw Global Legend ON the main canvas (c1) in the reserved empty space
c1.cd(0)  
leg = ROOT.TLegend(0.02, 0.93, 0.95, 0.99) # Fits perfectly in the >0.92 area
leg.SetTextSize(0.025)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetNColumns(2)
leg.AddEntry(h_p_no_d,   "e^{-} (Without RP deuteron Coincidence)", "l")
leg.AddEntry(h_p_with_d, "e^{-} (With RP deuteron Coincidence)", "l")
leg.Draw()

# Open PDF and save first canvas
c1.Print("../Plots/recon_electrons_rp_coincidence.pdf[")
c1.Print("../Plots/recon_electrons_rp_coincidence.pdf")

for h in (h2_x_Q2_with_d, h2_x_Q2_no_d):
    h.GetXaxis().SetTitleSize(0.05)
    h.GetYaxis().SetTitleSize(0.05)
    h.GetXaxis().SetTitleOffset(1.1)  # Pushes the X-axis title down slightly
    h.GetYaxis().SetTitleOffset(1.3)  # Pushes the Y-axis title left slightly

c2 = ROOT.TCanvas("c2", "2D Kinematics", 1800, 900)
c2.Divide(2, 1)

c2.cd(1)
ROOT.gPad.SetLeftMargin(0.15)   # Creates space on the left for the larger Y-axis title
ROOT.gPad.SetRightMargin(0.15)  # Keeps the space on the right for the color bar
ROOT.gPad.SetBottomMargin(0.15)  # Creates space on the bottom for the larger X-axis title
ROOT.gPad.SetLogx(); ROOT.gPad.SetLogy(); ROOT.gPad.SetLogz()
h2_x_Q2_with_d.Draw("COLZ")

c2.cd(2)
ROOT.gPad.SetLeftMargin(0.15)   
ROOT.gPad.SetRightMargin(0.15)  
ROOT.gPad.SetBottomMargin(0.15)  # Creates space on the bottom for the larger X-axis title
ROOT.gPad.SetLogx(); ROOT.gPad.SetLogy(); ROOT.gPad.SetLogz()
h2_x_Q2_no_d.Draw("COLZ")

# Save second canvas and close PDF
c2.Print("../Plots/recon_electrons_rp_coincidence.pdf")
c1.Print("../Plots/recon_electrons_rp_coincidence.pdf]")