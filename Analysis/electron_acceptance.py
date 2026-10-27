import uproot
import awkward as ak
import numpy as np
import vector
import ROOT
from pathlib import Path

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
# Make the main pad titles larger
ROOT.gStyle.SetTitleFontSize(0.06)  

max_events = -1

# Constants
ELECTRON_PDG   = 11
REC_GEN_STATUS = 1  # Final-state MC particles

MASS_ELECTRON = 0.000511  # GeV
MASS_HELIUM3  = 2.8084    # GeV

#Luminosity and cross-section for normalization
TARGET_IL = 1500000.0  # 1.5 fb-1 converted to /nb
GENERATED_IL = 4914.32 # /nb (from generator output)
EVENT_WEIGHT = TARGET_IL / GENERATED_IL  # ~305.23

# Optimized Binning
nBins_lin = 50
nBins_log = 40
x_bins  = np.logspace(-3, 0, nBins_log + 1)
Q2_bins = np.logspace(0, 3, nBins_log + 1)

# Initialize Histograms
h_gen_p = ROOT.TH1F("h_gen_p", "Momentum;p_{e} [GeV];Electrons", nBins_lin, 0, 10)
h_rec_p = ROOT.TH1F("h_rec_p", "Momentum;p_{e} [GeV];Electrons", nBins_lin, 0, 10)

h_gen_pT = ROOT.TH1F("h_gen_pT", "Electron Transverse Momentum;p_{e}^{T} [GeV];Electrons", nBins_lin, 0, 10)
h_rec_pT = ROOT.TH1F("h_rec_pT", "Electron Transverse Momentum;p_{e}^{T} [GeV];Electrons", nBins_lin, 0, 10)

h_gen_theta = ROOT.TH1F("h_gen_theta", "Polar Angle;#theta [deg];Electrons", nBins_lin, 0, 180)
h_rec_theta = ROOT.TH1F("h_rec_theta", "Polar Angle;#theta [deg];Electrons", nBins_lin, 0, 180)

h_gen_Q2 = ROOT.TH1F("h_gen_Q2", " Q^{2};Q^{2} [GeV^{2}];Electrons", nBins_log, Q2_bins)
h_rec_Q2 = ROOT.TH1F("h_rec_Q2", " Q^{2};Q^{2} [GeV^{2}];Electrons", nBins_log, Q2_bins)

h_gen_x = ROOT.TH1F("h_gen_x", " x_{B};x_{B};Electrons", nBins_log, x_bins)
h_rec_x = ROOT.TH1F("h_rec_x", "x_{B};x_{B};Electrons", nBins_log, x_bins)

for h in (h_gen_p, h_gen_theta, h_gen_Q2, h_gen_x, h_gen_pT):
    h.SetLineWidth(2)
    h.SetLineColor(ROOT.kRed)
for h in (h_rec_p, h_rec_theta, h_rec_Q2, h_rec_x, h_rec_pT):
    h.SetLineWidth(2)
    h.SetLineColor(ROOT.kBlue)

# Apply global styling to all histograms for bigger text and offsets
for h in (h_gen_p, h_gen_theta, h_gen_Q2, h_gen_x, h_gen_pT, h_rec_p, h_rec_theta, h_rec_Q2, h_rec_x, h_rec_pT):
    h.Sumw2()  
    h.GetXaxis().SetTitleSize(0.06)
    h.GetYaxis().SetTitleSize(0.06)
    h.GetXaxis().SetLabelSize(0.065)
    h.GetYaxis().SetLabelSize(0.065)
    
    # Increased offsets to prevent the larger text from hitting the numbers
    h.GetXaxis().SetTitleOffset(1.2)
    h.GetYaxis().SetTitleOffset(1.4)
    
    h.GetXaxis().CenterTitle(True)
    h.GetYaxis().CenterTitle(True)

# File list setup
dir_path  = Path("/work/eic/users/aabhishe/Recon_out_1M")
file_list = list(dir_path.glob("recon_chunk_*.root"))

generated_electron_count = 0
detected_electron_count  = 0

for f_dx, filepath in enumerate(file_list):
    if f_dx == max_events:
        break
    filename = str(filepath)
    print(f"Processing file {f_dx+1}/{len(file_list)}: {filename}")
    
    with uproot.open(filename) as file:
        tree = file["events"]

        # 1. Generated MC Electrons & Beam Particles
        mc_pdg    = tree["MCParticles.PDG"].array()
        mc_status = tree["MCParticles.generatorStatus"].array()

        e_beam_mask = (mc_pdg == ELECTRON_PDG) & (mc_status == 4)
        e_beam_px = tree["MCParticles.momentum.x"].array()[e_beam_mask]
        e_beam_py = tree["MCParticles.momentum.y"].array()[e_beam_mask]
        e_beam_pz = tree["MCParticles.momentum.z"].array()[e_beam_mask]
        e_beam_E = np.sqrt(e_beam_px**2 + e_beam_py**2 + e_beam_pz**2 + MASS_ELECTRON**2)
        e_beam = vector.zip({"px": e_beam_px, "py": e_beam_py, "pz": e_beam_pz, "E": e_beam_E})
        e_beam_single = ak.firsts(e_beam)
        
        e_prime_mc_mask = (mc_pdg == ELECTRON_PDG) & (mc_status == REC_GEN_STATUS)
        mc_px = tree["MCParticles.momentum.x"].array()[e_prime_mc_mask]
        mc_py = tree["MCParticles.momentum.y"].array()[e_prime_mc_mask]
        mc_pz = tree["MCParticles.momentum.z"].array()[e_prime_mc_mask]
        mc_pT = np.sqrt(mc_px**2 + mc_py**2)
        mc_theta = 180.0 - np.degrees(np.arctan2(mc_pT, mc_pz))
        mc_p = np.sqrt(mc_px**2 + mc_py**2 + mc_pz**2)
        mc_E = np.sqrt(mc_px**2 + mc_py**2 + mc_pz**2 + MASS_ELECTRON**2)
        mc_e_prime = vector.zip({"px": mc_px, "py": mc_py, "pz": mc_pz, "E": mc_E})
        mc_prime_leading = ak.firsts(mc_e_prime[ak.argsort(mc_e_prime.E, axis=1, ascending=False)])

        # 2. Reconstructed Electrons
        rec_pdg = tree["ReconstructedChargedParticles.PDG"].array()
        e_prime_mask = (rec_pdg == ELECTRON_PDG)
        rec_px = tree["ReconstructedChargedParticles.momentum.x"].array()[e_prime_mask]
        rec_py = tree["ReconstructedChargedParticles.momentum.y"].array()[e_prime_mask]
        rec_pz = tree["ReconstructedChargedParticles.momentum.z"].array()[e_prime_mask]
        rec_pT = np.sqrt(rec_px**2 + rec_py**2)
        rec_theta = 180.0 - np.degrees(np.arctan2(rec_pT, rec_pz))
        rec_p = np.sqrt(rec_px**2 + rec_py**2 + rec_pz**2)
        rec_E = np.sqrt(rec_px**2 + rec_py**2 + rec_pz**2 + MASS_ELECTRON**2)
        rec_e_prime = vector.zip({"px": rec_px, "py": rec_py, "pz": rec_pz, "E": rec_E})
        rec_prime_leading = ak.firsts(rec_e_prime[ak.argsort(rec_e_prime.E, axis=1, ascending=False)])

        # Ion Beam setup
        p_beam_mask = (mc_pdg == 1000020030) & (mc_status == 4)
        p_beam_px = tree["MCParticles.momentum.x"].array()[p_beam_mask]
        p_beam_py = tree["MCParticles.momentum.y"].array()[p_beam_mask]
        p_beam_pz = tree["MCParticles.momentum.z"].array()[p_beam_mask]
        p_beam_E = np.sqrt(p_beam_px**2 + p_beam_py**2 + p_beam_pz**2 + MASS_HELIUM3**2)
        p_beam = vector.zip({"px": p_beam_px, "py": p_beam_py, "pz": p_beam_pz, "E": p_beam_E}) 
        p_beam_single = ak.firsts(p_beam)

        # Kinematics
        mc_q = (e_beam_single - mc_prime_leading)
        mc_Q2 = -mc_q.mass2
        rec_q = (e_beam_single - rec_prime_leading)
        rec_Q2 = -rec_q.mass2

        mc_x = 3.0 * mc_Q2 / (2 * mc_q.dot(p_beam_single))
        rec_x = 3.0 * rec_Q2 / (2 * rec_q.dot(p_beam_single))

        # 3. Update Counters
        generated_electron_count += ak.sum(ak.num(mc_p))
        detected_electron_count  += ak.sum(ak.num(rec_p))

        # 4. Flatten and Fill
        flat_mc_p      = ak.to_numpy(ak.drop_none(ak.flatten(mc_p))).astype(np.float64)
        flat_rec_p     = ak.to_numpy(ak.drop_none(ak.flatten(rec_p))).astype(np.float64)
        flat_mc_pT     = ak.to_numpy(ak.drop_none(ak.flatten(mc_pT))).astype(np.float64)
        flat_rec_pT    = ak.to_numpy(ak.drop_none(ak.flatten(rec_pT))).astype(np.float64)
        flat_mc_theta  = ak.to_numpy(ak.drop_none(ak.flatten(mc_theta))).astype(np.float64)
        flat_rec_theta = ak.to_numpy(ak.drop_none(ak.flatten(rec_theta))).astype(np.float64)
        flat_mc_Q2     = ak.to_numpy(ak.drop_none(mc_Q2)).astype(np.float64)
        flat_rec_Q2    = ak.to_numpy(ak.drop_none(rec_Q2)).astype(np.float64)
        flat_mc_x      = ak.to_numpy(ak.drop_none(mc_x)).astype(np.float64)
        flat_rec_x     = ak.to_numpy(ak.drop_none(rec_x)).astype(np.float64)

        weights=np.full_like(flat_mc_p, EVENT_WEIGHT, dtype=np.float64)  # Apply event weight for normalization
        if len(flat_mc_p) > 0:      h_gen_p.FillN(len(flat_mc_p), flat_mc_p, weights)
        if len(flat_rec_p) > 0:     h_rec_p.FillN(len(flat_rec_p), flat_rec_p, weights)
        if len(flat_mc_pT) > 0:     h_gen_pT.FillN(len(flat_mc_pT), flat_mc_pT, weights)
        if len(flat_rec_pT) > 0:    h_rec_pT.FillN(len(flat_rec_pT), flat_rec_pT, weights)
        if len(flat_mc_theta) > 0:  h_gen_theta.FillN(len(flat_mc_theta), flat_mc_theta, weights)
        if len(flat_rec_theta) > 0: h_rec_theta.FillN(len(flat_rec_theta), flat_rec_theta, weights)
        if len(flat_mc_Q2) > 0:     h_gen_Q2.FillN(len(flat_mc_Q2), flat_mc_Q2, weights)
        if len(flat_rec_Q2) > 0:    h_rec_Q2.FillN(len(flat_rec_Q2), flat_rec_Q2, weights)
        if len(flat_mc_x) > 0:      h_gen_x.FillN(len(flat_mc_x), flat_mc_x, weights)
        if len(flat_rec_x) > 0:     h_rec_x.FillN(len(flat_rec_x), flat_rec_x, weights)

print("\n--- Electron Acceptance Summary ---")
print(f"Total generated electrons: {generated_electron_count}") 
print(f"Total detected electrons:  {detected_electron_count}")
if generated_electron_count > 0:
    print(f"Overall Electron Acceptance: {detected_electron_count/generated_electron_count:.4f}")

# ==========================================
# Canvas 1: Kinematic Distributions
# ==========================================
c_dist = ROOT.TCanvas("c_dist", "Electron Kinematics", 1200, 1000)
main_pad = ROOT.TPad("main_pad", "main_pad", 0.0, 0.0, 1.0, 0.92)
main_pad.Draw()
main_pad.Divide(2, 2)

# Helper function for setting perfect margins on every pad
def set_pad_margins():
    ROOT.gPad.SetLeftMargin(0.18)   # Wide left margin for big Y-axis titles
    ROOT.gPad.SetBottomMargin(0.16) # Wide bottom margin for big X-axis titles
    ROOT.gPad.SetRightMargin(0.05)  
    ROOT.gPad.SetTopMargin(0.12)    # Room at the top for the main title

main_pad.cd(1); ROOT.gPad.SetLogy(); set_pad_margins()
h_gen_p.Draw("HIST"); h_rec_p.Draw("HIST SAME")

main_pad.cd(2); ROOT.gPad.SetLogy(); set_pad_margins()
h_gen_theta.Draw("HIST"); h_rec_theta.Draw("HIST SAME")

main_pad.cd(3); ROOT.gPad.SetLogy(); ROOT.gPad.SetLogx(); set_pad_margins()
h_gen_Q2.Draw("HIST"); h_rec_Q2.Draw("HIST SAME")

main_pad.cd(4); ROOT.gPad.SetLogy(); ROOT.gPad.SetLogx(); set_pad_margins()
h_gen_x.Draw("HIST"); h_rec_x.Draw("HIST SAME")

c_dist.cd(0)
leg = ROOT.TLegend(0.20, 0.93, 0.80, 0.99)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetNColumns(2)
leg.SetTextSize(0.035)
leg.SetTextFont(62)
leg.AddEntry(h_gen_p, "Generated (MC Truth)", "l")
leg.AddEntry(h_rec_p, "Reconstructed", "l")
leg.Draw()

c_dist.Print("../Plots/electron_acceptance.pdf[")
c_dist.Print("../Plots/electron_acceptance.pdf")

# ==========================================
# Canvas 2: Binwise Acceptances
# ==========================================
c_acc = ROOT.TCanvas("c_acc", "Binwise Acceptances", 1200, 1000)
c_acc.Divide(2, 2)

# Helper function to compute and format acceptance histograms
def draw_acceptance(pad_num, h_rec, h_gen, title, xtitle, is_logx=False):
    c_acc.cd(pad_num)
    if is_logx: ROOT.gPad.SetLogx()
    set_pad_margins()
    ROOT.gPad.SetGridy()
    
    h_acc = h_rec.Clone(f"{h_rec.GetName()}_acc")
    h_acc.Divide(h_acc, h_gen, 1.0, 1.0, "B")
    
    h_acc.SetTitle(title)
    h_acc.GetXaxis().SetTitle(xtitle)
    h_acc.GetYaxis().SetTitle("#frac{Reconstructed}{Generated}")
    
    # Re-apply the text sizes specifically for the cloned histogram
    h_acc.GetXaxis().SetTitleSize(0.06)
    h_acc.GetYaxis().SetTitleSize(0.06)
    h_acc.GetXaxis().SetLabelSize(0.055)
    h_acc.GetYaxis().SetLabelSize(0.055)
    h_acc.GetXaxis().SetTitleOffset(1.2)
    h_acc.GetYaxis().SetTitleOffset(1.4)
    h_acc.GetXaxis().CenterTitle(True)
    h_acc.GetYaxis().CenterTitle(True)

    h_acc.SetLineColor(ROOT.kMagenta)
    h_acc.SetMarkerStyle(20)
    h_acc.SetMarkerSize(0.8)
    h_acc.SetMinimum(0.0)
    h_acc.SetMaximum(1.1)  
    
    h_acc.Draw("PE")
    return h_acc

acc_hists = []
acc_hists.append(draw_acceptance(1, h_rec_p, h_gen_p, "Momentum Acceptance", "p_{e} [GeV]"))
acc_hists.append(draw_acceptance(2, h_rec_theta, h_gen_theta, "Polar Angle Acceptance", "#theta [deg]"))
acc_hists.append(draw_acceptance(3, h_rec_Q2, h_gen_Q2, "Q^{2} Acceptance", "Q^{2} [GeV^{2}]", is_logx=True))
acc_hists.append(draw_acceptance(4, h_rec_x, h_gen_x, "x_{B} Acceptance", "x_{B}", is_logx=True))

c_acc.Print("../Plots/electron_acceptance.pdf")
c_acc.Print("../Plots/electron_acceptance.pdf]")