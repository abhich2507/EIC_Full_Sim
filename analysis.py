import uproot
import awkward as ak
import vector
import numpy as np
import ROOT

# Clean up stat boxes for the overlaid plots
ROOT.gStyle.SetOptStat(0)

file_pattern = "/work/eic/users/aabhishe/EIC_3He_5x41_XRot_pi_bc_recon_chunk_*.root:events"
output_file = "/home/aabhishe/eic/EIC_Full_Sim/Plots/Q2_x_distributions.root"
out = ROOT.TFile(output_file, "RECREATE")

events = uproot.concatenate(
    file_pattern,
    filter_name=[
        "ReconstructedChargedParticles.PDG",
        "ReconstructedChargedParticles.momentum.*",
        "ReconstructedChargedParticles.energy",
        "MCParticles.PDG",
        "MCParticles.generatorStatus",
        "MCParticles.momentum.*"
    ]
)

e_prime_pdg_arrays = events["ReconstructedChargedParticles.PDG"]
mc_pdg_arrays = events["MCParticles.PDG"]
mc_gen_status = events["MCParticles.generatorStatus"]

# Masses
e_mass = 0.000511
p_beam_mass = 2.8084  # Helium-3 mass in GeV

# Apply masks
e_prime_mask = (e_prime_pdg_arrays == 11)
e_prime_mc_mask = (mc_pdg_arrays == 11) & (mc_gen_status == 1)
e_beam_mask = (mc_pdg_arrays == 11) & (mc_gen_status == 4)
p_beam_mask = (mc_pdg_arrays == 1000020030) & (mc_gen_status == 4)

# Extract scattered (reco) electrons
e_prime_pz = events["ReconstructedChargedParticles.momentum.z"][e_prime_mask]
e_prime_px = events["ReconstructedChargedParticles.momentum.x"][e_prime_mask]
e_prime_py = events["ReconstructedChargedParticles.momentum.y"][e_prime_mask]
e_prime_E = events["ReconstructedChargedParticles.energy"][e_prime_mask]

# Extract scattered (MC truth) electrons
e_prime_mc_pz = events["MCParticles.momentum.z"][e_prime_mc_mask]
e_prime_mc_px = events["MCParticles.momentum.x"][e_prime_mc_mask]
e_prime_mc_py = events["MCParticles.momentum.y"][e_prime_mc_mask]
e_prime_mc_E = np.sqrt(e_prime_mc_px**2 + e_prime_mc_py**2 + e_prime_mc_pz**2 + e_mass**2)

# Extract beam (MC truth) electrons
e_beam_pz = events["MCParticles.momentum.z"][e_beam_mask]
e_beam_px = events["MCParticles.momentum.x"][e_beam_mask]
e_beam_py = events["MCParticles.momentum.y"][e_beam_mask]
e_beam_E = np.sqrt(e_beam_px**2 + e_beam_py**2 + e_beam_pz**2 + e_mass**2) 

# Extract beam (MC truth) ions
p_beam_pz = events["MCParticles.momentum.z"][p_beam_mask]
p_beam_px = events["MCParticles.momentum.x"][p_beam_mask]
p_beam_py = events["MCParticles.momentum.y"][p_beam_mask]
p_beam_E = np.sqrt(p_beam_px**2 + p_beam_py**2 + p_beam_pz**2 + p_beam_mass**2) 

# Zip into vectors
e_beam = vector.zip({"px": e_beam_px, "py": e_beam_py, "pz": e_beam_pz, "E": e_beam_E})
p_beam = vector.zip({"px": p_beam_px, "py": p_beam_py, "pz": p_beam_pz, "E": p_beam_E})
e_prime = vector.zip({"px": e_prime_px, "py": e_prime_py, "pz": e_prime_pz, "E": e_prime_E})
e_prime_mc = vector.zip({"px": e_prime_mc_px, "py": e_prime_mc_py, "pz": e_prime_mc_pz, "E": e_prime_mc_E})

# Sort and extract single leading particles
e_prime_sorted = e_prime[ak.argsort(e_prime.E, ascending=False)]
e_prime_mc_sorted = e_prime_mc[ak.argsort(e_prime_mc.E, ascending=False)]

e_beam_single = ak.firsts(e_beam) 
p_beam_single = ak.firsts(p_beam)
e_prime_leading = ak.firsts(e_prime_sorted)
e_prime_mc_leading = ak.firsts(e_prime_mc_sorted)

# Kinematics - Reconstructed
q = e_beam_single - e_prime_leading
Q2 = -q.mass2
x = 3.0 * Q2 / (2.0 * p_beam_single.dot(q))

# Kinematics - MC Truth
q_mc = e_beam_single - e_prime_mc_leading
Q2_mc = -q_mc.mass2
x_mc = 3.0 * Q2_mc / (2.0 * p_beam_single.dot(q_mc))

# Clean and convert for histogram filling
Q2_clean = ak.to_numpy(ak.drop_none(Q2)).astype(np.float64)
x_clean = ak.to_numpy(ak.drop_none(x)).astype(np.float64)
Q2_mc_clean = ak.to_numpy(ak.drop_none(Q2_mc)).astype(np.float64)
x_mc_clean = ak.to_numpy(ak.drop_none(x_mc)).astype(np.float64)

# Define bin edges
Q2_bin_edges = np.logspace(0, 3, 101)  
xbin_edges = np.logspace(-4, 0, 101)   

# Create Histograms
Q2_hist    = ROOT.TH1F("his1", "Q^{2} Distribution; Q^{2} [GeV^{2}]; Normalized Events", len(Q2_bin_edges) - 1, Q2_bin_edges)
Q2_mc_hist = ROOT.TH1F("his1_mc", "Q^{2} Distribution; Q^{2} [GeV^{2}]; Normalized Events", len(Q2_bin_edges) - 1, Q2_bin_edges)

x_hist    = ROOT.TH1F("his2", "Bjorken x Distribution; x_{B}; Normalized Events", len(xbin_edges) - 1, xbin_edges)
x_mc_hist = ROOT.TH1F("his2_mc", "Bjorken x Distribution; x_{B}; Normalized Events", len(xbin_edges) - 1, xbin_edges)

Q2_x_hist = ROOT.TH2F("his3", "Q^{2} vs x Distribution; x_{B}; Q^{2} [GeV^{2}]", len(xbin_edges) - 1, xbin_edges, len(Q2_bin_edges) - 1, Q2_bin_edges)

# Fill Histograms
Q2_hist.FillN(len(Q2_clean), Q2_clean, np.ones(len(Q2_clean), dtype=np.float64))
x_hist.FillN(len(x_clean), x_clean, np.ones(len(x_clean), dtype=np.float64))
Q2_x_hist.FillN(len(x_clean), x_clean, Q2_clean, np.ones(len(Q2_clean), dtype=np.float64))

Q2_mc_hist.FillN(len(Q2_mc_clean), Q2_mc_clean, np.ones(len(Q2_mc_clean), dtype=np.float64))
x_mc_hist.FillN(len(x_mc_clean), x_mc_clean, np.ones(len(x_mc_clean), dtype=np.float64))

# Setup Canvas
canvas = ROOT.TCanvas("canvas", "Q2 and x Distributions", 1200, 400)
canvas.Divide(3, 1)

# Pad 1: x_B Distribution
canvas.cd(1)
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.05)
ROOT.gPad.SetBottomMargin(0.15)

if x_mc_hist.Integral() > 0: x_mc_hist.Scale(1.0 / x_mc_hist.Integral())
x_mc_hist.SetLineColor(ROOT.kRed)
x_mc_hist.SetLineWidth(2)
x_mc_hist.Draw("HIST")

if x_hist.Integral() > 0: x_hist.Scale(1.0 / x_hist.Integral())
x_hist.SetLineColor(ROOT.kBlue)
x_hist.SetLineWidth(2)
x_hist.Draw("HIST SAME")

leg1 = ROOT.TLegend(0.40, 0.75, 0.70, 0.88)
leg1.SetBorderSize(0)
leg1.AddEntry(x_mc_hist, "MC Truth", "l")
leg1.AddEntry(x_hist, "Reconstructed", "l")
leg1.Draw()

# Pad 2: Q2 Distribution
canvas.cd(2)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.05)
ROOT.gPad.SetBottomMargin(0.15)

if Q2_mc_hist.Integral() > 0: Q2_mc_hist.Scale(1.0 / Q2_mc_hist.Integral())
Q2_mc_hist.SetLineColor(ROOT.kRed)
Q2_mc_hist.SetLineWidth(2)
Q2_mc_hist.Draw("HIST")

if Q2_hist.Integral() > 0: Q2_hist.Scale(1.0 / Q2_hist.Integral())
Q2_hist.SetLineColor(ROOT.kBlue)
Q2_hist.SetLineWidth(2)
Q2_hist.Draw("HIST SAME")

leg2 = ROOT.TLegend(0.40, 0.75, 0.70, 0.88)
leg2.SetBorderSize(0)
leg2.AddEntry(Q2_mc_hist, "MC Truth", "l")
leg2.AddEntry(Q2_hist, "Reconstructed", "l")
leg2.Draw()

# Pad 3: Q2 vs x (Reconstructed)
canvas.cd(3)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.15)
ROOT.gPad.SetBottomMargin(0.15)

if Q2_x_hist.Integral() > 0: Q2_x_hist.Scale(1.0 / Q2_x_hist.Integral())
Q2_x_hist.Draw("COLZ")

# Save outputs
canvas.SaveAs("/home/aabhishe/eic/EIC_Full_Sim/Plots/Q2_x_distributions.pdf")

out.cd()
x_hist.Write()
x_mc_hist.Write()
Q2_hist.Write()
Q2_mc_hist.Write()
Q2_x_hist.Write()
out.Write()
out.Close()