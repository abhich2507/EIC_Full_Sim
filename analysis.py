import uproot
import awkward as ak
import vector
import numpy as np
import ROOT

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

e_beam_energy = 5.0
p_beam_energy = 41.0*3.0 # total energy here 
A = 3.0  # Mass number for Helium-3

# Apply masks
e_prime_mask = (e_prime_pdg_arrays == 11)
e_beam_mask = (mc_pdg_arrays == 11) & (mc_gen_status == 4)
p_beam_mask = (mc_pdg_arrays == 1000020030) & (mc_gen_status == 4)

# Extract scattered (reco) electrons
e_prime_pz = events["ReconstructedChargedParticles.momentum.z"][e_prime_mask]
e_prime_px = events["ReconstructedChargedParticles.momentum.x"][e_prime_mask]
e_prime_py = events["ReconstructedChargedParticles.momentum.y"][e_prime_mask]
e_prime_E = events["ReconstructedChargedParticles.energy"][e_prime_mask]

# Extract beam (MC truth) electrons
e_beam_pz = events["MCParticles.momentum.z"][e_beam_mask]
e_beam_px = events["MCParticles.momentum.x"][e_beam_mask]
e_beam_py = events["MCParticles.momentum.y"][e_beam_mask]
#e_beam_E = ak.full_like(e_beam_pz, e_beam_energy)
e_beam_E = np.sqrt(e_beam_px**2 + e_beam_py**2 + e_beam_pz**2 + (0.000511)**2)  # Electron mass in GeV

# Extract beam (MC truth) protons/ions
p_beam_pz = events["MCParticles.momentum.z"][p_beam_mask]
p_beam_px = events["MCParticles.momentum.x"][p_beam_mask]
p_beam_py = events["MCParticles.momentum.y"][p_beam_mask]
p_beam_E = np.sqrt(p_beam_px**2 + p_beam_py**2 + p_beam_pz**2 + (2.8084)**2)  # Proton mass in GeV

# Zip into vectors
e_beam = vector.zip({"px": e_beam_px, "py": e_beam_py, "pz": e_beam_pz, "E": e_beam_E})
e_prime = vector.zip({"px": e_prime_px, "py": e_prime_py, "pz": e_prime_pz, "E": e_prime_E})
p_beam = vector.zip({"px": p_beam_px, "py": p_beam_py, "pz": p_beam_pz, "E": p_beam_E})
p_beam= p_beam/A # using per nucleon momentum for Helium-3

# Sort and extract single leading particles
sorted_indices = ak.argsort(e_prime.E, ascending=False)
e_prime_sorted = e_prime[sorted_indices]

e_beam_single = ak.firsts(e_beam) # change from lists of sublists to lists of single values
p_beam_single = ak.firsts(p_beam)
e_prime_leading = ak.firsts(e_prime_sorted)

#  kinematics
q = e_beam_single - e_prime_leading
Q2 = -q.mass2

# Fixed: calculating dot product using p_beam_single instead of the jagged p_beam
x = Q2 / (2 * p_beam_single.dot(q))

print(f"Q2: {Q2[:10]}")
print(f"x: {x[:10]}")

# Clean and convert for histogram filling
Q2_clean = ak.to_numpy(ak.drop_none(Q2)).astype(np.float64)
x_clean = ak.to_numpy(ak.drop_none(x)).astype(np.float64)

# Plots starts here
Q2_bin_edges = np.logspace(0, 3, 101)  # 100 bins from 10^-2 to 10^2

Q2_hist = ROOT.TH1F("his1", "Q2 Distribution; Q2 [GeV^2]; Events", len(Q2_bin_edges) - 1, Q2_bin_edges)
xbin_edges = np.logspace(-4, 0, 101)  # 100 bins from 10^-4 to 10^-1

x_hist = ROOT.TH1F("his2", "Bjorken x Distribution; x; Events", len(xbin_edges) - 1, xbin_edges)

Q2_x_hist = ROOT.TH2F("his3", "Q2 vs x Distribution; x; Q2 [GeV^2]", len(xbin_edges) - 1, xbin_edges, len(Q2_bin_edges) - 1, Q2_bin_edges)

Q2_hist.FillN(len(Q2_clean), Q2_clean, np.ones(len(Q2_clean), dtype=np.float64))
x_hist.FillN(len(x_clean), x_clean, np.ones(len(x_clean), dtype=np.float64))
Q2_x_hist.FillN(len(x_clean), x_clean, Q2_clean, np.ones(len(Q2_clean), dtype=np.float64))
canvas= ROOT.TCanvas("canvas", "Q2 and x Distributions", 800, 600)
canvas.Divide(2, 2)
canvas.cd(1)
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogx()
x_hist.Scale(1.0 / x_hist.Integral())  
x_hist.Draw()
canvas.cd(2)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
Q2_hist.Scale(1.0 / Q2_hist.Integral())
Q2_hist.Draw()
canvas.cd(3)
Q2_x_hist.Scale(1.0 / Q2_x_hist.Integral())
Q2_x_hist.Draw("COLZ")

canvas.SaveAs("/home/aabhishe/eic/EIC_Full_Sim/Plots/Q2_x_distributions.pdf")

out.cd()
x_hist.Write()
Q2_hist.Write()
Q2_x_hist.Write()
out.Write()
out.Close()
