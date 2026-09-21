import ROOT

# Enable stat boxes (1111 shows Name, Entries, Mean, RMS)
ROOT.gStyle.SetOptStat(1111)

gen_output = ROOT.TFile("/home/aabhishe/He3DIS/Plots/ed_5x41_XROT_Pi.root", "READ")
ana_output = ROOT.TFile("/home/aabhishe/eic/EIC_Full_Sim/Plots/Q2_x_distributions.root", "READ")

gen_Q2_hist = gen_output.Get("h_Q2_ed") 
gen_x_hist  = gen_output.Get("h_xB_ed") 

ana_Q2_hist = ana_output.Get("his1")
ana_x_hist  = ana_output.Get("his2")

if not gen_x_hist or not gen_Q2_hist:
    print("Error: Could not find generator histograms in the file!")
    exit(1)
if not ana_x_hist or not ana_Q2_hist:
    print("Error: Could not find analysis histograms in the file!")
    exit(1)

# Change Histogram Titles and Axis Labels
gen_Q2_hist.SetTitle("Q^{2} Distribution;Q^{2} [GeV^{2}];Normalized Events")
gen_x_hist.SetTitle("x_{B} Distribution;x_{B};Normalized Events")

canvas = ROOT.TCanvas("canvas", "Q2 and x Distributions", 800, 600)
canvas.Divide(2, 1)

# --- First Pad (Left: Q2) ---
canvas.cd(1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()

ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.05)
ROOT.gPad.SetBottomMargin(0.15)
ROOT.gPad.SetTopMargin(0.10)

gen_Q2_hist.Scale(1.0 / gen_Q2_hist.Integral())
gen_Q2_hist.SetLineColor(ROOT.kRed)
gen_Q2_hist.SetMarkerColor(ROOT.kRed)
gen_Q2_hist.SetLineWidth(2)
gen_Q2_hist.SetMarkerStyle(20)
gen_Q2_hist.Draw("P")

ana_Q2_hist.Scale(1.0 / ana_Q2_hist.Integral())
ana_Q2_hist.SetLineColor(ROOT.kBlue)
ana_Q2_hist.SetMarkerColor(ROOT.kBlue)
ana_Q2_hist.SetLineWidth(2)
ana_Q2_hist.SetMarkerStyle(20)
ana_Q2_hist.Draw("P SAME")

leg1 = ROOT.TLegend(0.35, 0.35, 0.65, 0.50)
leg1.SetBorderSize(1)
leg1.AddEntry(gen_Q2_hist, "Generated", "p")
leg1.AddEntry(ana_Q2_hist, "Reconstructed", "p")
leg1.Draw()

# Force the pad to update so the stat box is drawn
ROOT.gPad.Update()

# --- Second Pad (Right: x-Bjorken) ---
canvas.cd(2)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()

ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.05)
ROOT.gPad.SetBottomMargin(0.15)
ROOT.gPad.SetTopMargin(0.10)

gen_x_hist.Scale(1.0 / gen_x_hist.Integral())
gen_x_hist.SetLineColor(ROOT.kRed)
gen_x_hist.SetMarkerColor(ROOT.kRed)
gen_x_hist.SetLineWidth(2)
gen_x_hist.SetMarkerStyle(20)
gen_x_hist.Draw("P")

ana_x_hist.Scale(1.0 / ana_x_hist.Integral())
ana_x_hist.SetLineColor(ROOT.kBlue)
ana_x_hist.SetMarkerColor(ROOT.kBlue)
ana_x_hist.SetLineWidth(2)
ana_x_hist.SetMarkerStyle(20)
ana_x_hist.Draw("P SAME")

leg2 = ROOT.TLegend(0.35, 0.35, 0.65, 0.50)
leg2.SetBorderSize(1)
leg2.AddEntry(gen_x_hist, "Generated", "p")
leg2.AddEntry(ana_x_hist, "Reconstructed", "p")
leg2.Draw()

# Force the pad to update so the stat box is drawn
ROOT.gPad.Update()

canvas.SaveAs("/home/aabhishe/eic/EIC_Full_Sim/Plots/Q2_x_distributions_comparison.pdf")