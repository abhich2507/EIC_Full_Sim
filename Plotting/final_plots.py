import ROOT

# Enable stat boxes
ROOT.gStyle.SetOptStat(1111)

gen_output = ROOT.TFile("/home/aabhishe/He3DIS/Plots/inclusive_5x41_XROT_Pi.root", "READ")
ana_output = ROOT.TFile("/home/aabhishe/eic/EIC_Full_Sim/Plots/Q2_x_distributions.root", "READ")

# 1. Generator Data (He3DIS)
gen_Q2_hist = gen_output.Get("h_Q2_incl") 
gen_x_hist  = gen_output.Get("h_xB_incl") 

# 2. MC Truth Data (EIC Full Sim)
mc_Q2_hist = ana_output.Get("his1_mc")
mc_x_hist  = ana_output.Get("his2_mc")

# 3. Reconstructed Data (EIC Full Sim)
ana_Q2_hist = ana_output.Get("his1")
ana_x_hist  = ana_output.Get("his2")

if not gen_x_hist or not gen_Q2_hist:
    print("Error: Could not find generator histograms in the file!")
    exit(1)
if not mc_x_hist or not mc_Q2_hist:
    print("Error: Could not find MC Truth histograms in the file!")
    exit(1)
if not ana_x_hist or not ana_Q2_hist:
    print("Error: Could not find reconstructed histograms in the file!")
    exit(1)

# Normalize all histograms first
gen_Q2_hist.Scale(1.0 / gen_Q2_hist.Integral())
mc_Q2_hist.Scale(1.0 / mc_Q2_hist.Integral())
ana_Q2_hist.Scale(1.0 / ana_Q2_hist.Integral())

gen_x_hist.Scale(1.0 / gen_x_hist.Integral())
mc_x_hist.Scale(1.0 / mc_x_hist.Integral())
ana_x_hist.Scale(1.0 / ana_x_hist.Integral())

# Determine Global Y-Axis Max for a truly common Y-axis across both pads
global_max = max(
    gen_Q2_hist.GetMaximum(), mc_Q2_hist.GetMaximum(), ana_Q2_hist.GetMaximum(),
    gen_x_hist.GetMaximum(), mc_x_hist.GetMaximum(), ana_x_hist.GetMaximum()
)

y_min = 1e-6
y_max = global_max * 10.0

# Change Histogram Titles and Axis Labels
gen_Q2_hist.SetTitle("Q^{2} Distribution;Q^{2} [GeV^{2}];Normalized Events")
gen_x_hist.SetTitle("x_{B} Distribution;x_{B};") # Removed Y-title to avoid overlap

# Helper function to move and color stat boxes
def format_stat_box(hist, x1, y1, x2, y2, color):
    ROOT.gPad.Update()
    st = hist.FindObject("stats")
    if st:
        st.SetX1NDC(x1)
        st.SetX2NDC(x2)
        st.SetY1NDC(y1)
        st.SetY2NDC(y2)
        st.SetTextColor(color)
        st.SetLineColor(color)

canvas = ROOT.TCanvas("canvas", "Q2 and x Distributions", 1200, 600)

# --- Define custom attached pads ---
# Pad 1 takes up 45% of width. Pad 2 takes up 55%.
# Margins are mathematically balanced so the plotting areas are exactly the same width.
pad1 = ROOT.TPad("pad1", "", 0.0, 0.0, 0.45, 1.0)
pad1.SetLeftMargin(0.333)
pad1.SetRightMargin(0.0)  # Flush to the right edge
pad1.SetBottomMargin(0.15)
pad1.SetTopMargin(0.10)
pad1.Draw()

pad2 = ROOT.TPad("pad2", "", 0.45, 0.0, 1.0, 1.0)
pad2.SetLeftMargin(0.0)   # Flush to the left edge
pad2.SetRightMargin(0.454) # Large right margin for legend/stats
pad2.SetBottomMargin(0.15)
pad2.SetTopMargin(0.10)
pad2.Draw()

# --- First Pad (Left: Q2) ---
pad1.cd()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetTickx(1)
ROOT.gPad.SetTicky(1)

# Gen
gen_Q2_hist.SetMaximum(y_max)
gen_Q2_hist.SetMinimum(y_min)
gen_Q2_hist.SetLineColor(ROOT.kRed)
gen_Q2_hist.SetMarkerColor(ROOT.kRed)
gen_Q2_hist.SetLineWidth(2)
gen_Q2_hist.SetMarkerStyle(20)
gen_Q2_hist.Draw("P")
# Stat boxes inside the top right of the Q2 plot area
format_stat_box(gen_Q2_hist, 0.45, 0.74, 0.95, 0.88, ROOT.kRed)

# MC Truth 
mc_Q2_hist.SetLineColor(ROOT.kGreen+2)
mc_Q2_hist.SetMarkerColor(ROOT.kGreen+2)
mc_Q2_hist.SetLineWidth(2)
mc_Q2_hist.SetMarkerStyle(21)
mc_Q2_hist.Draw("P SAMES")
format_stat_box(mc_Q2_hist, 0.45, 0.60, 0.95, 0.74, ROOT.kGreen+2)

# Reco
ana_Q2_hist.SetLineColor(ROOT.kBlue)
ana_Q2_hist.SetMarkerColor(ROOT.kBlue)
ana_Q2_hist.SetLineWidth(2)
ana_Q2_hist.SetMarkerStyle(22)
ana_Q2_hist.Draw("P SAMES")
format_stat_box(ana_Q2_hist, 0.45, 0.46, 0.95, 0.60, ROOT.kBlue)

# --- Second Pad (Right: x-Bjorken) ---
pad2.cd()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetTickx(1)
ROOT.gPad.SetTicky(1)

# Suppress Y-axis labels on Pad 2 so it behaves as a common axis
gen_x_hist.GetYaxis().SetLabelSize(0)
gen_x_hist.GetYaxis().SetTitleSize(0)

# Gen
gen_x_hist.SetMaximum(y_max)
gen_x_hist.SetMinimum(y_min)
gen_x_hist.SetLineColor(ROOT.kRed)
gen_x_hist.SetMarkerColor(ROOT.kRed)
gen_x_hist.SetLineWidth(2)
gen_x_hist.SetMarkerStyle(20)
gen_x_hist.Draw("P")
# Stat boxes pushed safely into the right-hand margin
format_stat_box(gen_x_hist, 0.56, 0.59, 0.98, 0.73, ROOT.kRed)

# MC Truth
mc_x_hist.SetLineColor(ROOT.kGreen+2)
mc_x_hist.SetMarkerColor(ROOT.kGreen+2)
mc_x_hist.SetLineWidth(2)
mc_x_hist.SetMarkerStyle(21)
mc_x_hist.Draw("P SAMES")
format_stat_box(mc_x_hist, 0.56, 0.45, 0.98, 0.59, ROOT.kGreen+2)

# Reco
ana_x_hist.SetLineColor(ROOT.kBlue)
ana_x_hist.SetMarkerColor(ROOT.kBlue)
ana_x_hist.SetLineWidth(2)
ana_x_hist.SetMarkerStyle(22)
ana_x_hist.Draw("P SAMES")
format_stat_box(ana_x_hist, 0.56, 0.31, 0.98, 0.45, ROOT.kBlue)

# Only One Legend drawn in Pad 2's right margin
leg = ROOT.TLegend(0.56, 0.75, 0.98, 0.88)
leg.SetBorderSize(0)
leg.SetTextSize(0.04)
leg.AddEntry(gen_x_hist, "Generated (He3DIS)", "p")
leg.AddEntry(mc_x_hist, "MC Truth (Full Sim)", "p")
leg.AddEntry(ana_x_hist, "Reconstructed", "p")
leg.Draw()

canvas.SaveAs("/home/aabhishe/eic/EIC_Full_Sim/Plots/Q2_x_distributions_comparison.pdf")