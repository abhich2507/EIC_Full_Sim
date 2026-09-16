#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TVector3.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TMath.h>
#include <iostream>

void AB_angle(const char* filename= "/work/eic/users/aabhishe/EIC_3He_5x41_XRot_pi.root") {
    // Open the file and connect to the tree
    TFile *file = TFile::Open(filename);
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    TTreeReader reader("hepmc3_tree", file);

    // Link the necessary branches
    TTreeReaderArray<int> pid(reader, "particles.pid");
    TTreeReaderArray<int> status(reader, "particles.status");
    TTreeReaderArray<double> px(reader, "particles.momentum.m_v1");
    TTreeReaderArray<double> py(reader, "particles.momentum.m_v2");
    TTreeReaderArray<double> pz(reader, "particles.momentum.m_v3");

    // Create a histogram centered around the expected 25 mrad crossing angle
    TH1D *h_cross = new TH1D("h_cross", "EIC Beam Crossing Angle;Crossing Angle [mrad];Events", 200, -1., +1.0);

    // Loop over all events
    while (reader.Next()) {
        TVector3 p_e, p_h;
        bool found_e = false;
        bool found_h = false;

        // Loop over particles in the current event
        for (int i = 0; i < pid.GetSize(); ++i) {
            if (status[i] == 4) { // Status 4 = Incoming beam particle
                if (pid[i] == 11) {
                    p_e.SetXYZ(px[i], py[i], pz[i]);
                    found_e = true;
                } 
                // Catch any ion (like He-3 1000020030) or proton
                else if (pid[i] > 2000) { 
                    p_h.SetXYZ(px[i], py[i], pz[i]);
                    found_h = true;
                }
            }
        }

        // If we found both beams, calculate the angle
        if (found_e && found_h) {
            // TVector3::Angle() returns the total 3D angle between 0 and pi radians.
            double angle_rad = p_e.Angle(p_h);
            
            // Since beams are almost anti-parallel, the crossing angle is pi - angle
            double crossing_angle_mrad = (TMath::Pi() - angle_rad) * 1000.0;
            
            h_cross->Fill(crossing_angle_mrad);
        }
    }

    // Draw the result
    TCanvas *c1 = new TCanvas("c1", "Angle", 800, 600);
    h_cross->Draw();
    c1->SaveAs("/home/aabhishe/eic/EIC_Full_Sim/debug/Raw_angle.png");
    // Optional: fit with a Gaussian to see the beam divergence smearing
    // h_cross->Fit("gaus");
}