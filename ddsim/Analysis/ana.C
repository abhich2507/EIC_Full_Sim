void ana(){

    auto file = TFile::Open("/work/eic/users/aabhishe/EIC_3He_5x41_bc.hepmc3.tree_sim_output_recon.root","READ");
    auto tree = (TTree*)file->Get("events");
    
    // Argument 1: What you want to look at (e.g., energy)
    // Argument 2: The filter condition (only show protons)
    //tree->Scan("ReconstructedChargedParticles.energy", "ReconstructedChargedParticles.PDG == 11");
    // Show the charge, mass, and energy for all positive tracks
    //tree->Scan("ReconstructedChargedParticles.charge:ReconstructedChargedParticles.mass:ReconstructedChargedParticles.energy", "ReconstructedChargedParticles.charge > 0");
    // Scan the true generated protons (PDG = 2212)
    tree->Scan("MCParticles.momentum.z:MCParticles.momentum.x:MCParticles.momentum.y", "MCParticles.PDG == 2212");
    //tree->Scan("ForwardRomanPotRecParticles.momentum.z:ForwardRomanPotRecParticles.momentum.x", "Length$(ForwardRomanPotRecParticles.momentum.z) > 0");
    //tree->Scan("ForwardOffMRecParticles.momentum.z:ForwardOffMRecParticles.momentum.x", "Length$(ForwardOffMRecParticles.momentum.z) > 0");
    // 1. Did they hit the Off-Momentum Detector?
    //tree->Scan("ForwardOffMTrackerHits.position.z:ForwardOffMTrackerHits.position.x", "Length$(ForwardOffMTrackerHits.position.z) > 0");
    // 2. Did they bend so much they hit the B0 Tracker instead?
     //tree->Scan("B0TrackerHits.position.z:B0TrackerHits.position.x", "Length$(B0TrackerHits.position.z) > 0");
     // Check if the electron is going the wrong way (+Z)
    //tree->Scan("MCParticles.momentum.z:MCParticles.momentum.x", "MCParticles.PDG == 11");
    }   