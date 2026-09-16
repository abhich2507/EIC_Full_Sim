source /opt/detector/epic-main/bin/thisepic.sh
ddsim --runType batch \
      --compactFile $DETECTOR_PATH/epic_craterlake_5x41_He3.xml \
      --inputFiles /work/eic/users/aabhishe/EIC_3He_5x41_bc.hepmc3.tree.root \
      --outputFile /work/eic/users/aabhishe/EIC_3He_5x41_bc.hepmc3.tree_sim_output.root \
--numberOfEvents 1000 \