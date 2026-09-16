source /opt/detector/epic-main/bin/thisepic.sh

eicrecon -Ppodio:output_file=/work/eic/users/aabhishe/EIC_3He_5x41_bc.hepmc3.tree_sim_output_recon_test.root \
         -Pjana:nevents=100 \
         -Pdd4hep:xml_files=$DETECTOR_PATH/epic_craterlake_5x41_He3.xml \
         /work/eic/users/aabhishe/EIC_3He_5x41_bc.hepmc3.tree_sim_output.root