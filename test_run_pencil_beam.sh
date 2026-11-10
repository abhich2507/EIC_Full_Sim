source /opt/detector/epic-main/bin/thisepic.sh
npsim \
  --compactFile $DETECTOR_PATH/epic_craterlake_5x41_He3.xml \
  --numberOfEvents 10000 \
  --enableGun \
  --gun.particle "proton" \
  --gun.position "(0.0*mm, 0.0*mm, 0.0*mm)" \
  --gun.direction "(-0.024997396, 0.0, 0.999687516)" \
  --gun.momentumMin "41*GeV" \
  --gun.momentumMax "41*GeV" \
  --outputFile /work/eic/users/aabhishe/Test_run_pencil/test_proton_gun.edm4hep.root