#!/usr/bin/env bash
# filepath: /home/aabhishe/eic/EIC_Full_Sim/run_full_sim.sh

set -Eeuo pipefail

# 0. Setup the ePIC environment (Required for DETECTOR_PATH and software)
source /opt/detector/epic-main/bin/thisepic.sh

# 1. Define exact file paths
INPUT="/work/eic/users/aabhishe/EIC_3He_5x41_XRot_pi.root"
AB_PREFIX="/work/eic/users/aabhishe/EIC_3He_5x41_XRot_pi_bc"

# Note: abconv automatically appends extensions. Your manual commands show it produced .hepmc3.tree.root
AFTERBURNER_FILE="${AB_PREFIX}.hepmc3.tree.root"
SIM_FILE="${AB_PREFIX}.hepmc3.tree_sim_output.root"
RECON_FILE="${AB_PREFIX}.hepmc3.tree_sim_output_recon.root"
COMPACT_FILE="$DETECTOR_PATH/epic_craterlake_5x41_He3.xml"
NEVENTS=1000

echo "=== Afterburner ==="
abconv \
  -p ip6_hidiv_41x5 \
  "$INPUT" \
  -o "$AB_PREFIX"

# Verify the afterburner actually created the expected file before proceeding
if [[ ! -f "$AFTERBURNER_FILE" ]]; then
    echo "Error: Afterburner output not found: $AFTERBURNER_FILE" >&2
    exit 1
fi

echo "=== DD4hep simulation ==="
ddsim \
  --runType batch \
  --compactFile "$COMPACT_FILE" \
  --inputFiles "$AFTERBURNER_FILE" \
  --outputFile "$SIM_FILE" \
  --numberOfEvents $NEVENTS

echo "=== Reconstruction ==="
eicrecon \
  -Ppodio:output_file="$RECON_FILE" \
  -Pjana:nevents=$NEVENTS \
  -Pdd4hep:xml_files="$COMPACT_FILE" \
  "$SIM_FILE"

echo
echo "=== Completed successfully ==="
echo "  Afterburner: $AFTERBURNER_FILE"
echo "  Simulation:  $SIM_FILE"
echo "  Reco:        $RECON_FILE"