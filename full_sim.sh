#!/usr/bin/env bash
# filepath: /home/aabhishe/eic/EIC_Full_Sim/run_full_sim.sh

set -Eeuo pipefail

# 1. Parse command-line flags
RUN_AB=false
RUN_SIM=false
RUN_RECO=false

usage() {
    echo "Usage: $0 [-a] [-s] [-r] [-A]"
    echo "  -a : Run Afterburner (abconv)"
    echo "  -s : Run Simulation (ddsim)"
    echo "  -r : Run Reconstruction (eicrecon)"
    echo "  -A : Run ALL steps (default if no flags are provided)"
    exit 1
}

# If no arguments are passed, default to running everything
if [ $# -eq 0 ]; then
    RUN_AB=true
    RUN_SIM=true
    RUN_RECO=true
else
    while getopts "asrAh" opt; do
        case $opt in
            a) RUN_AB=true ;;
            s) RUN_SIM=true ;;
            r) RUN_RECO=true ;;
            A) RUN_AB=true; RUN_SIM=true; RUN_RECO=true ;;
            h|*) usage ;;
        esac
    done
fi

# 2. Setup the ePIC environment
source /opt/detector/epic-main/bin/thisepic.sh

# 3. Define exact file paths
INPUT="/work/eic/users/aabhishe/EIC_3He_5x41_XRot_pi.root"
AB_PREFIX="/work/eic/users/aabhishe/EIC_3He_5x41_XRot_pi_bc"

AFTERBURNER_FILE="${AB_PREFIX}.hepmc3.tree.root"
SIM_FILE="${AB_PREFIX}.hepmc3.tree_sim_output.root"
RECON_FILE="${AB_PREFIX}.hepmc3.tree_sim_output_recon.root"
COMPACT_FILE="$DETECTOR_PATH/epic_craterlake_5x41_He3.xml"
NEVENTS=10000

# 4. Execute Selected Steps

if [ "$RUN_AB" = true ]; then
    echo "=== Afterburner ==="
    abconv \
      -p ip6_hidiv_41x5 \
      "$INPUT" \
      -o "$AB_PREFIX"
fi

if [ "$RUN_SIM" = true ]; then
    # Verify the afterburner file exists before attempting simulation
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
fi

if [ "$RUN_RECO" = true ]; then
    # Verify the simulation file exists before attempting reconstruction
    if [[ ! -f "$SIM_FILE" ]]; then
        echo "Error: Simulation output not found: $SIM_FILE" >&2
        exit 1
    fi

    echo "=== Reconstruction ==="
    eicrecon \
      -Ppodio:output_file="$RECON_FILE" \
      -Pjana:nevents=$NEVENTS \
      -Pdd4hep:xml_files="$COMPACT_FILE" \
      "$SIM_FILE"
fi

echo
echo "=== Script finished successfully ==="