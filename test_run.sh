#!/usr/bin/env bash
#SBATCH --job-name=eic_test_162
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --time=00:20:00                # Shortened walltime so 1-event test queues faster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail

# Base work directory (no trailing slash, no dirname needed)
WORKDIR="/work/eic/users/aabhishe"

mkdir -p logs
mkdir -p "${WORKDIR}/Sim_out_1M"
mkdir -p "${WORKDIR}/Recon_out_1M"

# Emulate task 162 for 1 event
TASK_ID=162
EVENTS_PER_JOB=1
SKIP_EVENTS=$(( TASK_ID * EVENTS_PER_JOB ))

AFTERBURNER_FILE="${WORKDIR}/EIC_3He_5x41_XRot_pi_bc.hepmc3.tree.root"
SIM_FILE="${WORKDIR}/Sim_out_1M/sim_chunk_${TASK_ID}.root"
RECON_FILE="${WORKDIR}/Recon_out_1M/recon_chunk_${TASK_ID}.root"

echo "Testing Chunk ${TASK_ID} - Skipping ${SKIP_EVENTS} events (Running ${EVENTS_PER_JOB} event)"

# 1. Create a unique temporary directory inside your work folder
JOB_TMPDIR="${WORKDIR}/tmp_test_${SLURM_JOB_ID}_${TASK_ID}"
mkdir -p "$JOB_TMPDIR"

# Automatically remove temp directory and intermediate sim file on exit or crash
trap 'rm -rf "$JOB_TMPDIR" "$SIM_FILE"' EXIT

# 2. Tell Singularity/Apptainer to use this new directory
export SINGULARITY_TMPDIR="$JOB_TMPDIR"
export SINGULARITY_CACHEDIR="$JOB_TMPDIR"
export APPTAINER_TMPDIR="$JOB_TMPDIR"
export APPTAINER_CACHEDIR="$JOB_TMPDIR"

CONTAINER="/cvmfs/singularity.opensciencegrid.org/eicweb/eic_xl:nightly"

# 3. Bind the custom tmp dir to /tmp inside the container
singularity exec -B "${WORKDIR}:${WORKDIR}" -B "${JOB_TMPDIR}:/tmp" "$CONTAINER" bash -c "
    source /opt/detector/epic-main/bin/thisepic.sh

    echo 'Running ddsim...'
    ddsim \
      --runType batch \
      --compactFile \$DETECTOR_PATH/epic_craterlake_5x41_He3.xml \
      --inputFiles $AFTERBURNER_FILE \
      --outputFile $SIM_FILE \
      --skipNEvents $SKIP_EVENTS \
      --numberOfEvents $EVENTS_PER_JOB

    echo 'Running eicrecon...'
    eicrecon \
      -Pnthreads=4 \
      -Ppodio:output_file=$RECON_FILE \
      -Ppodio:output_collections=\"MCParticles,ReconstructedChargedParticles,ForwardRomanPotHits,ForwardRomanPotRecHits,ForwardRomanPotRecParticles\" \
      -Pdd4hep:xml_files=\$DETECTOR_PATH/epic_craterlake_5x41_He3.xml \
      $SIM_FILE

    echo 'Removing intermediate simulation file...'
    rm -f $SIM_FILE
"

echo "Test 162 completed successfully."