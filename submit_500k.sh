#!/usr/bin/env bash
#SBATCH --job-name=eic_sim_500k
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --time=02:30:00                # 2.5 hours for 2,000 events
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --array=250-999%100              # 250 jobs * 2,000 events = 500,000 events (max 100 at once)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -Eeuo pipefail

# Base work directory (no trailing slash)
WORKDIR="/work/eic/users/aabhishe"

mkdir -p logs
mkdir -p "${WORKDIR}/Sim_out_1M"
mkdir -p "${WORKDIR}/Recon_out_1M"

# 1. Chunking math using Slurm Array ID
TASK_ID="${SLURM_ARRAY_TASK_ID}"
EVENTS_PER_JOB=2000
SKIP_EVENTS=$(( TASK_ID * EVENTS_PER_JOB ))

AFTERBURNER_FILE="${WORKDIR}/EIC_3He_5x41_XRot_pi_bc.hepmc3.tree.root"
SIM_FILE="${WORKDIR}/Sim_out_1M/sim_chunk_${TASK_ID}.root"
RECON_FILE="${WORKDIR}/Recon_out_1M/recon_chunk_${TASK_ID}.root"

echo "Processing Chunk ${TASK_ID} - Skipping ${SKIP_EVENTS} events (Running ${EVENTS_PER_JOB} events)"

# 2. Create a unique temporary directory inside your work folder
JOB_TMPDIR="${WORKDIR}/tmp_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
mkdir -p "$JOB_TMPDIR"

# Automatically remove temp directory and intermediate sim file on exit or crash
trap 'rm -rf "$JOB_TMPDIR" "$SIM_FILE"' EXIT

# 3. Tell Singularity/Apptainer to use this new directory
export SINGULARITY_TMPDIR="$JOB_TMPDIR"
export SINGULARITY_CACHEDIR="$JOB_TMPDIR"
export APPTAINER_TMPDIR="$JOB_TMPDIR"
export APPTAINER_CACHEDIR="$JOB_TMPDIR"

CONTAINER="/cvmfs/singularity.opensciencegrid.org/eicweb/eic_xl:nightly"

# 4. Bind the custom tmp dir to /tmp inside the container and run
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

echo "Chunk ${TASK_ID} completed successfully."