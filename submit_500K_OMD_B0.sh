#!/usr/bin/env bash
#SBATCH --job-name=eic_sim_500k_omd_b0
#SBATCH --account=clas12               # Verify with: sacctmgr show user <username>
#SBATCH --partition=production         # Priority is limited to 32 concurrent/16 queued; production is for bulk
#SBATCH --time=02:30:00                # Updated: Increased to 2.5 hours for 2k events
#SBATCH --nodes=1                      # Single node job
#SBATCH --ntasks=1                     # Single task
#SBATCH --cpus-per-task=4              # 4 virtual processing cores for multithreading
#SBATCH --mem-per-cpu=2000             # Memory in MB per core (2000 MB * 4 = 8 GB total)
#SBATCH --array=0-499%100              # Updated: 500 jobs total, throttled to 100 running concurrently
#SBATCH --output=/farm_out/%u/%x_%A_%a.out
#SBATCH --error=/farm_out/%u/%x_%A_%a.err

set -Eeuo pipefail

# Base work directory (no trailing slash)
WORKDIR="/work/eic/users/aabhishe"


mkdir -p "${WORKDIR}/Sim_out_1M_omd_b0"
mkdir -p "${WORKDIR}/Recon_out_1M_omd_b0"

# 1. Chunking math using Slurm Array ID
TASK_ID="${SLURM_ARRAY_TASK_ID}"
EVENTS_PER_JOB=2000
SKIP_EVENTS=$(( TASK_ID * EVENTS_PER_JOB ))

AFTERBURNER_FILE="${WORKDIR}/EIC_3He_5x41_XRot_pi_bc.hepmc3.tree.root"
SIM_FILE="${WORKDIR}/Sim_out_1M_omd_b0/sim_chunk_${TASK_ID}.root"
RECON_FILE="${WORKDIR}/Recon_out_1M_omd_b0/recon_chunk_${TASK_ID}.root"

echo "Processing Chunk ${TASK_ID} - Skipping ${SKIP_EVENTS} events (Running ${EVENTS_PER_JOB} events)"

# 2. Create a unique temporary directory inside your work folder
JOB_TMPDIR="${WORKDIR}/tmp_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
mkdir -p "$JOB_TMPDIR"

# Automatically remove temp directory and intermediate sim file on exit or crash
trap 'rm -rf "$JOB_TMPDIR" "$SIM_FILE"' EXIT


export SINGULARITY_TMPDIR="$JOB_TMPDIR"
export SINGULARITY_CACHEDIR="$JOB_TMPDIR"
export APPTAINER_TMPDIR="$JOB_TMPDIR"
export APPTAINER_CACHEDIR="$JOB_TMPDIR"

CONTAINER="/cvmfs/singularity.opensciencegrid.org/eicweb/eic_xl:nightly"


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
      -Ppodio:output_collections=\"MCParticles,ReconstructedChargedParticles,ForwardOffMTrackerHits,ForwardOffMTrackerRecHits,ForwardOffMTrackerRecParticles,B0TrackerHits,B0TrackerRecHits,B0TrackerRecParticles\" \
      -Pdd4hep:xml_files=\$DETECTOR_PATH/epic_craterlake_5x41_He3.xml \
      $SIM_FILE

    echo 'Removing intermediate simulation file...'
    rm -f $SIM_FILE
"

echo "Chunk ${TASK_ID} completed successfully."