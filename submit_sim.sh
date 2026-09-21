#!/usr/bin/env bash
#SBATCH --job-name=eic_sim_batch
#SBATCH --account=clas12               # Verify with: sacctmgr show user <username>
#SBATCH --partition=production      # Priority is limited to 32 concurrent/16 queued; production is for bulk
#SBATCH --time=02:00:00             # Specifying tight walltime improves queue throughput
#SBATCH --nodes=1                   # Single node job
#SBATCH --ntasks=1                  # Single task
#SBATCH --cpus-per-task=4           # 4 virtual processing cores for multithreading
#SBATCH --mem-per-cpu=2000          # Memory in MB per core (2000 MB * 4 = 8 GB total)
#SBATCH --array=0-9                 # Job array for 10 concurrent chunks
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -Eeuo pipefail

mkdir -p logs

# 1. Flexible Arguments: Make the script flexible by taking the input prefix as a command-line variable
if [ $# -eq 0 ]; then
    echo "Error: Please provide the input base prefix."
    echo "Usage: sbatch submit_sim.sh /work/eic/users/aabhishe/EIC_3He_5x41_XRot_pi_bc"
    exit 1
fi

AB_PREFIX="$1"
WORKDIR="$(dirname "$AB_PREFIX")"   # directory to bind into the container

# 2. Define chunking math based on the Slurm array ID
EVENTS_PER_JOB=2000
SKIP_EVENTS=$(( SLURM_ARRAY_TASK_ID * EVENTS_PER_JOB ))

AFTERBURNER_FILE="${AB_PREFIX}.hepmc3.tree.root"
SIM_FILE="${AB_PREFIX}_sim_chunk_${SLURM_ARRAY_TASK_ID}.root"
RECON_FILE="${AB_PREFIX}_recon_chunk_${SLURM_ARRAY_TASK_ID}.root"

echo "Processing Job ID ${SLURM_ARRAY_TASK_ID} - Skipping ${SKIP_EVENTS} events"

# 3. Environment & Container Access:
# CVMFS-provided singularity is normally already on PATH on ifarm; module load
# is only needed if it isn't. Uncomment/adjust the version if required:
# module load singularity/3.9.5

CONTAINER="/cvmfs/singularity.opensciencegrid.org/eicweb/eic_xl:nightly"

# Explicitly bind the data directory so the container can see your files
# regardless of default bind-path configuration.
singularity exec -B "${WORKDIR}:${WORKDIR}" "$CONTAINER" bash -c "
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
      -Pdd4hep:xml_files=\$DETECTOR_PATH/epic_craterlake_5x41_He3.xml \
      $SIM_FILE
"
