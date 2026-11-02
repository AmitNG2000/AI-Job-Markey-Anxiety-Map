#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			                ### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 7-00:00:00			                ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name 1_load_gdelt_data			        ### name of the job
#SBATCH --output=./logs/%j_%x.out                   ### %j – Job ID, %x – Job name
#SBATCH --error=./logs/%j_%x.err
#SBATCH --cpus-per-task=10
##SBATCH --gpus=rtx_4090:1		                    ### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1

### Note: Lines are with ## are commented out
#SBATCH --mail-user=amitner@post.bgu.ac.il	        ### user's email for sending job status messages
#SBATCH --mail-type=ALL			                    ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --mem=60G				                    ### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################



################################### run from the IN_NOTEBOOK folder ###################################


### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo -e "Job name:\t\t$SLURM_JOB_NAME"

# Print environment variables
for var in \
    GDELT_DATE_START \
    GDELT_DATE_END \
    GDELT_AGGREGATION_PERIOD \
    GDELT_SAVE_INTERVAL
do
    if [[ -n "${!var:-}" ]]; then
        echo "$var=${!var}"
    else
        echo "$var is not set"
    fi
done

### Start your code below ####
module load anaconda				### load anaconda module
source activate ai_map_env

# papermill
IN_NOTEBOOK="./src/${SLURM_JOB_NAME}.ipynb"                               # run from Project folder
OUT_NOTEBOOK="./logs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.ipynb"

echo
echo "Input notebook:  $IN_NOTEBOOK"
echo "Output notebook: $OUT_NOTEBOOK"

cd "/home/amitner/Data-Mining-Assignments/Project"
conda run -n ai_map_env papermill "${IN_NOTEBOOK}" "${OUT_NOTEBOOK}"
