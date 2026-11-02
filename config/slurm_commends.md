### Interactive Job ###

sinteractive  --time 0-8:00:00  --gpu rtx_6000:1
sinteractive --time 0-8:00:00 --cpu 4
sinteractive --time 0-8:00:00 --mem 58


### Sync Notebooks

After editing a `.py` file, sync the notebook from the repository root:

```bash
cd Data-Mining-Assignments
conda deactivate
conda activate ai_map_env
conda run -n ai_map_env jupytext --sync Project/src/1_load_gdelt_data.py
conda run -n ai_map_env jupytext --sync Project/src/2_ai_anxiety_map.py

```


### 1_load_gdelt_data

#### Run Single Jobs

```bash
cd "/home/${USER}/Data-Mining-Assignments/Project"

# GDELT parameters
export GDELT_DATE_START="2024-01-01"
export GDELT_DATE_END="2025-01-02"
export GDELT_AGGREGATION_PERIOD="7D"
export GDELT_SAVE_INTERVAL="1W" # M is no longer supported

# Submit the job
sbatch --export=ALL config/1_run_load_gdelt_data.sbatch
```


#### Run Multiple Sbatch Jobs
```bash
cd "/home/${USER}/Data-Mining-Assignments/Project"

export GDELT_AGGREGATION_PERIOD="7D"
export GDELT_SAVE_INTERVAL="1W"

for year in 2021 2022 2023 2024; do
  for start in 01 03 05 07 09 11; do
    export GDELT_DATE_START="${year}-${start}-01"

    # Keep the inclusive boundary day to match existing aggregate filenames.
    end=$(date -d "${GDELT_DATE_START} +2 months" +%F)
    export GDELT_DATE_END="$end"

    echo "Submitting $GDELT_DATE_START -> $GDELT_DATE_END"
    sbatch --export=ALL config/1_run_load_gdelt_data.sbatch
  done
done
```



### 2_ai_anxiety_map
```bash

cd "/home/${USER}/Data-Mining-Assignments/Project"
sbatch config/2_ai_anxiety_map.sbatch

```
