#!/bin/bash

scripts_dir="/home/sibwang/contrastive-unpaired-translation/scripts_on_cluster/20250711_parameter_sweep/batch_scripts"

files=($(ls $scripts_dir/*.run | sort))
file_count=${#files[@]}

echo -n "Are you sure you want to submit $file_count jobs? (y/Y to confirm) "
read -r confirmation
if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
    echo "Submission canceled."
    exit 1
fi

for file in "${files[@]}"; do
    echo "Submitting $file"
    sbatch $file
done

echo "Submitted $file_count files to the scheduler"
