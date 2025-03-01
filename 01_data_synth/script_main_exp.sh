#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="syno_hybrid"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256g
#SBATCH --time=5-0:00:00


conda activate /rhome/msaee007/bigdata/conda_packages
export PATH=/rhome/msaee007/bigdata/conda_packages/bin/:$PATH

# python resnet_baseline.exp
# python pointnet_main_exp.py
# label: all_values_parametrized_real_data_rotated (this is the synth data based one: no rotation and not augmentation)
# all_values_parametrized_weather (this is based on weather data)
python p1_pred_summary.py
# python scalability_test.py
# python pointnet_hybrid_exp.py
