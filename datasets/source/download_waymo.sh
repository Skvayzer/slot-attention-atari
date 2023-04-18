#!/bin/bash -l
#SBATCH --job-name=download_waymo
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
##SBATCH --time=0-0:05:00
#SBATCH --partition=titan_X
##SBATCH --gres=gpu:1
##SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --mail-user=k.smirnov@innopolis.university
#SBATCH --mail-type=END
#SBATCH --no-kill
#SBATCH --comment="Обучение нейросетевой модели в рамках НИР Центра когнитивного моделирования"

singularity instance start \
                     --nv  \
                     -f \
                     --bind /home/AI/yudin.da/smirnov_cv/sa_atari:/home/sa_atari \
                     ml_env.sif ml_env

singularity exec instance://ml_env /bin/bash -c -f "
      source /miniconda/etc/profile.d/conda.sh;
      conda activate ml_env;
      pip install openexr==1.3.9;
      set -x;
      nvidia-smi;
      free -m;
      cd /home/sa_atari/datasets;
      gsutil -m cp -r "gs://waymo_open_dataset_v_2_0_0/training" .;
      free -m;
";

singularity instance stop ml_env