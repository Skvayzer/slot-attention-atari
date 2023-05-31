#!/bin/bash -l
#SBATCH --job-name=sa_od_waymo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
##SBATCH --time=0-0:05:00
#SBATCH --partition=DGX-1v100
##SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --mail-user=k.smirnov@innopolis.university
#SBATCH --mail-type=END
#SBATCH --no-kill
#SBATCH --comment="Обучение нейросетевой модели в рамках НИР Центра когнитивного моделирования"

singularity instance start \
                     --nv  \
                     --bind /home/AI/yudin.da/smirnov_cv/sa_atari:/home/sa_atari \
                     ml_env_isa.sif ml_env_isa

singularity exec instance://ml_env_isa /bin/bash -c "
      source /miniconda/etc/profile.d/conda.sh;
      conda activate ml_env;
      export WANDB_API_KEY=c84312b58e94070d15277f8a5d58bb72e57be7fd;
      set -x;
      lscpu;
      ulimit -Hn;
      ulimit -Sn;
      nvidia-smi;
      free -m;
      cd /home/sa_atari;
      python3 -u slot_attention_atari/training_od.py --dataset 'waymo' --train_path "/mnt/data/users_data/smirnov/sa_atari/datasets/waymo/training/camera_image" --val_path "/mnt/data/users_data/smirnov/sa_atari/datasets/waymo/testing/camera_image" --task 'isa-t waymo 31.06' --beta 0 --device 'gpu' --batch_size 64 --max_steps 5 --seed 17 --num_workers 4;
      free -m;
";

singularity instance stop ml_env_isa
