#!/bin/bash -l
#SBATCH --job-name=sa_od_tetrominoes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
##SBATCH --time=0-0:05:00
#SBATCH --partition=titan_X
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
                     ml_env.sif ml_env

singularity exec instance://ml_env /bin/bash -c "
      source /miniconda/etc/profile.d/conda.sh;
      conda activate ml_env;
      export WANDB_API_KEY=c84312b58e94070d15277f8a5d58bb72e57be7fd;
      set -x;
      ulimit -Hn;
      ulimit -Sn;
      nvidia-smi;
      free -m;
      cd /home/sa_atari;
      python3 -u slot_attention_atari/training_od.py --dataset 'tetrominoes' --train_path "/home/sa_atari/datasets/tetrominoes" --val_path "/home/sa_atari/datasets/tetrominoes" --task '18.04'  --beta 0 --device 'gpu' --batch_size 64 --max_steps 50000 --seed 17 --num_workers 4 --invariance;
      free -m;
";

singularity instance stop ml_env