#!/bin/bash

#SBATCH --job-name=firsttry
#SBATCH --output=/wolflab/itaytuviah/JiT/jobs-out-err/%x-%j.out
#SBATCH --error=/wolflab/itaytuviah/JiT/jobs-out-err/%x-%j.err
#SBATCH --open-mode=append      # Append instead of overwriting logs
#SBATCH --time=1500 # max time (minutes)
#SBATCH --gres=gpu:A100:1
#SBATCH --account=gpu-tad-users_v2
#SBATCH --partition=gpu-tad-pool
#SBATCH --mem-per-gpu=55000
#SBATCH --qos=owner

###### ignore SBATCH --mem-per-gpu=55000
###### salloc -A gpu-tad-users_v2 -p gpu-tad-pool --qos=owner --gres=gpu:A100:1 --time=02:00:00 --job-name=vscode && srun --pty bash

srun --gres=gpu:1 nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export USER_DIR="/wolflab/itaytuviah"
export PROJECT_DIR="${USER_DIR}/JiT"
export PATH="${USER_DIR}/miniconda3/bin:${PATH}"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# keep all caches/pkgs off $HOME
export SCRATCH_DIR="/scratch200/itaytuviah"
export XDG_CACHE_HOME=${USER_DIR}/.cache
export PIP_CACHE_DIR=${USER_DIR}/.cache/pip
export CONDA_PKGS_DIRS=${USER_DIR}/.cache/conda/pkgs
export CONDA_ENVS_PATH=${USER_DIR}/conda-envs

export IMAGENET_PATH=${SCRATCH_DIR}/datasets/tiny-imagenet-200
export OUTPUT_DIR=${SCRATCH_DIR}/JiT/outputs/first_try

export TMPDIR=${SCRATCH_DIR}/tmp
export TMP=${TMPDIR}
export TEMP=${TMPDIR}
mkdir -p "${TMPDIR}"


module purge 2>/dev/null || true
unset LD_PRELOAD
export LD_LIBRARY_PATH=""
export TORCH_DISABLE_ITTNOTIFY=1


conda run -n jit torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
 main_jit.py \
 --model JiT-B/16 \
 --proj_dropout 0.0 \
 --P_mean -0.8 --P_std 0.8 \
 --img_size 64 --noise_scale 1.0 \
 --batch_size 128 --blr 5e-5 \
 --epochs 200 --warmup_epochs 5 \
 --gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
 --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
 --data_path ${IMAGENET_PATH} --online_eval 

# conda env remove -n jit
# conda env create -f environment.yaml

# conda run -n jit kaggle datasets download -p /scratch200/itaytuviah/datasets ayaroshevskiy/downsampled-imagenet-64x64

# conda run -n jit kaggle competitions files -c imagenet-object-localization-challenge

# conda run -n jit kaggle datasets download mariamalkuwaiti/tiny-imagenet-200-zip -p /scratch200/itaytuviah/datasets

# unzip /scratch200/itaytuviah/datasets/tiny-imagenet-200-zip.zip -d /scratch200/itaytuviah/datasets
