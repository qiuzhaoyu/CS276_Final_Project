#!/bin/bash 
#SBATCH --job-name=mysbatch
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=8
##SBATCH --mem=256G                 # 分配的内存
#SBATCH --gres=gpu:1
##SBATCH -x bme_gpu[01,02,03,04,05,06,07,08,09,10]
#SBATCH -x bme_gpu[01,02]
#SBATCH --partition=bme_gpu 
#SBATCH -o /public_bme2/bme-dgshen/ZhaoyuQiu/CS276_Final_Project/scripts/out.out  # 标准输出日志
#SBATCH -e /public_bme2/bme-dgshen/ZhaoyuQiu/CS276_Final_Project/scripts/err.err  # 错误日志

# 打印作业信息
echo "Job running on nodes: ${SLURM_JOB_NODELIST}"  # 打印节点列表
echo "Job started at: $(date)"                      # 打印开始时间

# module load apps/7/glibc/2.18
# module load compiler/gnu/13.1.0 
module load cuda/7/11.8

python -u /public_bme2/bme-dgshen/ZhaoyuQiu/CS276_Final_Project/DINOv2_for_GI_Tract_Image_Segmentation/efficientnet-segmentation.py

# 打印结束时间
echo "Job ended at: $(date)"
