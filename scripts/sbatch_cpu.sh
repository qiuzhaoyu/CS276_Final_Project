#!/bin/bash 
#SBATCH -J test                # 作业名
#SBATCH -p bme_gpu                  # 分区名
#SBATCH -N 1                       # 使用的节点数
#SBATCH -n 8                    # 总核心数
#SBATCH -t 0:30:00                 # 运行时间
#SBATCH --mem=256G                 # 分配的内存
#SBATCH -o /public_bme2/bme-dgshen/ZhaoyuQiu/CS276_FinalProject/scripts/out.out  # 标准输出日志
#SBATCH -e /public_bme2/bme-dgshen/ZhaoyuQiu/CS276_FinalProject/scripts/err.err  # 错误日志

# 打印作业信息
echo "Job running on nodes: ${SLURM_JOB_NODELIST}"  # 打印节点列表
echo "Job started at: $(date)"                      # 打印开始时间

# module load apps/7/glibc/2.18
module load cuda/7/12.2

python -u /public_bme2/bme-dgshen/ZhaoyuQiu/CS276_FinalProject/test.py

# 打印结束时间
echo "Job ended at: $(date)"
