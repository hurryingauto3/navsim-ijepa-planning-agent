#SBATCH --job-name=pdm_eval_ego_mlp_10pct
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128GB
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/eval_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/eval_%j.err

# FOR GPU JOBS, ADJUST --gres AND RESOURCES AS NEEDED

#SBATCH --job-name=ego_mlp_100pct
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_ego_mlp_100_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_ego_mlp_100_%j.err
#SBATCH --requeue

# =============================================================================

#!/bin/bash
#SBATCH --job-name=transfuser_ijepa_100pct
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=384GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_transfuser_ijepa_100_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_transfuser_ijepa_100_%j.err
#SBATCH --requeue


# =============================================================================


#!/bin/bash
#SBATCH --job-name=transfuser_ijepa_100pct
#SBATCH --partition=h200_tandon
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_transfuser_ijepa_100_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_transfuser_ijepa_100_%j.err
#SBATCH --requeue