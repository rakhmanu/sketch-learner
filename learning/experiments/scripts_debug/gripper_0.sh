#!/bin/bash
#
#SBATCH -J gripper_0
#SBATCH -t 1-00:00:00
#SBATCH -C thin --exclusive
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dominik.drexler@liu.se

bash ./gripper.sh 0
