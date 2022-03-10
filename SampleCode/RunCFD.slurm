#!/bin/bash
################################################################
##                                                            ##
##                    Campus Cluster                          ##
##            Sample Open MPI Job Batch Script                ##
##                                                            ##
## PBS Options                                                ##
##                                                            ##
##  option -l                                                 ##
##    walltime: maximum wall clock time (hh:mm:ss)            ##
##       nodes: number of 16-core nodes                       ##
##                        20-core nodes                       ##
##                        24-core nodes                       ##
##                        28-core nodes                       ##
##                        40-core nodes                       ##
##                                                            ##
##         ppn: cores per node to use (1 thru 16)             ##
##                                    (1 thru 20)             ##
##                                    (1 thru 24)             ##
##                                    (1 thru 28)             ##
##                                    (1 thru 40)             ##
##                                                            ##
##  option -N                                                 ##
##    job name (default = name of script file)                ##
##                                                            ##
##  option -q                                                 ##
##    queue name ( -q name_of_queue )                         ##
##                                                            ##
##  option -o                                                 ##
##     filename for standard output at end of job             ##
##     (default = <job_name>.o<job_id>).  File is written to  ##
##     directory from which qsub was executed. Remove extra   ##
##     "##" from the PBS -o line if you want to name your     ##
##     own file.                                              ##
##                                                            ##
##  option -e                                                 ##
##     filename for standard error at end of job              ##
##     (default = <job_name>.e<job_id>).  File is written to  ##
##     directory from which qsub was executed. Remove extra   ##
##     "##" from the PBS -e line if you want to name your     ##
##     own file.                                              ##
##                                                            ##
##  option -j                                                 ##
##     Join the standard output and standard error streams    ##
##     of the job                                             ##
##     ( -j oe  merges stderr with stdout and generates a     ## 
##              stdout file <job_name>.o<job_id>              ##
##       -j eo  merges stdout with stderr and generates a     ##
##              stderr file <job_name>.e<job_id>  )           ##
##                                                            ##
##  option -m                                                 ##
##     mail_options (email notifications)                     ##
##     The mail_options argument is a string which consists   ## 
##     of either the single character "n", or one or more of  ##
##     the characters "a", "b", and "e".                      ##
##     ( -m a   Send mail when the job is aborted.            ##
##       -m be  Send mail when the job begins execution and   ##
##              when the job terminates.                      ##
##       -m n   Do not send mail.  )                          ##
##                                                            ##
################################################################
#

#SBATCH --time=4:0:00
#SBATCH --job-name="C_A0"
#SBATCH --mail-user=wz10@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=SOLUTION.OUT
#SBATCH --error=FAILURE.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --partition=test
#####################################


# Change to the directory from which the batch job was submittei
export SLURM_SUBMIT_DIR=/home/wz10/scratch/Deformed/RAE_Deform_1

# Change to the job directory 
cd $SLURM_SUBMIT_DIR
# nproc='cat $PBS_NODEFILE | wc -l'
# Run the MPI code



# shape_optimization.py -n 12 -g DISCRETE_ADJOINT -o SLSQP -f turb_SA_RAE2822.cfg 
#parallel_computation.py -n 20 -f turb_NACA0012.cfg
mpirun -n 14 SU2_DEF turb_RAE2822.cfg
 
# python3 Run.py
