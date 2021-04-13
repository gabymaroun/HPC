#!/usr/bin/env bash 
#SBATCH -N 4
#SBATCH -n 112
#SBATCH -c 1

cd /home/3/gmaroun/Bureau/tp3/Lab3/Ex1

mpirun -np 2 lab3mpi 1120
mpirun -np 4 lab3mpi 1120 
mpirun -np 8 lab3mpi 1120 
mpirun -np 16 lab3mpi 1120 
mpirun -np 28 lab3mpi 1120 
mpirun -np 56 lab3mpi 1120 
mpirun -np 84 lab3mpi 1120 
mpirun -np 112 lab3mpi 1120 


