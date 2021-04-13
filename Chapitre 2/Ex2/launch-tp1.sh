#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1

cd ${HOME}/Téléchargements/tp1
time /tp1.exe
