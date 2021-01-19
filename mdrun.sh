#!/bin/bash

# gromacs 2018
# plumed 2.8

nsteps=5000000 #10ns
#nsteps=50000000 #100ns

gmx_mpi mdrun -plumed plumed.dat -s ../input.tpr -nsteps $nsteps
