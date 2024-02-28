Relaxation of LaH10 (R-3m) structure at 160 GPa and 300 K within stochastic self-constistent harmonic approxiamtion (SSCHA) using moment tensor potentials (MTP) actively learned on energies, forces, and stresses calculated with Quantum Espresso (QE) package. The relaxed structure should have Fm-3m spacegroup.

Files:

POSCAR - initial structure of LaH10 (R-3m) relaxed at 160 GPa and 0 K. 
pot.almtp - file with untrained MTP that will be learned during the SSCHA relaxation.
input_sscha.py - file with input parameters where strings with parameters that should be inevitably changed by user are starting from "!!!".
jobscript - the script for SLURM workload manager that is also should be edited or completely changed according to user's hpc settings.

Directories:

pseudos - directory that contains pseudopotentials for QE.
reference - directory that contains result of SSCHA relaxation for comparison.