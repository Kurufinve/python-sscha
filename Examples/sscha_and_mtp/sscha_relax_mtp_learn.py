#~/miniconda3/envs/sscha/bin/python
# from __future__ import print_function

"""
START INPUT PARAMETERS BLOCK (CHANGEBALE BY USER)
"""

# Input parameters are in input_sscha.py file

"""
END INPUT PARAMETERS BLOCK
"""




















"""
START MAIN BLOCK (NOT CHANGEBALE BY USER)
"""

# Try to load the parallel library if any
try:
    from mpi4py import MPI
    __MPI__ = True
except:
    __MPI__ = False

import sys,os
import itertools
from scipy.spatial.distance import cdist


# default values for some parameters
relax_zero_chgnet = False # we do not relax structure at 0 K with CHGNet on default 
relax_zero_mtp = False # we do not relax structure at 0 K with MTP on default 

init_dyn_chgnet = True # we calculate initial dynamicl matrix with CHGNet on default 
init_dyn_mtp = False # we do not calculate initial dynamical matrix with MTP on default

relax_sscha_chgnet = False # we do not conduct SSCHA relaxation with CHGNet on default
relax_sscha_mtp = True # we conduct SSCHA relaxation with actively learned MTP on default

relax_sscha_supercell = False # final sscha relaxation on large ensemble

pretrain_on_rand_structures = False # we do not pretrain MTP on randomly displaced structures on default because we use CHGNet for calculating initial dynamical matrix 
rand_disp_magnitude = 0.05

retrain = False
train_on_initial_ensemble = False
train_on_every_ensemble = True # we train MTP every time the new ensemble is generated on default  
train_local_mtps = False # we do not train specific MTP for every new ensemble on default but retrain exisitng one
iteration_limit = 500
include_stress = True # we include ab initio calculated stresses in the training set on default

np_ab_initio = 1
np_mlp_train = 1 

min_distances = None # we do not specify the minimal distance constraints between atoms by default

correct_pressure = True # we do not correct pressure by default
correct_free_energy = True # we correct the free energy by default

nq1 = 1 
nq2 = 1 
nq3 = 1 

nqs1 = nq1*4 
nqs2 = nq2*4 
nqs3 = nq3*4 

try:
    input_sscha = sys.argv[1]
except IndexError: 
    print('The input file should be pointed')
    exit()


from input_sscha import *
import cellconstructor as CC
import cellconstructor.Phonons
from cellconstructor.Phonons import compute_phonons_finite_displacements_sym
import cellconstructor.Structure
from cellconstructor.calculators import Espresso
from ase.calculators.espresso import Espresso as ASEspresso

import sscha, sscha.Ensemble, sscha.SchaMinimizer, sscha.Relax, sscha.Utilities
if use_hpc: import sscha.Cluster

import ase
from ase.atoms import Atoms
from ase.build.supercells import make_supercell
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
# from ase.optimize import GPMin
from ase.phonons import Phonons
from ase.calculators.lammpsrun import LAMMPS
# from ase.calculators.lammps import write_lammps_in
from ase.units import Bohr, Ry
import subprocess
from os.path import join as pj, exists as ex
import seekpath as sp
import numpy as np
import matplotlib.pyplot as plt  # noqa
import dill
from datetime import datetime
import spglib

# importing CHGNet only if we are going to use it
if relax_zero_chgnet or init_dyn_chgnet or relax_sscha_chgnet:
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator

# importing MTP functions only if we are going to use them
# if relax_zero_mtp or init_dyn_mtp or relax_sscha_mtp:
if 1:
    from sscha.MTP import train_mtp_on_cfg, \
                          ase_structures_list_to_cfg, \
                          calc_ngkpt, \
                          one_cfg_to_atoms, \
                          split_cfg, \
                          calc_to_cfg,\
                          are_ion_distances_good, \
                          convert_min_distances_to_bl, \
                          atoms_too_close



if retrain:
    # force setting not to train local MTPs if retrain (i.e. active learning) mode is activated
    train_local_mtps = False 

### Conversion factors
RY_TO_EV = 13.6057039763
icm_to_eV = 1.23981e-4
icm_to_thz = 2.99792458e-2

# Physical constants
hbar = 6.582119569e-16  # eV*s
kB = 8.617333262145e-5  # eV/K
e = 1.60217662e-19
Na = 6.0221409e23

"""
Dos functions

"""

def plot_phonons_ase(ase_ph, ase_struct):

    input_sp = (ase_struct.get_cell(), ase_struct.get_scaled_positions(), ase_struct.get_atomic_numbers())
    # determining k-path
    sp_obj = sp.get_path(input_sp) # seek path object
    N_POINTS = 1000
    SPECIAL_POINTS = {point.replace('GAMMA','G'): coords for (point,coords) in sp_obj['point_coords'].items()}
    PATH = ''.join(p[0]+p[1] for p in sp_obj['path']).replace('GAMMA','G')
    print(PATH)
    print(SPECIAL_POINTS)

    path = ase_struct.cell.bandpath(PATH, npoints=100)
    bs = ase_ph.get_band_structure(path)

    dos = ase_ph.get_dos(kpts=(20, 20, 20)).sample_grid(npts=100, width=1e-3)

    # Plot the band structure and DOS:
    fig = plt.figure(1, figsize=(7, 4))
    ax = fig.add_axes([.12, .07, .67, .85])

    emax = 0.035
    bs.plot(ax=ax, emin=0.0, emax=emax)

    dosax = fig.add_axes([.8, .07, .17, .85])
    dosax.fill_between(dos.get_weights(), dos.get_energies(), y2=0, color='grey',
                    edgecolor='k', lw=1)

    dosax.set_ylim(0, emax)
    dosax.set_yticks([])
    dosax.set_xticks([])
    dosax.set_xlabel("DOS", fontsize=18)

    fig.savefig('phonon_dispersion_ase.png')

    return

def plot_phonons_cc(dyn):

    ase_struct = dyn.structure.get_ase_atoms()

    input_sp = (ase_struct.get_cell(), ase_struct.get_scaled_positions(), ase_struct.get_atomic_numbers())
    # determining k-path
    sp_obj = sp.get_path(input_sp) # seek path object
    N_POINTS = 1000
    SPECIAL_POINTS = {point.replace('GAMMA','G'): coords for (point,coords) in sp_obj['point_coords'].items()}
    PATH = ''.join(p[0]+p[1] for p in sp_obj['path']).replace('GAMMA','G')
    print(PATH)
    print(SPECIAL_POINTS)

    qpath, data = CC.Methods.get_bandpath(dyn.structure.unit_cell,
                                        PATH,
                                        SPECIAL_POINTS,
                                        N_POINTS)
    xaxis, xticks, xlabels = data

    dispersion = CC.ForceTensor.get_phonons_in_qpath(dyn, qpath)
    nmodes = dyn.structure.N_atoms*3

    plt.figure(dpi = 150)
    ax = plt.gca()
    for i in range(nmodes):
        lbl=None
        ax.plot(xaxis, dispersion[:,i], color = 'r', label = lbl)
            
    # Plot vertical lines for each high symmetry points
    for x in xticks:
        ax.axvline(x, 0, 1, color = "k", lw = 0.4)
        ax.axhline(0, 0, 1, color = 'k', ls = ':', lw = 0.4)
        # Set the x labels to the high symmetry points
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Q path")
    ax.set_ylabel("Phonons [cm-1]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("phonon_dispersion_cc.png")

    return

def get_debye_temperature(phonon_freq_cm, phonon_dos, N_at):
    """
    Get Debye temperature from phonon dos.

    input:
    phonon_freq_cm - list of phonon frequencies in cm units
    phonon_dos - list with values of phonon dos at phonon freqs
    N_at - number of atoms in unit cell

    output:
    Debye temperature

    function was taken from here:
    https://github.com/usnistgov/jarvis/blob/master/jarvis/analysis/phonon/dos.py
    formulation for Debye temperature was taken from here:
    http://dx.doi.org/10.1103/PhysRevB.89.024304
    Eq. 10    
    """

    n = N_at
    omega = np.array(phonon_freq_cm) * icm_to_eV
    gomega = np.array(phonon_dos)
    integ = np.trapz(omega ** 2 * gomega, omega) / np.trapz(gomega, omega)
    prefact = 1 / kB
    # TODO: check if np.pi / 2 is required
    moment_debye = (
        n ** (-1 / 3)
        * (prefact)
        * np.sqrt(5 / 3 * integ)
        # np.pi / 2 * n ** (-1 / 3) * (prefact) * np.sqrt(5 / 3 * integ)
    )
    return moment_debye


"""
Helping functions

"""

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def get_min_dist(input_ase_structure):
    """
    Function for finding minimum distance between atoms in the structure.


    """

    positions = input_ase_structure.positions



    return


def get_randomly_displaced_structures(input_ase_structure, Nrand, magnitude=0.05):

    """
    Function for obtaining randomly displaced structures from the input structure.
    Useful in cases when pre-learning of MTP is needed.

    Input:

    input_ase_structure - ase structure object
    Nrand - number of randomly displaced configurations
    magnitude - displacement magnitude in Angstroms 

    Returns:

    list of ase_structure objects with randomly displaced configurations

    """

    # cc_struct = CC.Structure.Structure()
    # cc_struct.generate_from_ase_atoms(input_ase_structure)

    # dyn = CC.Phonons.Phonons(cc_struct)

    # random_structures = dyn.ExtractRandomStructures(Nrand, T, sobol = sobol)

    # return [s.get_ase_atoms() for s in random_structures]

    positions = input_ase_structure.positions

    randomly_displaced_structures = []

    randomly_displaced_structures.append(input_ase_structure)

    for i in range(Nrand):
        # displacement_mask = np.zeros_like(np.array(positions))
        displacements = np.random.uniform(low=-magnitude, high=magnitude, size=positions.shape)
        positions_new = positions + displacements
        displaced_ase_structure = input_ase_structure.copy()
        displaced_ase_structure.positions = positions_new
        randomly_displaced_structures.append(displaced_ase_structure)

    return randomly_displaced_structures

def setup_cluster(hpc_hostname,
                  hpc_workdir,
                  local_workdir,
                  hpc_partition_name,
                  hpc_use_account,
                  hpc_add_set_minus_x,
                  hpc_binary,
                  hpc_load_modules,
                  hpc_n_cpu,
                  hpc_n_nodes,
                  hpc_n_pool,
                  hpc_mpi_cmd,
                  hpc_use_partition,
                  hpc_use_qos,
                  hpc_batch_size,
                  hpc_job_number,
                  hpc_time,
                  hpc_terminal):

    my_hpc = sscha.Cluster.Cluster(pwd = None)

    my_hpc.hostname = hpc_hostname
    # In the name of workdir the datetime temporarily used to distinguish different calculations with the same labels in the hpc_workdir
    # TODO: bind the name of the workdir on cluster with USPEX structure ID
    my_hpc.workdir = hpc_workdir + '/' + local_workdir
    my_hpc.partition_name = hpc_partition_name
    my_hpc.use_account = hpc_use_account
    my_hpc.label = "ESP"
    my_hpc.add_set_minus_x = hpc_add_set_minus_x
    my_hpc.binary = hpc_binary
    my_hpc.load_modules = hpc_load_modules
    my_hpc.n_cpu = hpc_n_cpu
    my_hpc.n_nodes = hpc_n_nodes
    my_hpc.n_pool = hpc_n_pool
    my_hpc.mpi_cmd = hpc_mpi_cmd
    my_hpc.use_partition = hpc_use_partition 
    my_hpc.use_qos = hpc_use_qos
    my_hpc.batch_size = hpc_batch_size
    my_hpc.job_number = hpc_job_number
    my_hpc.time = hpc_time
    my_hpc.local_workdir = local_workdir
    my_hpc.terminal = hpc_terminal

    my_hpc.setup_workdir()
    
    return my_hpc

def get_nq_translations():
    return


"""
Functions that do the main job

"""

def get_relaxed_structure_chgnet(input_structure, pressure = 0.0):

    # Get the ASE structure object
    ase_struct = get_ordered_structure(ase.io.read(input_structure),specorder) 

    # if rank == 0:
    #     ase_struct = ase.io.read(input_structure) 
    #     if __MPI__:
    #         ase_struct = comm.bcast(ase_struct, root=0)

    chgnet = CHGNet.load()
    calc = CHGNetCalculator()

    ase_struct.calc = calc

    ucf = UnitCellFilter(ase_struct, mask = [1,1,1,1,1,1], scalar_pressure=pressure)

    dyn = BFGS(ucf)
    print("Initial Energy", ase_struct.get_potential_energy())
    dyn.run(fmax=0.001)
    print("Final Energy", ase_struct.get_potential_energy())

    ase_struct.write('POSCAR_r', format = 'vasp')

    return ase_struct


def get_relaxed_structure_mtp(input_structure, calculator = 'LAMMPS', calculator_run_command = 'lmp', calculator_parameters = None,
                          calculator_files = None):
    """
    This function gets an input structure and relax it with ASE and the specified calculator.
    The obtained harmonic dynamical matrix can be used as a starting point for SSCHA optimization of the structure.  
    The input_struct should be in the format that is readable by ase.io.read method (see https://wiki.fysik.dtu.dk/ase/ase/io/io.html)

    Returns: dynamical matrix as CellConstructor object
    """

    """
    This function gets an input structure and calculate the dynamical matrix with ASE and the specified calculator within the harmonic approximation.
    The obtained harmonic dynamical matrix can be used as a starting point for SSCHA optimization of the structure.  
    The input_struct should be in the format that is readable by ase.io.read method (see https://wiki.fysik.dtu.dk/ase/ase/io/io.html)

    Returns: dynamical matrix as CellConstructor object
    """
    print(f'calculator is {calculator}')
    print(f'calculator_run_command is {calculator_run_command}')
    print(f'calculator_parameters are {calculator_parameters}')
    print(f'calculator_files are {calculator_files}')

    # Get the ASE structure object
    ase_struct = ase.io.read(input_structure) 

    # print(ase_struct)

    # Attach the calculator and relax the structure
    if calculator == 'LAMMPS':
        if 'ASE_LAMMPSRUN_COMMAND' not in os.environ:
            os.environ['ASE_LAMMPSRUN_COMMAND'] = calculator_run_command
        # exitcode = os.system(calculator_run_command)
        calc = LAMMPS(parameters = calculator_parameters,
                      specorder = specorder,
                      files = calculator_files,
                      keep_alive = False,
                    #   trajectory_out = open('relax.dump','w'),
                      verbose = True)
        
        # ase_struct.write('relax.lmp', format = 'lammps-data')
        ase_struct.calc = calc

    # ase_struct.get_potential_energy()

    # Starting active learning loop
    while True:
        try: 
            ase_struct.get_potential_energy()
            break
        except RuntimeError:
            if retrain == True:
                # if MTP is not able to reproduce the configuration we add it to the training set
                train_mtp_on_cfg(specorder, mlip_run_command, pot_name, 
                    ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster,
                    iteration_limit, energy_weight, force_weight, stress_weight,include_stress,train_local_mtps=False,pop='retrain_relax')
                # read the last relaxation state to continue relaxation from this point
                ase_struct = ase.io.read('relax.dump',specorder = specorder)
                ase_struct.calc = calc
                continue
            else:
                print('Runtime Error!')
                break



    ase_struct_relaxed = ase.io.read('relax.dump',specorder = specorder)
    
    # optimizer = BFGS(ase_struct)
    # optimizer.run(fmax = 0.01)

    ase_struct_relaxed.write('POSCAR_r', format = 'vasp')

    return ase_struct_relaxed


def get_initial_dyn_mtp(input_structure, calculator = 'LAMMPS', calculator_run_command = 'lmp', calculator_parameters = None,
                    calculator_files = None, nq1 = 1, nq2 = 1, nq3 = 1):
    """
    This function gets an input structure and calculate the dynamical matrix with ASE and the specified calculator within the harmonic approximation.
    The obtained harmonic dynamical matrix can be used as a starting point for SSCHA optimization of the structure.  
    The input_struct should be in the format that is readable by ase.io.read method (see https://wiki.fysik.dtu.dk/ase/ase/io/io.html)

    Returns: dynamical matrix as CellConstructor object
    """
    print(f'calculator is {calculator}')
    print(f'calculator_run_command is {calculator_run_command}')
    print(f'calculator_parameters are {calculator_parameters}')
    print(f'calculator_files are {calculator_files}')

    # Get the ASE structure object
    # ase_struct = ase.io.read(input_structure) 
    ase_struct = input_structure 

    # print(ase_struct)

    # Attach the calculator and relax the structure
    if calculator == 'LAMMPS':
        if 'ASE_LAMMPSRUN_COMMAND' not in os.environ:
            os.environ['ASE_LAMMPSRUN_COMMAND'] = calculator_run_command
        # exitcode = os.system(calculator_run_command)
        calc = LAMMPS(parameters = calculator_parameters,
                      specorder = specorder,
                      files = calculator_files,
                      keep_alive = False,
                      verbose = True)
        ase_struct.calc = calc

    elif calculator == 'QE' and ab_initio_calculator == 'QE':
        k_points = calc_ngkpt(ase_struct.cell.reciprocal(),ab_initio_kresol)

        calc = ASEspresso(pseudopotentials = ab_initio_pseudos,
                          input_data = ab_initio_parameters,
                          kpts = k_points,
                          command = ab_initio_run_command,
                          label = 'ESP',
                          koffset  = (0,0,0))
        ase_struct.calc = calc

        # print(calc.get_lammps_command())

    # Get the Harmonic dynamical matrix
    print(f"Calculating phonons for input structure with ASE Phonons using {nq1}x{nq2}x{nq3} supercell")
    ase_ph = Phonons(ase_struct, ase_struct.calc,
                       supercell = (nq1,nq2,nq3),
                       delta = 0.05)

    # try: 
    #     ase_ph.run()
    # except RuntimeError:
    #     # if MTP is not able to reproduce the configuration we add it to the training set
    #     train_mtp_on_cfg(specorder, mlip_run_command, pot_name, ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, iteration_limit)

    while True:
        try: 
            ase_ph.run()
            break
        except RuntimeError:
            if retrain == True:
                # if MTP is not able to reproduce the configuration we add it to the training set
                if rank == 0:
                    train_mtp_on_cfg(specorder, mlip_run_command, pot_name, 
                        ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster,
                        iteration_limit, energy_weight, force_weight, stress_weight,include_stress,train_local_mtps=False,pop='retrain_ph')
                # comm.Barrier()
                continue
            else:
                print('Runtime Error!')
                break            

    try:
        ase_ph.read(acoustic = True)
    except TypeError:
        if rank == 0:
            os.system('rm -rf phonon')  
        # comm.Barrier()      
        ase_ph.run()
        ase_ph.read(acoustic = True)
        
    ase_ph.clean()

    # plot_phonons_ase(ase_ph, ase_struct)


    # Generate the relaxed CellConstructor structure from the relaxed ASE structure
    cc_struct = CC.Structure.Structure()
    cc_struct.generate_from_ase_atoms(ase_struct)

    # Generate the CellConstructor Dynamical Matrix
    print("Generating the CellConstructor Dynamical Matrix")
    dyn = CC.Phonons.get_dyn_from_ase_phonons(ase_ph)

    # Save the dynamical matrix in the QE format for SSCHA
    dyn.Symmetrize() # Apply the symmetries
    if rank == 0:
        dyn.save_qe("dyn_init")

    # Diagonalize the dynamical matrix
    try:
        w, pols = dyn.DiagonalizeSupercell()
        for i in range(len(w)):
            print("{:2d}) {:16.8f} cm-1".format(i+1,
                                                w[i] * CC.Units.RY_TO_CM))
    except:
        print('Error in diagonalizing the dynamical matrix!')
    
    # Plot phonon dispersion
    try:
        plot_phonons_cc(dyn)
    except:
        print('Error in ploting phonon dispersion!')

    return dyn


def get_initial_dyn_chgnet(input_structure, nq1=1, nq2=1, nq3=1):
    """
    This function gets an input structure and calculate the dynamical matrix with ASE and the specified calculator within the harmonic approximation.
    The obtained harmonic dynamical matrix can be used as a starting point for SSCHA optimization of the structure.  
    The input_struct should be in the format that is readable by ase.io.read method (see https://wiki.fysik.dtu.dk/ase/ase/io/io.html)

    Returns: dynamical matrix as CellConstructor object
    """

    # Get the ASE structure object
    # ase_struct = ase.io.read(input_structure) 
    if isinstance(input_structure, ase.Atoms):
        ase_struct = input_structure 
    elif isinstance(input_structure, str):
        ase_struct = ase.io.read(input_structure) 

    chgnet = CHGNet.load()
    calc = CHGNetCalculator()

    ase_struct.calc = calc

    # Get the Harmonic dynamical matrix
    if rank == 0:
        print(f"Calculating phonons for input structure with ASE Phonons using {nq1}x{nq2}x{nq3} supercell")
    ase_ph = Phonons(ase_struct, ase_struct.calc,
                       supercell = (nq1,nq2,nq3),
                       delta = 0.05)

    ase_ph.run()

    try:
        ase_ph.read(acoustic = True)
    except TypeError:
        if rank == 0:
            os.system('rm -rf phonon')        
        ase_ph.run()
        ase_ph.read(acoustic = True)
        
    ase_ph.clean()

    # plot_phonons_ase(ase_ph, ase_struct)


    # Generate the relaxed CellConstructor structure from the relaxed ASE structure
    cc_struct = CC.Structure.Structure()
    cc_struct.generate_from_ase_atoms(ase_struct)

    # Generate the CellConstructor Dynamical Matrix
    # dyn = CC.Phonons.Phonons(cc_struct)
    if rank == 0:
        print("Generating the CellConstructor Dynamical Matrix")
    dyn = CC.Phonons.get_dyn_from_ase_phonons(ase_ph)


    # Copy the ASE force constant into the dynamical matrix
    # dyn.dynmats[0][:,:] = struct_ph.get_force_constant()[0, :, :]

    # Convert into Ry / Bohr^2
    # dyn.dynmats[0][:,:] *= Bohr**2 / Ry
    
    # Save the dynamical matrix in the quantum espresso format
    dyn.Symmetrize() # Apply the symmetries
    if rank == 0:
        dyn.save_qe("dyn_init")

    # Diagonalize the dynamical matrix
    # w, pols = dyn.DiagonalizeSupercell()
    # for i in range(len(w)):
    #     print("{:2d}) {:16.8f} cm-1".format(i+1,
    #                                         w[i] * CC.Units.RY_TO_CM))
    if rank == 0:
        try:
            plot_phonons_cc(dyn)
        except:
            print('Error in ploting phonon dispersion!')

    return dyn


def get_initial_dyn_mtp_cc_phonons(input_structure, calculator = 'LAMMPS', calculator_run_command = 'lmp', calculator_parameters = None,
                               calculator_files = None):

    if calculator == 'LAMMPS':
        if 'ASE_LAMMPSRUN_COMMAND' not in os.environ:
            os.environ['ASE_LAMMPSRUN_COMMAND'] = calculator_run_command
        calc = LAMMPS(parameters = calculator_parameters,
                      specorder = specorder,
                      files = calculator_files,
                      tmp_dir = './lammps-cc-phonons',
                      keep_alive = False,
                      verbose = True)
        



    cc_struct = CC.Structure.Structure()
    cc_struct.generate_from_ase_atoms(input_structure)

    dyn = compute_phonons_finite_displacements_sym(structure=cc_struct, 
                                                   ase_calculator=calc, 
                                                   epsilon=0.05, 
                                                   supercell=(1,1,1),
                                                   progress=-1,
                                                   progress_bar=False,
                                                   debug=False, 
                                                   timer=None)

    return dyn


def sscha_relaxation_chgnet(dyn_init='dyn_init1', 
                        min_step_dyn=0.05, min_step_struc=0.05, kong_liu_ratio=0.5, gradi_op='all', meaningful_factor=0.2, precond_dyn=True, root_representation='normal',
                        N_CONFIGS=1, MAX_ITERATIONS=20, PRESSURE=0, TEMPERATURE=0):

    dyn_init.Symmetrize()
    dyn_init.ForcePositiveDefinite()
    # Initializing the ensemble
    ensemble = sscha.Ensemble.Ensemble(dyn_init, TEMPERATURE, 
                                    supercell = dyn_init.GetSupercell())

    # Initializing the free energy minimizer
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)

    # We set up the minimization parameters
    minim.min_step_dyn = min_step_dyn     # The minimization step on the dynamical matrix
    minim.min_step_struc = min_step_struc   # The minimization step on the structure
    minim.kong_liu_ratio = kong_liu_ratio     # The parameter that estimates whether the ensemble is still good
    minim.gradi_op = gradi_op # Check the stopping condition on both gradients
    minim.meaningful_factor = meaningful_factor # How much small the gradient should be before I stop?

    minim.precond_dyn = precond_dyn
    minim.root_representation = root_representation

    chgnet = CHGNet.load()
    calc = CHGNetCalculator()

    # Initialize the NPT simulation
    relax = sscha.Relax.SSCHA(minim, 
                            ase_calculator = calc,
                            N_configs = N_CONFIGS,
                            max_pop = MAX_ITERATIONS,
                            save_ensemble = True,
                            cluster = None)

    # Define the I/O operations
    # To save info about the free energy minimization after each step
    ioinfo = sscha.Utilities.IOInfo()
    ioinfo.SetupSaving("minim_info")
    relax.setup_custom_functions(custom_function_post = ioinfo.CFP_SaveAll)

    print ("The spacegroup before SSCHA relaxation with CHGNet is:", spglib.get_spacegroup(dyn_init.structure.get_ase_atoms(), 0.05))
    if RELAX_SIMPLE:
        # run fixed cell simulation
        relax.relax(ensemble_loc = "ensembles_fixed_cell_chgnet")
        relaxed_ase_structure = relax.minim.dyn.structure.get_ase_atoms()
        ase.io.write('CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after the fixed cell SSCHA relaxation')
        ase.io.write('CHGNet.fixed_cell.CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after after the fixed cell SSCHA relaxation')
        print ("The spacegroup after CHGNet fixed cell relaxation is:", spglib.get_spacegroup(relax.minim.dyn.structure.get_ase_atoms(), 0.05))

    if RELAX_NVT:
        # run NVT simulation
        relax.vc_relax(fix_volume = True, static_bulk_modulus = 150, ensemble_loc = "ensembles_nvt_chgnet")
        relaxed_ase_structure = relax.minim.dyn.structure.get_ase_atoms()
        ase.io.write('CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after the fixed volume SSCHA relaxation')
        ase.io.write('CHGNet.nvt.CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after after the fixed volume SSCHA relaxation')
        print ("The spacegroup after CHGNet NVT relaxation is:", spglib.get_spacegroup(relax.minim.dyn.structure.get_ase_atoms(), 0.05))

    # we do NPT relaxation with CHGNET only if further SSCHA relaxation with MTP is not planned
    if RELAX_NPT and not relax_sscha_mtp:
        # run NPT simulation
        relax.vc_relax(target_press = PRESSURE, static_bulk_modulus = 200, stress_numerical = False, ensemble_loc = "ensembles_npt_chgnet")
        relaxed_ase_structure = relax.minim.dyn.structure.get_ase_atoms()
        ase.io.write('CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after the fixed pressure SSCHA relaxation')
        ase.io.write('CHGNet.npt.CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after after the fixed pressure SSCHA relaxation')
        print ("The spacegroup after CHGNet NPT relaxation is:", spglib.get_spacegroup(relax.minim.dyn.structure.get_ase_atoms(), 0.05))

    # sscha.Utilities.save_binary(relax,'relax_fix_volume.bin')
    # Saving the relax object with dill

    try:
        relax.minim.finalize()
        relax.minim.plot_results(save_filename='CHGNet_SSCHA_results.txt')
    except:
        pass
        print('CHGNet SSCHA RELAXATION PROBABLY WAS NOT FINISHED... BUT RESULTS CAN BE STILL USEFULL')

    print('CHGNet SSCHA RELAXATION ENDED')

    if relax.minim.is_converged():
        print('The CHGNet SSCHA relaxation is CONVERGED')
    elif not relax.minim.is_converged():
        print('WARNING! The CHGNet SSCHA relaxation is NOT CONVERGED!')

    N_AT = relax.minim.dyn.structure.N_atoms
    FREE_ENERGY, FREE_ENERGY_ERROR = relax.minim.get_free_energy(return_error=True)
    print(f'Final CHGNet Free Energy = {FREE_ENERGY*RY_TO_EV} eV/cell')
    print(f'Final CHGNet per atom Free Energy = {FREE_ENERGY*RY_TO_EV/N_AT} eV/atom')
    print(f'Final CHGNet Error in the Free Energy = {FREE_ENERGY_ERROR*RY_TO_EV} eV/cell')
    print(f'Final CHGNet Error per atom in the Free Energy = {FREE_ENERGY_ERROR*RY_TO_EV/N_AT} eV/atom')
    # print(dir(relax.minim))
    # print(dir(relax.minim.dyn))
    # print(dir(relax.minim.ensemble))
    ENS_FREE_ENERGY, ENS_FREE_ENERGY_ERROR = relax.minim.ensemble.get_free_energy(return_error=True)
    print(f'Final CHGNet Ensemble Free Energy = {ENS_FREE_ENERGY*RY_TO_EV} eV/cell')
    print(f'Final CHGNet Ensemble per atom Free Energy = {ENS_FREE_ENERGY*RY_TO_EV/N_AT} eV/atom')
    print(f'Final CHGNet Ensemble Error in the Free Energy = {ENS_FREE_ENERGY_ERROR*RY_TO_EV} eV/cell')
    print(f'Final CHGNet Ensemble Error per atom in the Free Energy = {ENS_FREE_ENERGY_ERROR*RY_TO_EV/N_AT} eV/atom')

    ENS_AVG_ENERGY, ENS_AVG_ENERGY_ERROR = relax.minim.ensemble.get_average_energy(return_error=True)
    print(f'Final CHGNet Ensemble Average Energy = {ENS_AVG_ENERGY*RY_TO_EV} eV/cell')
    print(f'Final CHGNet Ensemble per atom Average Energy = {ENS_AVG_ENERGY*RY_TO_EV/N_AT} eV/atom')
    print(f'Final CHGNet Ensemble Error in the Average Energy = {ENS_AVG_ENERGY_ERROR*RY_TO_EV/N_AT} eV/cell')
    print(f'Final CHGNet Ensemble Error per atom in the Average Energy = {ENS_AVG_ENERGY_ERROR*RY_TO_EV/N_AT} eV/atom')

    ase_structure = relax.minim.dyn.structure.get_ase_atoms()
    if rank == 0:
        ase.io.write('CONTCAR', ase_structure, "vasp", direct=True, label = 'Structure after the SSCHA relaxation')

        filename = 'CHGNet.ensemble.bin'
        try:
            relax.minim.ensemble.save_bin('CHGNet.ensemble.bin')
        except:
            print('Can not save the last ensemble in binary format')
            pass
        try:
            dill.dump(relax, open('CHGNet.relax.dump', "wb"))
        except:
            pass

    # comm.Barrier()

    return relax.minim.dyn, relax.minim.population




def sscha_relaxation_mtp(dyn_init='dyn_init1', calculator='LAMMPS', calculator_run_command='lmp', calculator_parameters=None, calculator_files=None, 
                        min_step_dyn=0.05, min_step_struc=0.05, kong_liu_ratio=0.5, gradi_op='all', meaningful_factor=0.2, precond_dyn=True, root_representation='normal',
                        N_CONFIGS=1, MAX_ITERATIONS=20, PRESSURE=0, TEMPERATURE=0,
                        specorder=None, pot_name=None, mlip_run_command=None,
                        ab_initio_calculator=None, ab_initio_parameters=None, ab_initio_run_command=None, ab_initio_kresol=None, ab_initio_pseudos=None, ab_initio_cluster=None, 
                        iteration_limit=500, energy_weight=1.0, force_weight=0.01, stress_weight=0.001, include_stress=False, 
                        train_on_every_ensemble=False, train_local_mtps=False, retrain=False, np_ab_initio = 1, start_pop = 1):

    dyn_init.Symmetrize()
    dyn_init.ForcePositiveDefinite()
    # Initializing the ensemble
    ensemble = sscha.MTP.Ensemble_MTP(dyn_init, TEMPERATURE, 
                                    supercell = dyn_init.GetSupercell(),
                                    specorder = specorder,
                                    pot_name = pot_name, 
                                    mlip_run_command = mlip_run_command, 
                                    ab_initio_calculator = ab_initio_calculator,
                                    ab_initio_parameters = ab_initio_parameters,
                                    ab_initio_run_command = ab_initio_run_command,
                                    ab_initio_kresol = ab_initio_kresol,
                                    ab_initio_pseudos = ab_initio_pseudos,
                                    ab_initio_cluster = ab_initio_cluster,
                                    iteration_limit = iteration_limit,
                                    energy_weight = energy_weight,
                                    force_weight = force_weight,
                                    stress_weight = stress_weight,
                                    include_stress = include_stress,
                                    retrain = retrain,
                                    np_ab_initio = np_ab_initio,
                                    np_mlp_train = np_mlp_train,
                                    min_distances = min_distances)

    # Initializing the free energy minimizer
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)

    # We set up the minimization parameters
    minim.min_step_dyn = min_step_dyn     # The minimization step on the dynamical matrix
    minim.min_step_struc = min_step_struc   # The minimization step on the structure
    minim.kong_liu_ratio = kong_liu_ratio     # The parameter that estimates whether the ensemble is still good
    minim.gradi_op = gradi_op # Check the stopping condition on both gradients
    minim.meaningful_factor = meaningful_factor # How much small the gradient should be before I stop?

    minim.precond_dyn = precond_dyn
    minim.root_representation = root_representation

    if calculator == 'LAMMPS':
        os.environ['ASE_LAMMPSRUN_COMMAND'] = calculator_run_command
        # exitcode = os.system(calculator_run_command)
        calc = LAMMPS(parameters = calculator_parameters,
                      specorder = specorder,
                      files = calculator_files,
                      keep_alive = False,
                      directory = './',
                      verbose = True)

    else:
        raise(NotImplementedError)

    # Initialize the NPT simulation
    relax = sscha.MTP.SSCHA_MTP(minim, 
                            ase_calculator = calc,
                            N_configs = N_CONFIGS,
                            max_pop = MAX_ITERATIONS,
                            save_ensemble = True,
                            cluster = None, 
                            specorder = specorder,
                            pot_name = pot_name, 
                            mlip_run_command = mlip_run_command, 
                            ab_initio_calculator = ab_initio_calculator,
                            ab_initio_parameters = ab_initio_parameters,
                            ab_initio_run_command = ab_initio_run_command,
                            ab_initio_kresol = ab_initio_kresol,
                            ab_initio_pseudos = ab_initio_pseudos,
                            ab_initio_cluster = ab_initio_cluster,
                            iteration_limit = iteration_limit,
                            energy_weight = energy_weight,
                            force_weight = force_weight,
                            stress_weight = stress_weight,
                            include_stress = include_stress,
                            train_on_every_ensemble = train_on_every_ensemble,
                            train_local_mtps = train_local_mtps,
                            retrain = retrain,
                            np_ab_initio = np_ab_initio)

    # Define the I/O operations
    # To save info about the free energy minimization after each step
    ioinfo = sscha.Utilities.IOInfo()
    ioinfo.SetupSaving("minim_info")
    relax.setup_custom_functions(custom_function_post = ioinfo.CFP_SaveAll)

    print ("The original spacegroup is:", spglib.get_spacegroup(dyn_init.structure.get_ase_atoms(), 0.05))
    if RELAX_SIMPLE:
        # run fixed cell simulation
        while True:
            try: 
                relax.relax(ensemble_loc = "ensembles_fixed_cell_mtp",start_pop=start_pop)
                relaxed_ase_structure = relax.minim.dyn.structure.get_ase_atoms()
                ase.io.write('CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after the fixed cell SSCHA relaxation')
                ase.io.write('fixed_cell.CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after after the fixed cell SSCHA relaxation')
                start_pop = relax.minim.population
                break
            except RuntimeError:
                if retrain == True:
                    # if MTP is not able to reproduce the configuration we add it to the training set
                    train_mtp_on_cfg(specorder, mlip_run_command, pot_name, 
                        ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster,
                        iteration_limit, energy_weight, force_weight, stress_weight,include_stress,train_local_mtps=False,pop='retrain_sscha_fixed_cell')
                    continue
                else:
                    print('Runtime Error!')
                    break   
        print ("The spacegroup after fixed cell relaxation with MTP is:", spglib.get_spacegroup(relax.minim.dyn.structure.get_ase_atoms(), 0.05))

    if RELAX_NVT:
        # run NVT simulation
        while True:
            try: 
                # relax.vc_relax(target_press = PRESSURE, ensemble_loc = "ensembles_nvt", fix_volume = True)
                relax.vc_relax(fix_volume = True, static_bulk_modulus = 150, ensemble_loc = "ensembles_nvt_mtp",start_pop=start_pop)
                relaxed_ase_structure = relax.minim.dyn.structure.get_ase_atoms()
                ase.io.write('CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after the fixed volume SSCHA relaxation')
                ase.io.write('nvt.CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after after the fixed volume SSCHA relaxation')
                start_pop = relax.minim.population
                break
            except RuntimeError:
                if retrain == True:
                    # if MTP is not able to reproduce the configuration we add it to the training set
                    train_mtp_on_cfg(specorder, mlip_run_command, pot_name, 
                        ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster,
                        iteration_limit, energy_weight, force_weight, stress_weight,include_stress,train_local_mtps=False,pop='retrain_sscha_nvt')
                    continue
                else:
                    print('Runtime Error!')
                    break   

        print ("The spacegroup after NVT relaxation with MTP is:", spglib.get_spacegroup(relax.minim.dyn.structure.get_ase_atoms(), 0.05))

    if RELAX_NPT:
        # get pressure correction for MTP
        if (RELAX_SIMPLE or RELAX_NVT) and correct_pressure:
            n_cfg = int(os.popen('grep "BEGIN_CFG" set.cfg | wc -l').read().split()[0])
            if n_cfg >= 100:
                n_conf = 100
            else: 
                n_conf = n_cfg
            press_err = get_pressure_correction_from_training_set(mlip_run_command = mlip_run_command,
                                    pot_name = pot_name,
                                    set_name = 'set.cfg',
                                    N_configs = n_conf
                                    )
            PRESSURE = PRESSURE - press_err        
        # run NPT simulation
        while True:
            try:
                relax.vc_relax(target_press = PRESSURE, static_bulk_modulus = 200, stress_numerical = False, ensemble_loc = "ensembles_npt_mtp",start_pop=start_pop)
                relaxed_ase_structure = relax.minim.dyn.structure.get_ase_atoms()
                ase.io.write('CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after the fixed pressure SSCHA relaxation')
                ase.io.write('npt.CONTCAR', relaxed_ase_structure, "vasp", direct=True, label = 'Structure after after the fixed pressure SSCHA relaxation')
                start_pop = relax.minim.population
                break
            except RuntimeError:
                if retrain == True:
                    # if MTP is not able to reproduce the configuration we add it to the training set
                    train_mtp_on_cfg(specorder, mlip_run_command, pot_name, 
                        ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster,
                        iteration_limit, energy_weight, force_weight, stress_weight,include_stress,train_local_mtps=False,pop='retrain_sscha_npt')

                    continue
                else:
                    print('Runtime Error!')
                    break   
        print ("The spacegroup after NPT relaxation with MTP is:", spglib.get_spacegroup(relax.minim.dyn.structure.get_ase_atoms(), 0.05))

    # sscha.Utilities.save_binary(relax,'relax_fix_volume.bin')
    # Saving the relax object with dill

    try:
        relax.minim.finalize()
        relax.minim.plot_results(save_filename='SSCHA_results.txt')
    except:
        pass
        print('SSCHA RELAXATION PROBABLY WAS NOT FINISHED... BUT RESULTS CAN BE STILL USEFULL')

    print('SSCHA RELAXATION ENDED')

    if relax.minim.is_converged():
        print('The SSCHA relaxation is CONVERGED')
    elif not relax.minim.is_converged():
        print('WARNING! The SSCHA relaxation is NOT CONVERGED!')

    N_AT = relax.minim.dyn.structure.N_atoms
    FREE_ENERGY, FREE_ENERGY_ERROR = relax.minim.get_free_energy(return_error=True)

    if correct_free_energy:
        n_cfg = int(os.popen('grep "BEGIN_CFG" set.cfg | wc -l').read().split()[0])
        if n_cfg >= 100:
            n_conf = 100
        else: 
            n_conf = n_cfg
        total_corr, first_order_corr, second_order_corr = get_thermodynamic_correction_from_training_set(T = TEMPERATURE,
                                 mlip_run_command = mlip_run_command,
                                 pot_name = pot_name,
                                 set_name = 'set.cfg',
                                 N_configs = n_conf
                                 )

    else:
        total_corr, first_order_corr, second_order_corr = 0.0, 0.0, 0.0

    print(f'Final Free Energy = {FREE_ENERGY*RY_TO_EV + total_corr*N_AT} eV/cell')
    print(f'Final per atom Free Energy = {FREE_ENERGY*RY_TO_EV/N_AT + total_corr} eV/atom')
    print(f'Final Error in the Free Energy = {FREE_ENERGY_ERROR*RY_TO_EV} eV/cell')
    print(f'Final Error per atom in the Free Energy = {FREE_ENERGY_ERROR*RY_TO_EV/N_AT} eV/atom')

    ENS_FREE_ENERGY, ENS_FREE_ENERGY_ERROR = relax.minim.ensemble.get_free_energy(return_error=True)
    print(f'Final Ensemble Free Energy = {ENS_FREE_ENERGY*RY_TO_EV + total_corr*N_AT} eV/cell')
    print(f'Final Ensemble per atom Free Energy = {ENS_FREE_ENERGY*RY_TO_EV/N_AT + total_corr} eV/atom')
    print(f'Final Ensemble Error in the Free Energy = {ENS_FREE_ENERGY_ERROR*RY_TO_EV} eV/cell')
    print(f'Final Ensemble Error per atom in the Free Energy = {ENS_FREE_ENERGY_ERROR*RY_TO_EV/N_AT} eV/atom')

    ENS_AVG_ENERGY, ENS_AVG_ENERGY_ERROR = relax.minim.ensemble.get_average_energy(return_error=True)
    print(f'Final Ensemble Average Energy = {ENS_AVG_ENERGY*RY_TO_EV + total_corr*N_AT} eV/cell')
    print(f'Final Ensemble per atom Average Energy = {ENS_AVG_ENERGY*RY_TO_EV/N_AT + total_corr} eV/atom')
    print(f'Final Ensemble Error in the Average Energy = {ENS_AVG_ENERGY_ERROR*RY_TO_EV/N_AT} eV/cell')
    print(f'Final Ensemble Error per atom in the Average Energy = {ENS_AVG_ENERGY_ERROR*RY_TO_EV/N_AT} eV/atom')

    ase_structure = relax.minim.dyn.structure.get_ase_atoms()
    ase.io.write('CONTCAR', ase_structure, "vasp", direct=True, label = 'Structure after the SSCHA relaxation')

    filename = 'ensemble.bin'
    try:
        relax.minim.ensemble.save_bin('ensemble.bin')
    except:
        print('Can not save the last ensemble in binary format')
        pass
    try:
        dill.dump(relax, open('relax.dump', "wb"))
    except:
        pass
    # Loading the relax object with dill
    # filename = 'relax.bin'
    # relax_loaded = dill.load(open(filename, "rb"))
    try:
        # relax.minim.finalize()
        relax.minim.plot_results()
    except:
        pass

    return relax.minim.dyn, relax.minim.population

def get_thermodynamic_correction_from_training_set(T = 0,
                                 mlip_run_command = 'mpirun -np 1 mlp',
                                 pot_name = 'pot.almtp',
                                 set_name = 'set.cfg',
                                 N_configs = 100
                                 ):


    """
    This function calculates a correction to the free energy using 
    thermodynamic perturbation theory from existing training set for MTP.
    """
    get_errors_cmd = f'{mlip_run_command} check_errors {pot_name} {set_name} --log=mtp_errors'

    print(get_errors_cmd)
    os.system(get_errors_cmd)

    diff_enes = []
    diff_epas = []

    with open('mtp_errors.0','r') as f:
        lines = f.readlines()
        for line in lines:
            diff_ene = -float(line.split()[5].split(':')[1])
            diff_epa = -float(line.split()[6].split(':')[1])
            diff_enes.append(diff_ene)
            diff_epas.append(diff_epa)
            N_at = int(line.split()[4])
    # print(f'diff_enes: {diff_enes}')
    diff_ene_mean = np.mean(np.array(diff_enes[-N_configs:]))
    print(f'diff_ene_mean: {diff_ene_mean}')
    K_B = 8.61733326*10**(-5) # Boltzman constant in eV/K
    # K_B = 1 # Boltzman constant in eV/K
    # first_order_corr = diff_ene_mean
    if T >= 2500:
        # print(f'np.array(diff_enes) - diff_ene_mean: {np.array(diff_enes) - diff_ene_mean}')
        # print(f'(np.array(diff_enes) - diff_ene_mean)**2: {(np.array(diff_enes) - diff_ene_mean)**2}')
        print(f'np.mean((np.array(diff_enes) - diff_ene_mean)**2 = {np.mean((np.array(diff_enes[-N_configs:]) - diff_ene_mean)**2)}')
        print(f'1/(2*K_B*T)*np.mean((np.array(diff_enes) - diff_ene_mean)**2) = {1/(2*K_B*T)*np.mean((np.array(diff_enes[-N_configs:]) - diff_ene_mean)**2)}')
        first_order_corr = 1/N_at*(diff_ene_mean)
        second_order_corr = 1/N_at*1/(2*K_B*T)*np.mean((np.array(diff_enes[-N_configs:]) - diff_ene_mean)**2)
        total_corr = (first_order_corr - second_order_corr)
    elif T < 2500:
        first_order_corr = 1/N_at*(diff_ene_mean)
        second_order_corr = None
        total_corr = first_order_corr

    print(f'Corrections for free helmholtz energy from {set_name}')
    print(f'Temperature = {T} K')
    print(f'First order correction (Ab initio energy - MTP energy)= {first_order_corr} eV/atom')
    print(f'Second order correction = {second_order_corr} eV/atom')
    print(f'Total Correction = {total_corr} eV/atom')

    return total_corr, first_order_corr, second_order_corr


def get_pressure_correction_from_training_set(
                                 mlip_run_command = 'mpirun -np 1 mlp',
                                 pot_name = 'pot.almtp',
                                 set_name = 'set.cfg',
                                 N_configs = 50
                                 ):

    """
    This function calculates a correction to the pressure from existing training set for MTP.
    """
    get_errors_cmd = f'{mlip_run_command} check_errors {pot_name} {set_name} --log=mtp_errors'

    print(get_errors_cmd)
    os.system(get_errors_cmd)

    diff_stresses = []

    with open('mtp_errors.0','r') as f:
        lines = f.readlines()
        for line in lines:
            diff_stress = -float(line.split()[14])
            diff_stresses.append(diff_stress)

    diff_stress_mean = np.mean(np.array(diff_stresses[-N_configs:]))
    print(f'diff_stress_mean: {diff_stress_mean}')

    stress_corr = diff_stress_mean

    print(f'Pressure correction from {set_name}:')
    print(f'Abinitio stress - MTP stress = {stress_corr} GPa')


    return stress_corr


def get_ordered_structure(structure, specorder):

    """
    structure - ase Atoms object
    specorder - list with order of species
    """

    mapping = {val: i for i, val in enumerate(specorder)}
    # symbols = np.asarray([el.short_name for el in structure.getAtomTypes()])
    symbols = np.asarray(structure.get_chemical_symbols())
    # default_order = np.argsort(symbols)
    order = np.argsort(np.array([mapping[val] for val in symbols]))
    ordered_structure = Atoms(symbols[order], structure.get_positions()[order], cell=structure.get_cell(), pbc=True)

    return ordered_structure



def sscha_relaxation_mtp_supercell(structure='CONTCAR', calculator='LAMMPS', calculator_run_command='lmp', calculator_parameters=None, calculator_files=None, 
                        min_step_dyn=0.05, min_step_struc=0.05, kong_liu_ratio=0.5, gradi_op='all', meaningful_factor=0.2, precond_dyn=True, root_representation='normal',
                        N_CONFIGS=1, MAX_ITERATIONS=20, PRESSURE=0, TEMPERATURE=0, nq1 = 4, nq2 = 4, nq3 = 4):

    """
    Function for relaxation and calculation of the free energy within SSCHA for structure using large supercell using pretrained MTP

    Note: the 

    """

    ase_struct_raw = ase.io.read(structure) 
    ase_struct = get_ordered_structure(ase_struct_raw,specorder) 
    # ase_supercell = make_supercell(ase_struct, [[nq1,0,0],[0,nq2,0],[0,0,nq3]])


    dyn_supercell = get_initial_dyn_mtp(relaxed_structure, calculator, calculator_run_command, calculator_parameters, calculator_files, nq1, nq2, nq3)



    dyn_final, end_pop = sscha_relaxation_mtp(dyn_supercell, calculator, calculator_run_command, calculator_parameters, calculator_files,
                    min_step_dyn, min_step_struc, kong_liu_ratio, gradi_op, meaningful_factor, precond_dyn, root_representation,
                    N_CONFIGS, MAX_ITERATIONS, PRESSURE, TEMPERATURE,
                    specorder, pot_name, mlip_run_command, 
                    ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster, 
                    iteration_limit, energy_weight, force_weight, stress_weight, include_stress, 
                    train_on_every_ensemble, train_local_mtps, retrain, np_ab_initio,  start_pop)


    return













if __name__ == "__main__":


    if __MPI__:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if rank == 0:
            print(f'Program run in parallel mode on {size} procs!')
    else:
        print(f'Program run in sequential mode on 1 proc!')
        size = 1
        rank = 0            

    if rank == 0:

        print('!!! Start of the main program !!!')
        start_main = datetime.now()

    """ Reading the input structure """

    ase_struct_raw = ase.io.read(input_structure) 
    ase_struct = get_ordered_structure(ase_struct_raw,specorder) 
    ase_supercell = make_supercell(ase_struct, [[nq1,0,0],[0,nq2,0],[0,0,nq3]])


    """ Setting up cluster for ab initio calculations (works only for QE) """

    if use_hpc:
        ab_initio_cluster = setup_cluster(hpc_hostname,
                                          hpc_workdir,
                                          ab_initio_dir,
                                          hpc_partition_name,
                                          hpc_use_account,
                                          hpc_add_set_minus_x,
                                          hpc_binary,
                                          hpc_load_modules,
                                          hpc_n_cpu,
                                          hpc_n_nodes,
                                          hpc_n_pool,
                                          hpc_mpi_cmd,
                                          hpc_use_partition,
                                          hpc_use_qos,
                                          hpc_batch_size,
                                          hpc_job_number,
                                          hpc_time,
                                          hpc_terminal)
    else: 
        ab_initio_cluster = None



    """ MTP pretraining """

    if pretrain_on_rand_structures and (relax_zero_mtp or init_dyn_mtp or relax_sscha_mtp):
        if rank == 0:
            ### start pretraining mtp on randomly displaced structures
            print('Starting pretraining mtp on randomly displaced structures')
            start_pretrain = datetime.now()
            rand_structures = get_randomly_displaced_structures(ase_supercell, N_CONFIGS, rand_disp_magnitude)
            if min_distances != None:
                constrained_rand_structures = []
                for ase_structure in rand_structures:
                    cc_structure = CC.Structure.Structure()
                    cc_structure.generate_from_ase_atoms(ase_structure)
                    if are_ion_distances_good(cc_structure, min_distances):
                        constrained_rand_structures.append(ase_structure)
            else:
                constrained_rand_structures = rand_structures
            ase_structures_list_to_cfg(constrained_rand_structures,'preselected.cfg')
            os.system('touch set.cfg')
            train_mtp_on_cfg(specorder, mlip_run_command, pot_name, 
                ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster,
                iteration_limit, energy_weight, force_weight, stress_weight,include_stress,train_local_mtps,pop='rand',np_ab_initio = np_ab_initio)
            end_pretrain = datetime.now()
            delta_pretrain = end_pretrain - start_pretrain
            print(f'The MTP pretraining on randomly displaced structures was done within {delta_pretrain.total_seconds()} seconds')
            print('')        
            ### end pretraining mtp on randomly displaced structures
    elif pretrain_on_rand_structures and not (relax_zero_mtp or init_dyn_mtp or relax_sscha_mtp):
        if rank == 0:
            print('Pretraining mtp on randomly displaced structures will not done because the mtp seems not going to be used!')


    """ Zero-Kelvin relaxation """

    if relax_zero_chgnet:
        if rank == 0:
            print(f'Starting the structure relaxation with CHGNet using BFGS ASE calculator')
            start_relax = datetime.now()
            relaxed_structure = get_relaxed_structure_chgnet(input_structure, pressure = PRESSURE)
            end_relax = datetime.now()
            delta_relax = end_relax - start_relax
            print(f'The relaxation using CHGNet was done within {delta_relax.total_seconds()} seconds')
            print('') 
        # comm.Barrier()
    # if relax_zero_mtp and not relax_zero_chgnet:
    elif relax_zero_mtp:
        retrain = True
        if rank == 0:
            print(f'Starting the structure relaxation with MTP using {pot_name} with {calculator} calculator')
            start_relax = datetime.now()
            calculator_parameters_1 = Merge(calculator_parameters,calculator_parameters_relax)
            relaxed_structure = get_relaxed_structure_mtp(input_structure, calculator, calculator_run_command, calculator_parameters_1, calculator_files)
            end_relax = datetime.now()
            delta_relax = end_relax - start_relax
            print(f'The relaxation using {pot_name} was done within {delta_relax.total_seconds()} seconds')
            print('')
        retrain = False
    else:
        relaxed_structure = get_ordered_structure(ase.io.read(input_structure),specorder) 


    """ Calculation of the initial dynamical matrix """

    if rank == 0:
        print('Starting calculation of the initial dynamical matrix')
        start_init_dyn = datetime.now()

    if init_dyn_chgnet:
        dyn_init = get_initial_dyn_chgnet(relaxed_structure, nq1, nq2, nq3)
    elif init_dyn_mtp:
        # if rank == 0:
        dyn_init = get_initial_dyn_mtp(relaxed_structure, calculator, calculator_run_command, calculator_parameters, calculator_files)
    else:
        try:
            # TODO: can be error in the case when NQIRR > 1
            dyn_init = CC.Phonons.Phonons("dyn_init")
        except Exception as e:
            # if rank == 0:
            #     print(f"Reason: {e.message}")
            raise("Initial matrix is not calulated and can not be loaded!")

    if rank == 0:
        end_init_dyn = datetime.now()
        delta_init_dyn = end_init_dyn - start_init_dyn
        print(f'The initial dynamical matrix was calculated within {delta_init_dyn.total_seconds()} seconds')
        print('')


    """ Initial SSCHA relaxation with CHGNet using small (super)cells """

    if relax_sscha_chgnet:
        if rank == 0:
            print('STARTING SSCHA RELAXATION WITH CHGNet')
            start_sscha = datetime.now()

            dyn_final_chgnet, end_pop_chgnet = sscha_relaxation_chgnet(dyn_init, 
                                min_step_dyn, min_step_struc, kong_liu_ratio, gradi_op, meaningful_factor, precond_dyn, root_representation,
                                N_CONFIGS, MAX_ITERATIONS, PRESSURE, TEMPERATURE)

            end_sscha = datetime.now()
            delta_sscha = end_sscha - start_sscha
            print(f'The initial SSCHA relaxation with CHGNet was finished within {delta_sscha.total_seconds()} seconds')

            # Moving all dynamical matrices and structures used in SSCHA relaxation with CHGNet to the separate directory
            os.system('mkdir dyn_chgnet_small; mv dyn* dyn_chgnet_small')
            os.system('mkdir contcar_chgnet_small; mv *.CONTCAR dyn_chgnet_small')


    """ Final SSCHA relaxation with CHGNet using large supercells """

    if relax_sscha_chgnet and relax_sscha_supercell:

        if rank == 0:
            print('STARTING FINAL SSCHA RELAXATION WITH CHGNet')
            start_sscha = datetime.now()

            ase_struct_sscha = ase.io.read('CONTCAR') 
            ase_struct_sscha_ordered = get_ordered_structure(ase_struct_sscha,specorder) 

            # Calculating initial dynamical matrix with large supercell
            dyn_supercell = get_initial_dyn_mtp(ase_struct_sscha_ordered, calculator, calculator_run_command, calculator_parameters, calculator_files, nqs1, nqs2, nqs3)


            dyn_final_chgnet, end_pop_chgnet = sscha_relaxation_chgnet(dyn_supercell, 
                                min_step_dyn, min_step_struc, kong_liu_ratio, gradi_op, meaningful_factor, precond_dyn, root_representation,
                                N_CONFIGS, MAX_ITERATIONS, PRESSURE, TEMPERATURE)

            end_sscha = datetime.now()
            delta_sscha = end_sscha - start_sscha
            print(f'The final SSCHA relaxation with CHGNet was finished within {delta_sscha.total_seconds()} seconds')

            # Moving all dynamical matrices and structures used in SSCHA relaxation with CHGNet to the separate directory
            os.system('mkdir dyn_chgnet_small; mv dyn* dyn_chgnet_small')
            os.system('mkdir contcar_chgnet_large; mv *.CONTCAR dyn_chgnet_large')


    """ Initial SSCHA relaxation with MTP training on small (super)cells """

    if relax_sscha_mtp:
        if rank == 0:
            print('STARTING SSCHA RELAXATION WITH MTP')
            retrain = False # forcing not to use extrapolation control
            start_sscha = datetime.now()
            if relax_sscha_chgnet:
                dyn_new = dyn_final_chgnet      
                # start_pop = end_pop_chgnet            
                start_pop = 1
            elif not relax_sscha_chgnet:
                dyn_new = dyn_init      
                start_pop = 1
            # train MTP on initial ensemble only if MTP is not trained on every ensemble (otherwise training on initial ensemble)
            if train_on_initial_ensemble and not train_on_every_ensemble:
                ### start training mtp on initial ensemble
                print('Starting training MTP on initial ensemble')
                start_train = datetime.now()
                dyn_new.Symmetrize()
                dyn_new.ForcePositiveDefinite()
                ensemble_structures = dyn_new.ExtractRandomStructures(N_CONFIGS, T = TEMPERATURE)
                ase_structures_list_to_cfg(ensemble_structures,'preselected.cfg')
                os.system('touch set.cfg')
                train_mtp_on_cfg(specorder, mlip_run_command, pot_name, 
                    ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster,
                    iteration_limit, energy_weight, force_weight, stress_weight,include_stress,train_local_mtps,pop='init')
                end_train = datetime.now()
                delta_train = end_train - start_train
                print(f'The MTP training on initial ensemble was done within {delta_train.total_seconds()} seconds')
                print('')        

                ### end training mtp on initial ensemble

            dyn_final, end_pop = sscha_relaxation_mtp(dyn_new, calculator, calculator_run_command, calculator_parameters, calculator_files,
                            min_step_dyn, min_step_struc, kong_liu_ratio, gradi_op, meaningful_factor, precond_dyn, root_representation,
                            N_CONFIGS, MAX_ITERATIONS, PRESSURE, TEMPERATURE,
                            specorder, pot_name, mlip_run_command, 
                            ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster, 
                            iteration_limit, energy_weight, force_weight, stress_weight, include_stress, 
                            train_on_every_ensemble, train_local_mtps, retrain, np_ab_initio,  start_pop)

            end_sscha = datetime.now()
            delta_sscha = end_sscha - start_sscha
            print(f'The initial SSCHA relaxation with MTP training was done within {delta_sscha.total_seconds()} seconds')

            # Moving all dynamical matrices used in SSCHA relaxation with MTP training to the separate directory
            os.system('mkdir dyn_mtp_training; mv dyn* dyn_mtp_training')
            os.system('mkdir contcar_mtp_training; mv *.CONTCAR dyn_mtp_training')

    """ Final SSCHA relaxation with MTP using large supercells (without MTP training)"""

    if relax_sscha_mtp and relax_sscha_supercell:

        if rank == 0:
            print('STARTING FINAL SSCHA RELAXATION WITH MTP')
            start_sscha = datetime.now()

            # Switching all MTP training flags to False
            retrain = False
            train_on_initial_ensemble = False
            train_on_every_ensemble = False  
            train_local_mtps = False


            ase_struct_sscha = ase.io.read('CONTCAR') 
            ase_struct_sscha_ordered = get_ordered_structure(ase_struct_sscha,specorder) 

            # Calculating initial dynamical matrix with large supercell
            dyn_supercell = get_initial_dyn_mtp(ase_struct_sscha_ordered, calculator, calculator_run_command, calculator_parameters, calculator_files, nqs1, nqs2, nqs3)

            # Setting population count to 1 
            start_pop = 1

            dyn_final, end_pop = sscha_relaxation_mtp(dyn_supercell, calculator, calculator_run_command, calculator_parameters, calculator_files,
                            min_step_dyn, min_step_struc, kong_liu_ratio, gradi_op, meaningful_factor, precond_dyn, root_representation,
                            N_CONFIGS*10, MAX_ITERATIONS, PRESSURE, TEMPERATURE,
                            specorder, pot_name, mlip_run_command, 
                            ab_initio_calculator, ab_initio_parameters, ab_initio_run_command, ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster, 
                            iteration_limit, energy_weight, force_weight, stress_weight, include_stress, 
                            train_on_every_ensemble, train_local_mtps, retrain, np_ab_initio,  start_pop)



            end_sscha = datetime.now()
            delta_sscha = end_sscha - start_sscha
            print(f'The final SSCHA relaxation with MTP was done within {delta_sscha.total_seconds()} seconds')

            os.system('mkdir dyn_mtp_supercell; mv dyn* dyn_mtp_supercell')
            os.system('mkdir contcar_mtp_supercell; mv *.CONTCAR dyn_mtp_supercell')

    if rank == 0:
        end_main = datetime.now() 
        delta_main = end_main - start_main
        print(f'The whole program was finished within {delta_main.total_seconds()} seconds')
        print('!!! End of the main program !!!')



