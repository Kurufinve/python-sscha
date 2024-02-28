NPROC = 10

### SSCHA relaxation parameters

PRESSURE = 160 
TEMPERATURE = 300     
N_CONFIGS = 1000      # Number of configurations in each ensemble
MAX_ITERATIONS = 100  # Number of relaxation steps at which a new dymanical matrix is generated
RELAX_SIMPLE = True     # Perform the NVT relaxation keeping the unit cell volume unchanged
RELAX_NVT = True     # Perform the NVT relaxation allowing to relax lattice vectors keeping the unit cell volume unchanged
RELAX_NPT = True    # Perform the NPT relaxation allowing to relax the lattice vectors and volume to the desired pressure

### SSCHA minimization parameters

min_step_dyn = 0.05     # The minimization step on the dynamical matrix
min_step_struc = 0.05   # The minimization step on the structure
kong_liu_ratio = 0.5     # The parameter that estimates whether the ensemble is still good
gradi_op = "all" # Check the stopping condition on both gradients
meaningful_factor = 0.2 # How much small the gradient should be before I stop?
root_representation = 'normal' # can be 'normal', 'sqrt', or 'root4'
precond_dyn = True # should be false if 'sqrt' or 'root4' root_repesenattion is used
# root_representation = 'sqrt' # can be 'normal', 'sqrt', or 'root4'
# precond_dyn = False # should be false if 'sqrt' or 'root4' root_repesenattion is used

# MLIP parameters
!!! path_to_mlip = '/path/to/mlip-3/bin/mlp'
# mlip_run_command = f'mpirun -np {NPROC} {path_to_mlip}'
mlip_run_command = f'{path_to_mlip}'
pot_name  = 'pot.almtp'

pretrain_on_rand_structures = True # we do not pretrain on radnom structures because we using CHGNet for generation of initial dynamical matrix
train_on_initial_ensemble = False
train_on_every_ensemble = True # if True the MTP is trained every time the new ensemble is generated  
train_local_mtps = False # if True the new MTP is trained from scratch every time the training new set is generated (e.g. from ensemble)  
retrain = False # if we wish to retrain the MTP on structures produced with extrapolation control (better to make it False)
energy_weight = 1.0
force_weight = 1.0
stress_weight = 1.0
include_stress = True


input_structure = 'POSCAR' # name of file with input structure 
autoreplicate = True # if True, the number of replications in each direction calculated automativcally based on desired minimum number of atoms in the supercell Nfinal
# if autoreplicate:
    # it is recommended to use autoreplicate in the case of structure search with unit cells containing different number of atoms
Nfinal = 20 # the desired minimum number of atoms in the supercell which is used to generate ensembles 
# else:
    # it is recommended to set number of multiplications in each direction manually if we want to get dynamical matrix with specific resolution
    # nq1, nq2, and nq3 are rewrited by automatically calculated values if autoreplicate == True
nq1 = 1 # supercell multiplicity in the 1st direction
nq2 = 1 # supercell multiplicity in the 2nd direction
nq3 = 1 # supercell multiplicity in the 3rd direction
calculator = 'LAMMPS' # calculator for energies, forces, and stresses
specorder = ['La','H']

# calculator parameters

calculator_files =  [pot_name] # file with interatomic potentials 

if retrain == True:
    threshold = 10
    threshold_break = 50
    iteration_limit = 500
    calculator_parameters = {'pair_style': f'mlip load_from={calculator_files[0]} extrapolation_control=true extrapolation_control:threshold_break={threshold_break} extrapolation_control:threshold={threshold} extrapolation_control:add_grade_feature=true extrapolation_control:save_extrapolative_to=preselected.cfg',
                            'pair_coeff': ['* *'],}
elif retrain == False:
    calculator_parameters = {'pair_style': f'mlip load_from={calculator_files[0]} extrapolation_control=false',
                            'pair_coeff': ['* *'],}

calculator_parameters_relax = {'fix': [f'ensemble all box/relax iso {PRESSURE*10000} \n dump relax all custom 1 relax.dump id type xsu ysu zsu fx fy fz vx vy vz'],
                               'minimize': '0.0 0.0001 10000 10000000'}

!!! calculator_run_command = f'/path/to/interface-lammps-mlip-3/lmp_mpi'

relax_zero_chgnet = False # we do not relax structure at 0 K with CHGNet
relax_zero_mtp = False # we do not relax structure at 0 K with MTP

init_dyn_chgnet = False # we do not calculate initial dynamicl matrix with CHGNet
init_dyn_mtp = True # we calculate initial dynamical matrix with MTP 

relax_sscha_chgnet = False # we do not conduct SSCHA relaxation with CHGNet 
relax_sscha_mtp = True # we conduct SSCHA relaxation with MTP

relax_sscha_supercell = False # final sscha relaxation on large ensemble

correct_pressure = False
correct_free_energy = True

### Parameters for ab initio calculator that will be used if the additional training of MTP is necessary

np_ab_initio = NPROC # number of processors for ab_initio calculator
ab_initio_calculator = "QE" # we use quantum espresso calulator
ab_initio_parameters = {
                        "system":{
                            "ecutwfc" : 120, # The plane-wave wave-function cutoff
                            "occupations": "smearing",
                            "smearing": "mp",
                            "degauss": 0.005,
                            "nosym": True
                            # "ecutrho": 240, # The density wave-function cutoff,
                            },
                        "control":{
                            "pseudo_dir" : "../pseudos", # The directory of the pseudo potentials
                            "tprnfor" : True, # Print the forces
                            "tstress" : True, # Print the stress tensor
                            "prefix": "EFS4MTP",
                            "wf_collect": False,
                            "disk_io": "none",
                            "calculation" : "scf",
                            "restart_mode": "from_scratch"
                            },
                        "electrons":{
                            "electron_maxstep": 9999,
                            "conv_thr": 1e-8, # The convergence for the DFT self-consistency
                            },
                        }
ab_initio_pseudos = {"La": "La.pbe-hgh.UPF",
                     "H": "H.pbe-hgh.UPF"}
 

ab_initio_kresol = 0.03

!!! ab_initio_modules_load = "intel2021/mpi/2021.2.0 intel2021/compiler-rt/2021.2.0 intel2021/tbb/2021.2.0 intel2021/mkl/2021.2.0 quantumespresso/6.7; export UCX_TLS=ud,sm,self"

!!! ab_initio_run_command = f"module load {ab_initio_modules_load}; mpirun -np {np_ab_initio} pw.x -in ESP.pwi > ESP.pwo"

use_hpc = False  # if True, an hpc for ab initio calculations will be used, if False the ab initio calculation will run on the same computer as the main script

min_distances = {'La La': 2.4, 'La H': 1.4, 'H H': 0.7}
