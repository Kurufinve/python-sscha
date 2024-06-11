# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os
import warnings
import numpy as np
import time
import subprocess
import ase

import ase.calculators.calculator
import cellconstructor.calculators

from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.vasp import Vasp
from ase.units import Bohr, Ry
from sscha.Ensemble import *
from sscha.Relax import *
from cellconstructor.calculators import Espresso

# Modules from Ensemble.py
import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.Methods
import cellconstructor.Manipulate
import cellconstructor.Settings

import sscha.Parallel as Parallel
from sscha.Parallel import pprint as print
from sscha.Tools import NumpyEncoder

import json

import SCHAModules

# Modules from Relax.py
import difflib
import sscha, sscha.Ensemble, sscha.SchaMinimizer
import sscha.Optimizer
import sscha.Calculator
import sscha.Cluster
import sscha.Utilities as Utilities
import cellconstructor.symmetries
from sscha.aiida_ensemble import AiiDAEnsemble

import itertools
from scipy.spatial.distance import cdist

# Try to load the parallel library if any
try:
    from mpi4py import MPI
    __MPI__ = True
except:
    __MPI__ = False


class Ensemble_MTP(Ensemble):

    def __init__(self, dyn0, T0, supercell = None,
                 specorder = None, 
                 pot_name = 'fitted.mtp', 
                 mlip_run_command = 'mpirun -np 1 mlp', 
                 ab_initio_calculator = 'QE',
                 ab_initio_parameters = None,
                 ab_initio_run_command = None,
                 ab_initio_kresol = 0.25,
                 ab_initio_pseudos = None,
                 ab_initio_cluster = None,
                 iteration_limit = 500,
                 energy_weight = 1.0,
                 force_weight = 0.01,
                 stress_weight = 0.001,
                 include_stress = True,
                 retrain = True, 
                 np_ab_initio = 1,
                 np_mlp_train = 1,  
                 min_distances = None,               
                 **kwargs):

        self.specorder = specorder 
        self.mlip_run_command = mlip_run_command
        self.pot_name = pot_name 
        self.ab_initio_calculator = ab_initio_calculator
        self.ab_initio_parameters = ab_initio_parameters 
        self.ab_initio_run_command = ab_initio_run_command
        self.ab_initio_kresol = ab_initio_kresol 
        self.ab_initio_pseudos = ab_initio_pseudos 
        self.ab_initio_cluster = ab_initio_cluster
        self.iteration_limit = iteration_limit 
        self.energy_weight = energy_weight 
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.include_stress = include_stress
        self.retrain = retrain # IMPORTANT TAG!!! if we wish to retrain the MTP on structures produced with extrapolation control
        self.np_ab_initio = np_ab_initio
        self.np_mlp_train = np_mlp_train
        self.min_distances = min_distances
        # super().__init__(dyn0, T0, supercell, **kwargs)

        # Setup any other keyword given in input (raising the error if not already defined)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])

        super().__init__(dyn0, T0, supercell)


    def generate(self, N, evenodd = True, project_on_modes = None, sobol = False, sobol_scramble = False, sobol_scatter = 0.0):
        """
        GENERATE THE ENSEMBLE
        =====================

        This subroutine generates the ensemble from dyn0 and T0 setted when this
        class is created.
        You still need to generate the forces for the configurations.

        Parameters
        ----------
            N : int
                The number of random configurations to be extracted
            evenodd : bool, optional
                If true for each configuration also the opposite is extracted
            project_on_modes : ndarray(size=(3*nat_sc, nproj)), optional
                If different from None the displacements are projected on the
                given modes.
            sobol : bool, optional (Default = False)
                 Defines if the calculation uses random Gaussian generator or Sobol Gaussian generator.
            sobol_scramble : bool, optional (Default = False)
                Set the optional scrambling of the generated numbers taken from the Sobol sequence.
            sobol_scatter : real (0.0 to 1) (Deafault = 0.0)
                Set the scatter parameter to displace the Sobol positions randommly.

        """

        if evenodd and (N % 2 != 0):
            raise ValueError("Error, evenodd allowed only with an even number of random structures")

        self.N = N
        Nat_sc = np.prod(self.supercell) * self.dyn_0.structure.N_atoms
        self.structures = []
        #super_dyn = self.dyn_0.GenerateSupercellDyn(self.supercell)
        super_struct = self.dyn_0.structure.generate_supercell(self.dyn_0.GetSupercell())

        structures = []

        if evenodd:
            structs = self.dyn_0.ExtractRandomStructures(N // 2, self.T0, project_on_vectors = project_on_modes, lock_low_w = self.ignore_small_w, sobol = sobol, sobol_scramble = sobol_scramble, sobol_scatter = sobol_scatter)  # normal Sobol generator****Diegom_test****



            for i, s in enumerate(structs):
                # structures.append(s)
                new_s = s.copy()
                # Get the opposite displacement structure
                new_s.coords = super_struct.coords - new_s.get_displacement(super_struct)
                # structures.append(new_s)
                if self.min_distances != None:
                    # adding the structure and its opposite counterpart only if both satisfy the min_distance constraints 
                    if are_ion_distances_good(s, self.min_distances) and are_ion_distances_good(new_s, self.min_distances):
                        structures.append(s)
                        structures.append(new_s)
                else:
                    structures.append(s)
                    structures.append(new_s)

            constrained_structures = structures

        else:
            structures = self.dyn_0.ExtractRandomStructures(N, self.T0, project_on_vectors = project_on_modes, lock_low_w = self.ignore_small_w, sobol = sobol, sobol_scramble = sobol_scramble, sobol_scatter = sobol_scatter)  # normal Sobol generator****Diegom_test****

            # filter structures by min_distance constraints if they are specified
            if self.min_distances != None:
                # creating list for storing only the structures that satisfy min_distance constraints 
                constrained_structures = []
                for structure in structures:
                    if are_ion_distances_good(structure, self.min_distances):
                        constrained_structures.append(structure)
            else:
                constrained_structures = structures

        # Enforce all the processors to share the same structures
        constrained_structures = CC.Settings.broadcast(constrained_structures)
        if self.min_distances != None:
            print(f'{len(structures)} are generated, and {len(constrained_structures)} are added to ensemble based on the user min_distances constraints')
            print(f'min_distances constraints are {self.min_distances}')

        self.init_from_structures(constrained_structures)



    def compute_ensemble(self, calculator, compute_stress = True, stress_numerical = False,
                         cluster = None, verbose = True):
        """
        GET ENERGY AND FORCES
        =====================

        This is the generic function to compute forces and stresses.
        It can be used both with clusters, and with simple ase calculators

        Paramters
        ---------
            calculator:
                The ase calculator
            compute_stress: bool
                If true compute the stress
            stress_numerical : bool
                Compute the stress tensor with finite difference,
                this is not possible with clusters
            cluster: Cluster, optional
                The cluster in which to send the calculation.
                If None the calculation is performed on the same computer of
                the sscha code.
        """


        # Check if the calculator is a cluster
        is_cluster = False
        if not cluster is None:
            is_cluster = True

        # Check consistency
        if stress_numerical and is_cluster:
            raise ValueError("Error, stress_numerical is not implemented with clusters")

        # Check if not all the calculation needs to be done
        n_calcs = np.sum( self.force_computed.astype(int))
        computing_ensemble = self

        if compute_stress:
            self.has_stress = True

        # Check wheter compute the whole ensemble, or just a small part
        should_i_merge = False
        if n_calcs != self.N:
            should_i_merge = True
            computing_ensemble = self.get_noncomputed()
            self.remove_noncomputed()

        # Remove the structures that do not satisfy minDist constraints 


        # Computing energy and forces for structures in the ensemble
        if is_cluster:
            print("BEFORE COMPUTING:", self.all_properties)
            cluster.compute_ensemble(computing_ensemble, calculator, compute_stress)

        else:
            computing_ensemble.get_energy_forces(calculator, compute_stress, stress_numerical, verbose = verbose)

        print("CE BEFORE MERGE:", len(self.force_computed))

        if should_i_merge:
            # Remove the noncomputed ensemble from here, and merge
            self.merge(computing_ensemble)
        print("CE AFTER MERGE:", len(self.force_computed))

        print('ENSEMBLE ALL PROPERTIES:', self.all_properties)

    def get_mindist_constrained(self):
        """
        Get another ensemble with only the configurations that satisfy the mindist constraints.
        This may be used to avoid the erroneus calculations with overlapping PAW spheres (in the ab initio case) 
        or within untrained domain (in the case of actively learned MTPs)
        """


        non_mask = ~self.force_computed

        return self.split(non_mask)

    def get_energy_forces(self, ase_calculator, compute_stress = True, stress_numerical = False, skip_computed = False, verbose = False):
        """
        GET ENERGY AND FORCES FOR THE CURRENT ENSEMBLE
        ==============================================

        This subroutine uses the ase calculator to compute the abinitio energies and forces
        of the self ensemble.
        This subroutine requires to have ASE installed and properly configured to
        interface with your favourite ab-initio software.


        Parameters
        ----------
            ase_calculator : ase.calculator
                The ASE interface to the calculator to run the calculation.
                also a CellConstructor calculator is accepted
            compute_stress : bool
                If true, the stress is requested from the ASE calculator. Be shure
                that the calculator you provide supports stress calculation
            stress_numerical : bool
                If the calculator does not support stress, it can be computed numerically
                by doing finite differences.
            skip_computed : bool
                If true the configurations already computed will be skipped.
                Usefull if the calculation crashed for some reason.

        """

        # Setup the calculator for each structure
        parallel = False
        print("Force computed shape:", len(self.force_computed))
        if __MPI__:
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()

            if size > 1:
                parallel = True
                # Broad cast to all the structures
                structures = comm.bcast(self.structures, root = 0)
                nat3 = comm.bcast(self.current_dyn.structure.N_atoms* 3* np.prod(self.supercell), root = 0)
                N_rand = comm.bcast(self.N, root=0)


                #if not Parallel.am_i_the_master():
                #    self.structures = structures
                #    self.init_from_structures(structures) # Enforce all the ensembles to have the same structures

                # Setup the label of the calculator
                #ase_calculator = comm.bcast(ase_calculator, root = 0)   # This broadcasting seems causing some issues on some fortran codes called by python (which may interact with MPI)
                ase_calculator.set_label("esp_%d" % rank) # Avoid overwriting the same file

                compute_stress = comm.bcast(compute_stress, root = 0)


                # Check if the parallelization is correct
                if N_rand % size != 0:
                    raise ValueError("Error, for paralelization the ensemble dimension must be a multiple of the processors")

        if not parallel:
            size = 1
            rank = 0
            structures = self.structures
            nat3 = self.current_dyn.structure.N_atoms* 3 * np.prod(self.supercell)
            N_rand = self.N

        # Only for the master

        # Prepare the energy, forces and stress array
        # TODO: Correctly setup the number of energies here


        # If an MPI istance is running, split the calculation
        tot_configs = N_rand // size
        remainer = N_rand % size

        if rank < remainer:
            start = rank * (tot_configs + 1)
            stop = start + tot_configs + 1
        else:
            start = rank * tot_configs + remainer
            stop = start + tot_configs

        num_confs = stop - start

        energies = np.zeros( num_confs, dtype = np.float64)
        forces = np.zeros( ( num_confs) * nat3 , dtype = np.float64)
        if compute_stress:
            stress = np.zeros( num_confs * 9, dtype = np.float64)

        if rank == 0:
            total_forces = np.zeros( N_rand * nat3, dtype = np.float64)
            total_stress = np.zeros( N_rand * 9, dtype = np.float64)
        else:
            total_forces = np.empty( N_rand * nat3, dtype = np.float64)
            total_stress = np.empty( N_rand * 9, dtype = np.float64)

        i0 = 0
        for i in range(start, stop):

            # Avoid performing this calculation if already done
            if skip_computed:
                if self.force_computed[i]:
                    if compute_stress:
                        if self.stress_computed[i]:
                            continue
                    else:
                        continue


            struct = structures[i]
            #atms = struct.get_ase_atoms()

            # Setup the ASE calculator
            #atms.set_calculator(ase_calculator)


            # Print the status
            if Parallel.am_i_the_master() and verbose:
                print ("Computing configuration %d out of %d (nat = %d)" % (i+1, stop, struct.N_atoms))
                sys.stdout.flush()

            # Avoid for errors
            run = True
            count_fails = 0
            retrain_count_fails = 0
            while run:
                try:
                    results = CC.calculators.get_results(ase_calculator, struct, get_stress = compute_stress)
                    energy = results["energy"] / Rydberg # eV => Ry
                    forces_ = results["forces"] / Rydberg

                    if compute_stress:
                        stress[9*i0 : 9*i0 + 9] = -results["stress"].reshape(9)* Bohr**3 / Rydberg
                        #energy = atms.get_total_energy() / Rydberg # eV => Ry
                        # Get energy, forces (and stress)
                        #energy = atms.get_total_energy() / Rydberg # eV => Ry
                        #forces_ = atms.get_forces() / Rydberg # eV / A => Ry / A
                        #if compute_stress:
                    #    if not stress_numerical:
                    #        stress[9*i0 : 9*i0 + 9] = -atms.get_stress(False).reshape(9) * Bohr**3 / Rydberg  # ev/A^3 => Ry/bohr
                    #    else:
                    #        stress[9*i0 : 9*i0 + 9] = -ase_calculator.calculate_numerical_stress(atms, voigt = False).ravel()* Bohr**3 / Rydberg

                    # Copy into the ensemble array
                    energies[i0] = energy
                    forces[nat3*i0 : nat3*i0 + nat3] = forces_.reshape( nat3 )
                    run = False
                except Exception as e:
                    print(f'Error in the ASE calculator: {e}')
                    if self.retrain:
                        print ("Retraining MTP on job %d" % i)
                        os.system('touch set.cfg')
                        try:
                            train_mtp_on_cfg(self.specorder,self.mlip_run_command, self.pot_name, 
                                self.ab_initio_calculator, self.ab_initio_parameters, self.ab_initio_run_command, self.ab_initio_kresol, self.ab_initio_pseudos, self.ab_initio_cluster, 
                                self.iteration_limit, self.energy_weight, self.force_weight, self.stress_weight, self.include_stress, 
                                train_local_mtps = False, pop = 'ensemble_retrain', np_ab_initio = self.np_ab_initio)
                            retrain_count_fails = 0
                        except:
                            retrain_count_fails += 1
                            if retrain_count_fails >= 5:
                                run = False
                                struct.save_scf("error_struct.scf")
                                sys.stderr.write("Error in the retrain MTP for more than 5 times\n     while computing 'error_struct.scf'")
                                # raise
                                continue

                    else:
                        print ("Rerun the job %d" % i)
                        count_fails += 1
                        if count_fails >= 5:
                            run = False
                            struct.save_scf("error_struct.scf")
                            sys.stderr.write("Error in the ASE calculator for more than 5 times\n     while computing 'error_struct.scf'")
                            # raise
                            continue



            i0 += 1


        self.remove_noncomputed()


        # Collect all togheter

        if parallel:
            comm.Allgather([energies, MPI.DOUBLE], [self.energies, MPI.DOUBLE])
            comm.Allgather([forces, MPI.DOUBLE], [total_forces, MPI.DOUBLE])

            if compute_stress:
                comm.Allgather([stress, MPI.DOUBLE], [total_stress, MPI.DOUBLE])


            #self.update_weights(self.current_dyn, self.current_T)
            CC.Settings.barrier()


        else:
            self.energies = energies
            total_forces = forces
            if compute_stress:
                total_stress = stress

        # Reshape the arrays
        self.forces[:, :, :] = np.reshape(total_forces, (N_rand, self.current_dyn.structure.N_atoms*np.prod(self.supercell), 3), order = "C")
        self.force_computed[:] = True
        print("Force computed shape:", len(self.force_computed))

        if compute_stress:
            self.stresses[:,:,:] = np.reshape(total_stress, (N_rand, 3, 3), order = "C")
            self.has_stress = True
            self.stress_computed[:] = True
        else:
            self.has_stress = False















class SSCHA_MTP(SSCHA):
    def __init__(self, minimizer = None, ase_calculator=None, N_configs=1, max_pop = 20,
                 save_ensemble = False, cluster = None, 
                 specorder = None,
                 pot_name = 'fitted.mtp', 
                 mlip_run_command = 'mpirun -np 1 mlp', 
                 ab_initio_calculator = 'QE',
                 ab_initio_parameters = None,
                 ab_initio_run_command = None,
                 ab_initio_kresol = 0.25,
                 ab_initio_pseudos = None,
                 ab_initio_cluster = None,
                 iteration_limit = 500,
                 energy_weight = 1.0,
                 force_weight = 0.01,
                 stress_weight = 0.001,
                 include_stress = False,
                 train_on_every_ensemble = False,
                 train_local_mtps = False,
                 retrain = False,
                 np_ab_initio = 1,
                 np_mlp_train = 1,
                 **kwargs):

        # super().__init__(minimizer=minimizer, ase_calculator=ase_calculator, N_configs=N_configs, max_pop=max_pop,
        #          save_ensemble = save_ensemble, cluster = cluster)

        # self.minimizer = minimizer
        # self.ase_calculator = ase_calculator
        # self.N_configs = N_configs
        # self.max_pop = max_pop
        # self.save_ensemble = save_ensemble
        # self.cluster = cluster

        self.specorder = specorder
        self.mlip_run_command = mlip_run_command
        self.pot_name = pot_name 
        self.ab_initio_calculator = ab_initio_calculator
        self.ab_initio_parameters = ab_initio_parameters 
        self.ab_initio_run_command = ab_initio_run_command
        self.ab_initio_kresol = ab_initio_kresol 
        self.ab_initio_pseudos = ab_initio_pseudos 
        self.ab_initio_cluster = ab_initio_cluster
        self.iteration_limit = iteration_limit 
        self.energy_weight = energy_weight 
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.include_stress = include_stress
        self.train_on_every_ensemble = train_on_every_ensemble # if True the MTP is trained every time the new ensemble is generated  
        self.train_local_mtps = train_local_mtps # if True the new MTP is trained from scratch every time the training new set is generated (e.g. from ensemble)  
        self.retrain = retrain # IMPORTANT TAG!!! if we wish to retrain the MTP on structures produced with extrapolation control
        self.np_ab_initio = np_ab_initio
        self.np_mlp_train = np_mlp_train
        # super().__init__(minimizer=self.minimizer, ase_calculator=self.ase_calculator, N_configs=self.N_configs, max_pop=self.max_pop,
        #          save_ensemble = self.save_ensemble, cluster = self.cluster)

        super().__init__(minimizer, ase_calculator, None, N_configs, max_pop,
                 save_ensemble, cluster)


    def relax(self, restart_from_ens = False, get_stress = False,
              ensemble_loc = None, start_pop = None, sobol = False,
              sobol_scramble = False, sobol_scatter = 0.0):#,
            #   train_on_every_ensemble = False,
            #   train_local_mtps = False,
            #   retrain = False):
        """
        COSTANT VOLUME RELAX
        ====================

        This function performs the costant volume SCHA relaxation, by submitting several populations
        until the minimization converges (or the maximum number of population is reached)

        Parameters
        ----------
            restart_from_ens : bool, optional
                If True the ensemble is used to start the first population, without recomputing
                energies and forces. If False (default) the first ensemble is overwritten with
                a new one, and the minimization starts.
            get_stress : bool, optional
                If true the stress tensor is calculated. This may increase the computational
                cost, as it will be computed for each ab-initio configuration (it may be not available
                with some ase calculator)
            ensemble_loc : string
                Where the ensemble of each population is saved on the disk. If none, it will
                use the content of self.data_dir. It is just a way to override the variable self.data_dir
            start_pop : int, optional
                The starting index for the population, used only for saving the ensemble and the dynamical
                matrix. If None, the content of self.start_pop will be used.
            sobol : bool, optional (Default = False)
                 Defines if the calculation uses random Gaussian generator or Sobol Gaussian generator.
            sobol_scramble : bool, optional (Default = False)
                Set the optional scrambling of the generated numbers taken from the Sobol sequence.
            sobol_scatter : real (0.0 to 1) (Deafault = 0.0)
                Set the scatter parameter to displace the Sobol positions randommly.

        Returns
        -------
            status : bool
                True if the minimization converged, False if the maximum number of
                populations has been reached.
        """

        if ensemble_loc is None:
            ensemble_loc = self.data_dir

        if (not ensemble_loc) and self.save_ensemble:
            ERR_MSG = """
Error, you must specify where to save the ensembles.
       this can be done either passing ensemble_loc = "path/to/dir"
       for the ensemble, or by setting the data_dir attribute of this object.
"""
            raise IOError(ERR_MSG)

        if self.save_ensemble:
            if not os.path.exists(ensemble_loc):
                os.makedirs(ensemble_loc)
            else:
                if not os.path.isdir(ensemble_loc):
                    ERR_MSG = """
Error, the specified location to save the ensemble:
       '{}'
       already exists and it is not a directory.
""".format(ensemble_loc)
                    raise IOError(ERR_MSG)


        if start_pop is None:
            start_pop = self.start_pop

        pop = start_pop

        running = True
        while running:
            # Generate the ensemble
            self.minim.ensemble.dyn_0 = self.minim.dyn.Copy()

            if pop != start_pop or not restart_from_ens:
                self.minim.ensemble.generate(self.N_configs, sobol = sobol, sobol_scramble = sobol_scramble, sobol_scatter = sobol_scatter)
                
                # Train MTP on generated ensemble
                if self.train_on_every_ensemble:
                    ensemble_structures = self.minim.ensemble.structures
                    ase_structures_list_to_cfg(ensemble_structures,'preselected.cfg',self.specorder)
                    os.system('touch set.cfg')
                    train_mtp_on_cfg(self.specorder,self.mlip_run_command, self.pot_name, 
                        self.ab_initio_calculator, self.ab_initio_parameters, self.ab_initio_run_command, self.ab_initio_kresol, self.ab_initio_pseudos, self.ab_initio_cluster, 
                        self.iteration_limit, self.energy_weight, self.force_weight, self.stress_weight, self.include_stress, self.train_local_mtps, pop, self.np_ab_initio)
                # elif not self.train_on_every_ensemble:
                #     if pop == start_pop:
                #         ensemble_structures = self.minim.ensemble.structures
                #         ase_structures_list_to_cfg(ensemble_structures,'preselected.cfg',self.specorder)
                #         os.system('touch set.cfg')
                #         train_mtp_on_cfg(self.specorder,self.mlip_run_command, self.pot_name, 
                #             self.ab_initio_calculator, self.ab_initio_parameters, self.ab_initio_run_command, self.ab_initio_kresol, self.ab_initio_pseudos, self.ab_initio_cluster, 
                #             self.iteration_limit, self.energy_weight, self.force_weight, self.stress_weight, self.include_stress, self.train_local_mtps,pop, self.np_ab_initio)
                #     else:
                #         pass

                # Compute energies and forces
                self.minim.ensemble.compute_ensemble(self.calc, get_stress,
                                                 cluster = self.cluster)
                #self.minim.ensemble.get_energy_forces(self.calc, get_stress)

                if ensemble_loc is not None and self.save_ensemble:
                    self.minim.ensemble.save_bin(ensemble_loc, pop)

            self.minim.population = pop
            self.minim.init(delete_previous_data = False)

            self.minim.run(custom_function_pre = self.__cfpre__,
                           custom_function_post = self.__cfpost__,
                           custom_function_gradient = self.__cfg__)


            self.minim.finalize()

            # Perform the symmetrization
            print ("Checking the symmetries of the dynamical matrix:")
            qe_sym = CC.symmetries.QE_Symmetry(self.minim.dyn.structure)
            qe_sym.SetupQPoint(verbose = True)

            print ("Forcing the symmetries in the dynamical matrix.")
            fcq = np.array(self.minim.dyn.dynmats, dtype = np.complex128)
            qe_sym.SymmetrizeFCQ(fcq, self.minim.dyn.q_stars, asr = "custom")
            for iq,q in enumerate(self.minim.dyn.q_tot):
                self.minim.dyn.dynmats[iq] = fcq[iq, :, :]

            # Save the dynamical matrix
            if self.save_ensemble:
                self.minim.dyn.save_qe("dyn_pop%d_" % pop)

            # Save the structure in CONTCAR (vasp) format for convenience
            current_ase_structure = self.minim.dyn.structure.get_ase_atoms()
            ase.io.write(f'dyn_pop{pop}.CONTCAR', current_ase_structure, "vasp", direct=True, label = f'Current structure after minimization during SSCHA relaxation corresponding to dyn_pop{pop}')
            ase.io.write('CONTCAR', current_ase_structure, "vasp", direct=True, label = f'Current structure after minimization during SSCHA relaxation corresponding to dyn_pop{pop}')


            # Check if it is converged
            running = not self.minim.is_converged()
            pop += 1


            if pop > self.max_pop:
                running = False


        self.start_pop = pop
        print('Population = ',pop) #**** Diegom_test ****
        return self.minim.is_converged()


    def vc_relax(self, target_press = 0, static_bulk_modulus = 100,
                 restart_from_ens = False,
                 ensemble_loc = None, start_pop = None, stress_numerical = False,
                 cell_relax_algorithm = "sd", fix_volume = False, sobol = False,
                 sobol_scramble = False, sobol_scatter = 0.0):#, 
                #  train_on_every_ensemble = False,
                #  train_local_mtps = False,
                #  retrain = False):
        """
        VARIABLE CELL RELAX
        ====================

        This function performs a variable cell SCHA relaxation at constant pressure,
        It is similar to the relax calculation, but the unit cell is updated according
        to the anharmonic stress tensor at each new population.

        By default, all the degrees of freedom compatible with the symmetry group are relaxed in the cell.
        You can constrain the cell to keep the same shape by setting fix_cell_shape = True.


        NOTE:
            remember to setup the stress_offset variable of the SCHA_Minimizer,
            because in ab-initio calculation the stress tensor converges porly with the cutoff,
            but stress tensor differences converges much quicker. Therefore, setup the
            stress tensor difference between a single very high-cutoff calculation and a
            single low-cutoff (the one you use), this difference will be added at the final
            stress tensor to get a better estimation of the true stress.


        Parameters
        ----------
            target_press : float, optional
                The target pressure of the minimization (in GPa). The minimization is stopped if the
                target pressure is the stress tensor is the identity matrix multiplied by the
                target pressure, with a tollerance equal to the stochastic noise. By default
                it is 0 (ambient pressure)
            static_bulk_modulus : float (default 100), or (9x9) matrix or string, optional
                The static bulk modulus, expressed in GPa. It is used to initialize the
                hessian matrix on the BFGS cell relaxation, to guess the volume deformation caused
                by the anharmonic stress tensor in the first steps. By default is 100 GPa (higher value
                are safer, since they means a lower change in the cell shape).
                It can be also the whole non isotropic matrix. If you specify a string, it
                can be both:
                    - "recalc" : the static bulk modulus is recomputed with finite differences after
                        each step
                    - "bfgs" : the bfgs algorithm is used to infer the Hessian from previous calculations.
            restart_from_ens : bool, optional
                If True the ensemble is used to start the first population, without recomputing
                energies and forces. If False (default) the first ensemble is overwritten with
                a new one, and the minimization starts.
            ensemble_loc : string
                Where the ensemble of each population is saved on the disk. You can specify None
                if you do not want to save the ensemble (useful to avoid disk I/O for force fields)
            start_pop : int, optional
                The starting index for the population, used only for saving the ensemble and the dynamical
                matrix.
            stress_numerical : bool
                If True the stress is computed by finite difference (usefull for calculators that
                does not support it by default)
            cell_relax_algorithm : string
                This identifies the stress algorithm. It can be both sd (steepest-descent),
                cg (conjugate-gradient) or bfgs (Quasi-newton).
                The most robust one is SD. Do not change if you are not sure what you are doing.
            fix_volume : bool, optional
                If true (default False) the volume is fixed, therefore only the cell shape is relaxed.
            sobol : bool, optional (Default = False)
                 Defines if the calculation uses random Gaussian generator or Sobol Gaussian generator.
            sobol_scramble : bool, optional (Default = False)
                Set the optional scrambling of the generated numbers taken from the Sobol sequence.
            sobol_scatter : real (0.0 to 1) (Deafault = 0.0)
                Set the scatter parameter to displace the Sobol positions randommly.

        Returns
        -------
            status : bool
                True if the minimization converged, False if the maximum number of
                populations has been reached.
        """

        # Prepare the saving directory
        if ensemble_loc is None:
            ensemble_loc = self.data_dir

        if (not ensemble_loc) and self.save_ensemble:
            ERR_MSG = """
Error, you must specify where to save the ensembles.
       this can be done either passing ensemble_loc = "path/to/dir"
       for the ensemble, or by setting the data_dir attribute of this object.
"""
            raise IOError(ERR_MSG)

        if self.save_ensemble:
            if not os.path.exists(ensemble_loc):
                os.makedirs(ensemble_loc)
            else:
                if not os.path.isdir(ensemble_loc):
                    ERR_MSG = """
Error, the specified location to save the ensemble:
       '{}'
       already exists and it is not a directory.
""".format(ensemble_loc)
                    raise IOError(ERR_MSG)



        # Rescale the target pressure in eV / A^3
        target_press_evA3 = target_press / sscha.SchaMinimizer.__evA3_to_GPa__
        I = np.eye(3, dtype = np.float64)

        SUPPORTED_ALGORITHMS = ["sd", "cg", "bfgs"]
        if not cell_relax_algorithm in SUPPORTED_ALGORITHMS:
            raise ValueError("Error, cell_relax_algorithm %s not supported." %  cell_relax_algorithm)

        # Read the bulk modulus
        kind_minimizer = "SD"
        if type(static_bulk_modulus) == type(""):
            if static_bulk_modulus == "recalc":
                kind_minimizer = "RPSD"
            elif static_bulk_modulus == "none":
                kind_minimizer = "SD"
                static_bulk_modulus = 100
            elif static_bulk_modulus == "bfgs":
                static_bulk_modulus = 100
                kind_minimizer = "BFGS"
            else:
                raise ValueError("Error, value '%s' not supported for bulk modulus." % static_bulk_modulus)
        elif len(np.shape(static_bulk_modulus)) == 0:
            kind_minimizer = cell_relax_algorithm.upper()
        elif len(np.shape(static_bulk_modulus)) == 2:
            kind_minimizer = "PSD"
        else:
            raise ValueError("Error, the given value not supported as a bulk modulus.")




        if static_bulk_modulus != "recalc":
            # Rescale the static bulk modulus in eV / A^3
            static_bulk_modulus /= sscha.SchaMinimizer.__evA3_to_GPa__

        # initilaize the cell minimizer
        #BFGS = sscha.Optimizer.BFGS_UC(self.minim.dyn.structure.unit_cell, static_bulk_modulus)
        if kind_minimizer in ["SD", "CG"] :
            BFGS = sscha.Optimizer.UC_OPTIMIZER(self.minim.dyn.structure.unit_cell)
            BFGS.alpha = 1 / (3 * static_bulk_modulus * self.minim.dyn.structure.get_volume())
            BFGS.algorithm = kind_minimizer.lower()
        elif kind_minimizer == "PSD":
            BFGS = sscha.Optimizer.SD_PREC_UC(self.minim.dyn.structure.unit_cell, static_bulk_modulus)
        elif kind_minimizer == "BFGS":
            BFGS = sscha.Optimizer.BFGS_UC(self.minim.dyn.structure.unit_cell, static_bulk_modulus)

        # Initialize the bulk modulus
        # The gradient (stress) is in eV/A^3, we have the cell in Angstrom so the Hessian must be
        # in eV / A^6
        if start_pop is not None:
            pop = start_pop
        else:
            pop = self.start_pop

        running = True
        while running:
            # Compute the static bulk modulus if required
            if kind_minimizer == "RPSD":
                # Compute the static bulk modulus
                sbm = GetStaticBulkModulus(self.minim.dyn.structure, self.calc)
                print ("BM:")
                print (sbm)
                BFGS = sscha.Optimizer.SD_PREC_UC(self.minim.dyn.structure.unit_cell, sbm)

            # Generate the ensemble
            self.minim.ensemble.dyn_0 = self.minim.dyn.Copy()
            if pop != start_pop or not restart_from_ens:
                self.minim.ensemble.generate(self.N_configs, sobol=sobol, sobol_scramble = sobol_scramble, sobol_scatter = sobol_scatter)

                # Save also the generation
                #if ensemble_loc is not None and self.save_ensemble:
                #    self.minim.ensemble.save_bin(ensemble_loc, pop)
                # Train MTP on generated ensemble
                if self.train_on_every_ensemble:
                    ensemble_structures = self.minim.ensemble.structures
                    ase_structures_list_to_cfg(ensemble_structures,'preselected.cfg',self.specorder)
                    os.system('touch set.cfg')
                    train_mtp_on_cfg(self.specorder,self.mlip_run_command, self.pot_name, 
                        self.ab_initio_calculator, self.ab_initio_parameters, self.ab_initio_run_command, self.ab_initio_kresol, self.ab_initio_pseudos, self.ab_initio_cluster, 
                        self.iteration_limit, self.energy_weight, self.force_weight, self.stress_weight, self.include_stress, self.train_local_mtps,pop, self.np_ab_initio)
                # elif not self.train_on_every_ensemble:
                #     if pop == start_pop:
                #         ensemble_structures = self.minim.ensemble.structures
                #         ase_structures_list_to_cfg(ensemble_structures,'preselected.cfg',self.specorder)
                #         os.system('touch set.cfg')
                #         train_mtp_on_cfg(self.specorder, self.mlip_run_command, self.pot_name, 
                #             self.ab_initio_calculator, self.ab_initio_parameters, self.ab_initio_run_command, self.ab_initio_kresol, self.ab_initio_pseudos, self.ab_initio_cluster, 
                #             self.iteration_limit, self.energy_weight, self.force_weight, self.stress_weight, self.include_stress, self.train_local_mtps,pop, self.np_ab_initio)
                #     else:
                #         pass

                # Compute energies and forces
                self.minim.ensemble.compute_ensemble(self.calc, True, stress_numerical,
                                                 cluster = self.cluster)
                #self.minim.ensemble.get_energy_forces(self.calc, True, stress_numerical = stress_numerical)

                print("RELAX force length:", len(self.minim.ensemble.force_computed))
                
                if ensemble_loc is not None and self.save_ensemble:
                    self.minim.ensemble.save_bin(ensemble_loc, pop)
                print("RELAX force length:", len(self.minim.ensemble.force_computed))

            self.minim.population = pop
            self.minim.init(delete_previous_data = False)

            print("RELAX force length:", len(self.minim.ensemble.force_computed))
            self.minim.run(custom_function_pre = self.__cfpre__,
                           custom_function_post = self.__cfpost__,
                           custom_function_gradient = self.__cfg__)
            


            self.minim.finalize()

            # Get the stress tensor [ev/A^3]
            stress_tensor, stress_err = self.minim.get_stress_tensor()
            stress_tensor *= sscha.SchaMinimizer.__RyBohr3_to_evA3__
            stress_err *=  sscha.SchaMinimizer.__RyBohr3_to_evA3__

            # Get the pressure
            Press = np.trace(stress_tensor) / 3

            # Get the volume
            Vol = self.minim.dyn.structure.get_volume()

            # Get the Helmoltz-Gibbs free energy
            helmoltz = self.minim.get_free_energy() * sscha.SchaMinimizer.__RyToev__
            gibbs = helmoltz + target_press_evA3 * Vol - self.minim.eq_energy

            # Prepare a mark to underline which quantity is actually minimized by the
            # Variable relaxation algorithm if the helmoltz free energy (in case of fixed volume)
            # Or the Gibbs free energy (in case of fixed pressure)
            mark_helmoltz = ""
            mark_gibbs = ""
            if fix_volume:
                mark_helmoltz = "<--"
            else:
                mark_gibbs = "<--"

            # Extract the bulk modulus from the cell minimization
            new_bulkmodulus = 1 / (3 * BFGS.alpha * self.minim.dyn.structure.get_volume())
            new_bulkmodulus *= sscha.SchaMinimizer.__evA3_to_GPa__

            # Print the enthalpic contribution
            message = """
 ======================
 ENTHALPIC CONTRIBUTION
 ======================

 P = {:.4f} GPa   V = {:.4f} A^3

 P V = {:.8e} eV

 Helmoltz Free energy = {:.10e} eV {}
 Gibbs Free energy = {:.10e} eV {}
 Zero energy = {:.10e} eV

 """.format(target_press , Vol,target_press_evA3 * Vol, helmoltz, mark_helmoltz, gibbs, mark_gibbs, self.minim.eq_energy)
            print(message)
            # print " ====================== "
            # print " ENTHALPIC CONTRIBUTION "
            # print " ====================== "
            # print ""
            # print "  P = %.4f GPa    V = %.4f A^3" % (target_press , Vol)
            # print ""
            # print "  P V = %.8e eV " % (target_press_evA3 * Vol)
            # print ""
            # print " Helmoltz Free energy = %.8e eV " % helmoltz,
            # if fix_volume:
            #     print "  <-- "
            # else:
            #     print ""
            # print " Gibbs Free energy = %.8e eV " % gibbs,
            # if fix_volume:
            #     print ""
            # else:
            #     print "  <-- "
            # print " (Zero energy = %.8e eV) " % self.minim.eq_energy
            # print ""

            # Perform the cell step
            if self.fix_cell_shape:
                # Use a isotropic stress tensor to keep the same cell shape
                cell_gradient = I * (Press - target_press_evA3)
            else:
                cell_gradient = (stress_tensor - I *target_press_evA3)

            new_uc = self.minim.dyn.structure.unit_cell.copy()
            BFGS.UpdateCell(new_uc,  cell_gradient, fix_volume)

            # Strain the structure and the q points preserving the symmetries
            self.minim.dyn.AdjustToNewCell(new_uc)
            #self.minim.dyn.structure.change_unit_cell(new_uc)

            message = """
 Currently estimated bulk modulus = {:8.3f} GPa
 (Note: this is just indicative, do not use it for computing bulk modulus)

 """.format(new_bulkmodulus)
            print(message)


            print (" New unit cell:")
            print (" v1 [A] = (%16.8f %16.8f %16.8f)" % (new_uc[0,0], new_uc[0,1], new_uc[0,2]))
            print (" v2 [A] = (%16.8f %16.8f %16.8f)" % (new_uc[1,0], new_uc[1,1], new_uc[1,2]))
            print (" v3 [A] = (%16.8f %16.8f %16.8f)" % (new_uc[2,0], new_uc[2,1], new_uc[2,2]))

            print ()
            print ("Check the symmetries in the new cell:")
            sys.stdout.flush()
            qe_sym = CC.symmetries.QE_Symmetry(self.minim.dyn.structure)
            qe_sym.SetupQPoint(verbose = True)

            print ("Forcing the symmetries in the dynamical matrix.")
            fcq = np.array(self.minim.dyn.dynmats, dtype = np.complex128)
            qe_sym.SymmetrizeFCQ(fcq, self.minim.dyn.q_stars, asr = "custom")
            for iq,q in enumerate(self.minim.dyn.q_tot):
                self.minim.dyn.dynmats[iq] = fcq[iq, :, :]

            # Save the dynamical matrix
            self.minim.dyn.save_qe("dyn_pop%d_" % pop)

            # Save the structure in CONTCAR (vasp) format for convenience
            current_ase_structure = self.minim.dyn.structure.get_ase_atoms()
            ase.io.write(f'dyn_pop{pop}.CONTCAR', current_ase_structure, "vasp", direct=True, label = f'Current structure after minimization during SSCHA relaxation corresponding to dyn_pop{pop}')
            ase.io.write('CONTCAR', current_ase_structure, "vasp", direct=True, label = f'Current structure after minimization during SSCHA relaxation corresponding to dyn_pop{pop}')


            # Check if the constant volume calculation is converged
            running1 = not self.minim.is_converged()

            # Check if the cell variation is converged
            running2 = True
            not_zero_mask = stress_err != 0
            if not fix_volume:
                if np.max(np.abs(cell_gradient[not_zero_mask]) / stress_err[not_zero_mask]) <= 1:
                    running2 = False
            else:
                if np.max(np.abs((stress_tensor - I * Press)[not_zero_mask] /
                                 stress_err[not_zero_mask])) <= 1:
                    running2 = False


            running = running1 or running2

            pop += 1

            if pop > self.max_pop:
                running = False

        self.start_pop = pop
        return (not running1) and (not running2)




















""" Helping functions """


def train_mtp_on_cfg(specorder, mlip_run_command, pot_name, 
                    ab_initio_calculator, ab_initio_parameters, ab_initio_run_command,
                    ab_initio_kresol, ab_initio_pseudos, ab_initio_cluster,
                    iteration_limit = 500, 
                    energy_weight = 1.0, 
                    force_weight = 0.01,
                    stress_weight = 0.001,
                    include_stress = False,
                    train_local_mtps = False,
                    pop = 0,
                    np_ab_initio = 1,
                    np_mlp_train = 1):

    print(f"Preparing files for (re)training MTP {pot_name}")
    # cmd_gain_cfg   = 'cat preselected.cfg* >> preselected.cfg; rm -f preselected.cfg.*'
    cmd_gain_cfg   = 'cat preselected.cfg* >> preselected.cfg'
    cmd_select_add = f'{mlip_run_command} \
                        select_add {pot_name} set.cfg preselected.cfg selected.cfg'
    cmd_mlip_train = f'{mlip_run_command} \
                        train {pot_name} set.cfg \
                        --tolerance=0.01 \
                        --energy_weight={energy_weight} \
                        --force_weight={force_weight} \
                        --stress_weight={stress_weight} \
                        --iteration_limit={iteration_limit}'

    if train_local_mtps:
        try:
            os.system(f'cp {pot_name}.bak {pot_name}')
        except:
            pass


    try:
        os.system(cmd_gain_cfg)
    except:
        pass
    # os.system('pwd')

    # selection of structures with MaxVol algorithm
    os.system(cmd_select_add)

    # gaining the number of selected structures
    n_cfg_cmd = 'grep "BEGIN" selected.cfg | wc -l'
    n_cfg = int(os.popen(n_cfg_cmd).read().split()[0])
    print(f"There are {n_cfg} configurations that need to be added to the training set") 

    # spliting the file with selected structures on n_cfg files 
    try:
        split_cfg('selected.cfg')
    except FileNotFoundError:
        print('There are no new structures that need to be added to the train set!')
        return

    if n_cfg > 0:
        # ab initio calculation of energies, forces, and stresses for selected structures 
        for i in range(n_cfg):
            print(f"Calculating ab initio energies and forces for configuration selected.cfg.{i}")
            # cmd_convert  = f'{mlip_run_command} {path_to_mlip} convert --output_format=poscar sampled.cfg.{i} {i}.POSCAR' 
            # os.system(cmd_convert)
            # ab_initio_dir = f'ab_initio_dir_{i}'+ '_' + str(datetime.now()).replace(' ','_').replace(':','-')
            ab_initio_dir = f'ab_initio_dir'
            cmd_mkdir_abinitio = f'mkdir {ab_initio_dir}'
            if not os.path.exists(ab_initio_dir): 
                os.system(cmd_mkdir_abinitio)
            else: 
                print(f'Ab initio directory {ab_initio_dir} already exists!')
            
            ab_initio_ase_atoms = one_cfg_to_atoms(f'selected.cfg.{i}',specorder) 

            k_points = calc_ngkpt(ab_initio_ase_atoms.cell.reciprocal(),ab_initio_kresol)
            print(f'KPOINTS: {k_points}')

            ### Seting up ab initio calculations ###
            is_converged = False

            if ab_initio_calculator == "QE":
                
                ab_initio_calc = Espresso(pseudopotentials = ab_initio_pseudos,
                                            input_data = ab_initio_parameters,
                                            kpts = k_points,
                                            command = ab_initio_run_command,
                                            koffset  = (0,0,0))
                ab_initio_calc.set_directory(ab_initio_dir)
                in_ext  = ".pwi"
                out_ext = ".pwo"   

                print(ab_initio_calc.command)

            elif ab_initio_calculator == "VASP":

                os.system(f'export VASP_PP_PATH={ab_initio_pseudos}')
                if 'SETUPS' in ab_initio_parameters.keys():
                    setups = ab_initio_parameters['SETUPS']  
                elif 'setups' in ab_initio_parameters.keys():
                    setups = ab_initio_parameters['setups']
                else:
                    setups = 'recommended'

                if 'PREC' in ab_initio_parameters.keys():
                    prec = ab_initio_parameters['PREC']
                elif 'prec' in ab_initio_parameters.keys():
                    prec = ab_initio_parameters['prec']  
                else: 
                    prec = 'Accurate' # Accurate setting for better forces accuracy

                if 'ENCUT' in ab_initio_parameters.keys():
                    encut = ab_initio_parameters['ENCUT']
                elif 'encut' in ab_initio_parameters.keys():
                    encut = ab_initio_parameters['encut']                    
                else:
                    encut = None

                if 'EDIFF' in ab_initio_parameters.keys():
                    ediff = ab_initio_parameters['EDIFF']
                elif 'ediff' in ab_initio_parameters.keys():
                    ediff = ab_initio_parameters['ediff']
                else:
                    ediff = 1e-4

                if 'NBANDS' in ab_initio_parameters.keys():
                    nbands = ab_initio_parameters['NBANDS']
                elif 'nbands' in ab_initio_parameters.keys():
                    nbands = ab_initio_parameters['nbands']
                else:
                    nbands = None

                if 'ALGO' in ab_initio_parameters.keys():
                    algo = ab_initio_parameters['ALGO']
                else:
                    algo = 'Normal'

                if 'ISMEAR' in ab_initio_parameters.keys():
                    ismear = ab_initio_parameters['ISMEAR']
                elif 'ismear' in ab_initio_parameters.keys():
                    ismear = ab_initio_parameters['ismear']
                else:
                    ismear = 1

                if 'SIGMA' in ab_initio_parameters.keys():
                    sigma = ab_initio_parameters['SIGMA']
                elif 'sigma' in ab_initio_parameters.keys():
                    sigma = ab_initio_parameters['sigma']
                else:
                    sigma = 0.05

                if 'NELM' in ab_initio_parameters.keys():
                    nelm = ab_initio_parameters['NELM']
                elif 'nelm' in ab_initio_parameters.keys():
                    nelm = ab_initio_parameters['nelm']
                else:
                    nelm = 500

                if 'ISTART' in ab_initio_parameters.keys():
                    istart = ab_initio_parameters['ISTART']
                elif 'istart' in ab_initio_parameters.keys():
                    istart = ab_initio_parameters['istart']
                else:
                    istart = 0

                if 'LCHARG' in ab_initio_parameters.keys():
                    lcharg = ab_initio_parameters['LCHARG']
                elif 'lcharg' in ab_initio_parameters.keys():
                    lcharg = ab_initio_parameters['lcharg']
                else:
                    lcharg = 'FALSE'

                if 'LWAVE' in ab_initio_parameters.keys():
                    lwave = ab_initio_parameters['LWAVE']
                elif 'lwave' in ab_initio_parameters.keys():
                    lwave = ab_initio_parameters['lwave']
                else:
                    lwave = 'FALSE'

                if 'LREAL' in ab_initio_parameters.keys():
                    lreal = ab_initio_parameters['LREAL']
                elif 'lreal' in ab_initio_parameters.keys():
                    lreal = ab_initio_parameters['lreal']                    
                else:
                    lreal = 'Auto'

                if 'ISPIN' in ab_initio_parameters.keys():
                    ispin = ab_initio_parameters['ISPIN']
                elif 'ispin' in ab_initio_parameters.keys():
                    ispin = ab_initio_parameters['ispin']
                else:
                    ispin = 1

                NATOMS = ab_initio_ase_atoms.get_number_of_atoms()
                if 'MAGMOM' in ab_initio_parameters.keys():
                    magmom = ab_initio_parameters['MAGMOM']
                elif 'magmom' in ab_initio_parameters.keys():
                    magmom = ab_initio_parameters['magmom']
                else:
                    magmom = [1.0 for i in range(NATOMS)]


                ab_initio_calc = Vasp(directory = ab_initio_dir,
                                      label = 'vasp',
                                      command = ab_initio_run_command,
                                      setups = setups,
                                      txt = 'vasp.out',
                                      pp = "PBE",
                                      kpts = k_points,
                                      prec = prec,
                                      encut = encut,
                                      ediff = ediff,
                                      nbands = nbands,
                                      algo = algo,
                                      ismear = ismear,
                                      sigma = sigma,
                                      nelm = nelm,
                                      istart = istart,
                                      lcharg = lcharg,
                                      lwave = lwave,
                                      lreal = lreal,
                                      ispin = ispin,
                                      magmom = magmom,
                                      )
                # ab_initio_calc.set_directory(ab_initio_dir)
                in_ext  = "POSCAR"
                out_ext = "OUTCAR"                  
                print(ab_initio_calc.command)


            ### Starting Ab initio calculations ###
            if ab_initio_cluster != None:
                print("Using HPC for ab initio calculations")
                ab_initio_cluster.run_atoms(ab_initio_calc, ab_initio_ase_atoms, in_extension = in_ext, out_extension = out_ext)
                ab_initio_cluster.read_results(ab_initio_calc, ab_initio_cluster.label)

            elif ab_initio_cluster == None:

                if isinstance(ab_initio_calc, cellconstructor.calculators.Calculator):
                    cc_struct = CC.Structure.Structure()
                    cc_struct.generate_from_ase_atoms(ab_initio_ase_atoms)
                    ab_initio_calc.set_label("ESP")
                    try:
                        ab_initio_calc.calculate(cc_struct)
                    except Exception as e:
                        print(f'Ab initio calculation for structure {i} is failed! Continuing with second structure...')
                        os.system(f'mv {ab_initio_dir} err_{ab_initio_dir}_{i}')
                        continue
                        
                elif isinstance(ab_initio_calc, ase.calculators.calculator.Calculator):
                    try:
                        ab_initio_calc.calculate(ab_initio_ase_atoms)
                    except Exception as e:
                        # print(e.message)
                        print(f'Ab initio calculation for structure {i} is failed! Continuing with second structure...')
                        os.system(f'mv {ab_initio_dir} err_{ab_initio_dir}_{i}')
                        continue

                else:
                    raise(NotImplementedError)


            ### Processing results of ab initio calculations
            # checking if calculation is converged
            if ab_initio_calculator == "QE":
                results = ab_initio_calc.results
                if results != None:
                    is_converged = True
            elif ab_initio_calculator == "VASP":
                is_converged = ab_initio_calc.read_convergence()
            
            # writing cfg file for MLIP
            if is_converged:
                calc_to_cfg(ab_initio_calc,f'input.cfg.{i}',specorder, include_stress)
            else:
                print(f'Ab initio calculation for structure {i} is not converged! Continuing with second structure...')

            os.system(f'rm -rf {ab_initio_dir}')


        os.system(f'cp set.cfg set.cfg.bak')

        os.system('cat input.cfg* >> set.cfg')
        os.system(f'cp {pot_name} {pot_name}.bak')
        print('Start training MLIP')
        os.system(cmd_mlip_train)
        print('End training MLIP')
        if train_local_mtps:
            os.system(f'cat input.cfg* >> all_input_pop_{str(pop)}.cfg')
            os.system(f'cat selected.cfg* >> all_selected_pop_{str(pop)}.cfg')
            os.system(f'mv set.cfg set_pop_{str(pop)}.cfg')
            os.system(f'cp {pot_name} {pot_name}.pop_{str(pop)}')
            # os.system(f'cp {pot_name}.bak {pot_name}')
            os.system('rm -f input.cfg*')
            os.system('rm -f selected.cfg*')
            os.system('rm -f preselected.cfg*')
        else:
            os.system(f'cat input.cfg* >> all_input.cfg')
            os.system(f'cat selected.cfg* >> all_selected.cfg')
            os.system('rm -f input.cfg*')
            os.system('rm -f selected.cfg*')
            os.system('rm -f preselected.cfg*')
    else:
        print('No new configurations are need to be added into the training set! Training will be ommited this time!')
        try:
            os.system('rm -f input.cfg*')
            os.system('rm -f selected.cfg*')
            os.system('rm -f preselected.cfg*')
        except:
            pass

    return  

def ase_structures_list_to_cfg(ase_structures_list,path_to_cfg,specorder):

    """
    Function for converting list of ase stuctures to one cfg file for MLIP package.

    ase_structures_list - list of ase structures
    path_to_cfg - path where the cfg file will be saved

    """

    with open(path_to_cfg, 'w') as f:
        for ase_structure in ase_structures_list:
            try:
                # if structure is true ase structure (i.e. not CC structure) create new CC structure object
                structure = CC.Structure.Structure()
                # and generate CC structure from ase structure
                structure.generate_from_ase_atoms(ase_structure)
            except AttributeError:
                # if structure is CC structure (i.e. not true ase structure) then copy it to the new object
                structure = ase_structure.copy()
                # and get true ase structure object from this CC structure
                ase_structure = structure.get_ase_atoms()
                
            # f.write('\n')
            f.write('BEGIN_CFG\n')
            f.write(' Size\n')
            f.write(f'    {structure.N_atoms}\n')
            f.write(' Supercell\n')
            cell = structure.unit_cell
            f.write(f'         {cell[0][0]:.6f}      {cell[0][1]:.6f}      {cell[0][2]:.6f}\n')
            f.write(f'         {cell[1][0]:.6f}      {cell[1][1]:.6f}      {cell[1][2]:.6f}\n')
            f.write(f'         {cell[2][0]:.6f}      {cell[2][1]:.6f}      {cell[2][2]:.6f}\n')
            f.write(' AtomData:  id type       cartes_x      cartes_y      cartes_z\n')
            # coords = structure.get_xcoords()
            coords = structure.coords
            typat = ase_structure.get_chemical_symbols()
            mapping = {val: i for i, val in enumerate(specorder)}
            for i in range(structure.N_atoms):
                f.write(f'             {i+1:3}    {mapping[typat[i]]}       {coords[i][0]:.6f}      {coords[i][1]:.6f}      {coords[i][2]:.6f} \n')
            f.write(f'END_CFG\n')
            f.write(f'\n')


    return

def calc_ngkpt(recip, kspacing):
    to_ang_local = 1
    
    N_from_kspacing = []
    for i in 0, 1, 2:
        N_from_kspacing.append( int(np.ceil( (np.linalg.norm(recip[i]) / to_ang_local) / kspacing)) )

    return N_from_kspacing

def one_cfg_to_atoms(path_to_cfg,specorder):
    """
    Function for converting cfg file with ONE structure to ase_atoms object

    path_to_cfg - a path to the cfg file th
    specorder - a list with species order

    """

    with open(path_to_cfg,'r') as f:
        lines = f.readlines()
        num_at = int(lines[2].split('/n')[0])
        vec1 = [float(lines[4].split()[0]), float(lines[4].split()[1]), float(lines[4].split()[2])]
        vec2 = [float(lines[5].split()[0]), float(lines[5].split()[1]), float(lines[5].split()[2])]
        vec3 = [float(lines[6].split()[0]), float(lines[6].split()[1]), float(lines[6].split()[2])]
        xcart = []
        symbols = []
        for i in range(num_at):
            xcart.append([])
            t = int(lines[8+i].split()[1])
            x = float(lines[8+i].split()[2])
            y = float(lines[8+i].split()[3])
            z = float(lines[8+i].split()[4])
            xcart[i].append(x)
            xcart[i].append(y)
            xcart[i].append(z)
            symbols.append(specorder[t])

        cell = [vec1,vec2,vec3]

    # print(f'num_at = {num_at}')
    # print(f'cell = {cell}')
    # print(f'xcart = {xcart}')
    # print(f'symbols = {symbols}')

    ase_atoms_cfg = ase.Atoms(symbols = symbols,
                              positions = xcart,
                              cell = cell,
                              pbc=[1, 1, 1]
                             )

    return ase_atoms_cfg

def split_cfg(path_to_cfg):
    """
    Function for splitting one cfg with multiple configurations to several cfg's with one configuration in each
    """
    n_config = 0
    configs = []
    with open(path_to_cfg, 'r') as f:
        lines = f.readlines()
        for line in lines:
        # while True:
            # line = f.readline()
            if 'BEGIN_CFG' in line:
                configs.append([])
                configs[n_config].append(line)
            elif 'END_CFG' in line:
                configs[n_config].append(line)
                n_config += 1
            else:
                try:
                    configs[n_config].append(line)
                except IndexError:
                    continue

    for i, config in enumerate(configs):
        path_to_cfg_i = f'{path_to_cfg}.{i}'
        with open(path_to_cfg_i, 'w') as f:
            f.writelines(config)

    return
                

def calc_to_cfg(calc,path_to_cfg,specorder,include_stress = False):
    """
    Function for converting cellconstructor or ASE calculator object to cfg file for MTP training.
    The cfg is written 

    calc - cellconstructor or ASE calculator object
    path_to_cfg - path where the cfg file will be saved
    specorder - a list with species order

    """

    if isinstance(calc, ase.calculators.calculator.Calculator):
        N_atoms = calc.atoms.get_number_of_atoms()
        cell = calc.atoms.get_cell()
        coords = calc.atoms.get_positions()

        count = 1
        dictionary = {}
        for i, atm in enumerate(calc.atoms.get_atomic_numbers()):
            if not atm in dictionary:
                dictionary[atm] = count
                count += 1
        
        # typat = [dictionary[x] for x in calc.atoms.get_atomic_numbers()]
        forces = calc.get_forces()
        energy = calc.get_potential_energy()
        volume = calc.atoms.get_volume()
        ase_structure = calc.atoms
        if include_stress:
            stresses = calc.get_stress()

    elif isinstance(calc, cellconstructor.calculators.Calculator):
        N_atoms = calc.structure.N_atoms
        cell = calc.structure.unit_cell
        coords = calc.structure.coords
        # typat  = calc.structure.get_atomic_types()
        ase_structure = calc.structure.get_ase_atoms()
        forces = calc.results["forces"]
        energy = calc.results["energy"]
        volume = calc.structure.get_volume()
        if "stress" in calc.results and include_stress:
            stresses = calc.results["stress"]

    else:
        raise ValueError("Error, unknown calculator type")

    typat = ase_structure.get_chemical_symbols()
    mapping = {val: i for i, val in enumerate(specorder)}    

    with open(path_to_cfg, 'w') as f:
        f.write('\n')
        f.write('BEGIN_CFG\n')
        f.write(' Size\n')
        f.write(f'    {N_atoms}\n')
        f.write(' Supercell\n')
        f.write(f'         {cell[0][0]:.6f}      {cell[0][1]:.6f}      {cell[0][2]:.6f}\n')
        f.write(f'         {cell[1][0]:.6f}      {cell[1][1]:.6f}      {cell[1][2]:.6f}\n')
        f.write(f'         {cell[2][0]:.6f}      {cell[2][1]:.6f}      {cell[2][2]:.6f}\n')
        f.write(' AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz\n')

        for i in range(N_atoms):
            f.write(f'             {i+1:3}    {mapping[typat[i]]}       {coords[i][0]:.6f}      {coords[i][1]:.6f}      {coords[i][2]:.6f}      {forces[i][0]:.6f} {forces[i][1]:.6f} {forces[i][2]:.6f}\n')
        f.write(f' Energy\n')
        f.write(f' {energy}\n')
        if include_stress:
            pxx = -stresses[0]*volume
            pyy = -stresses[1]*volume
            pzz = -stresses[2]*volume
            pyz = -stresses[3]*volume
            pxz = -stresses[4]*volume
            pxy = -stresses[5]*volume
            f.write(f' PlusStress:  xx          yy          zz          yz          xz          xy\n')
            f.write(f'        {pxx}    {pyy}    {pzz}    {pyz}    {pxz}    {pxy}\n')
        f.write(f'END_CFG\n')
        f.write(f'\n')

    return


def are_ion_distances_good(structure, min_distances):

    """
    structure - CellConstructor structure object
    min_distances - dictionary with minimal interatomic distances between atoms in Angstroms
    example of min_distances (similar to USPEX): 
    {'Mg Mg': 1.0, 'Mg Si': 1.0, 'Mg O': 0.8, 'Si Si': 1.0, 'Si O': 0.8, 'O O': 1.0}

    
    """

    # the case when min_distances are not specified
    if min_distances == None:
        return True

    # converting min_distances to blmin dict as in ase.ga.utilities
    blmin = convert_min_distances_to_bl(structure, min_distances)

    atoms = structure.get_ase_atoms()

    are_ion_distances_bad = atoms_too_close(atoms, blmin)

    if are_ion_distances_bad: 
        # print('Structure violates min_distance constraints and will not be added!')
        return False

    return True

def convert_min_distances_to_bl(structure,min_distances):

    """
    Function for converting USPEX-like min_distances dictionary to ase
    """

    s = structure.get_ase_atoms()

    symbols_numbers = dict(zip(s.get_chemical_symbols(), s.get_atomic_numbers()))

    # New dictionary to store the transformed data
    d = {}

    for key, value in min_distances.items():
        # Split the key into the individual symbols
        elements = key.split()
        
        # Attempt to replace symbols with numbers, creating a tuple as the new key
        # This approach assumes all elements can be replaced; otherwise, they are ignored
        try:
            new_key = tuple(symbols_numbers[elem] for elem in elements if elem in symbols_numbers)
            # Only add to the new dictionary if the new_key has two elements, matching your requirement
            if len(new_key) == 2:
                d[new_key] = value
        except KeyError:
            # Handle the case where an element isn't found in symbols_numbers, if necessary
            pass

    return d

# Function from ase.ga.utilities
def atoms_too_close(atoms, bl, use_tags=False):
    """Checks if any atoms in a are too close, as defined by
    the distances in the bl dictionary.

    use_tags: whether to use the Atoms tags to disable distance
        checking within a set of atoms with the same tag.

    Note: if certain atoms are constrained and use_tags is True,
    this method may return unexpected results in case the
    contraints prevent same-tag atoms to be gathered together in
    the minimum-image-convention. In such cases, one should
    (1) release the relevant constraints,
    (2) apply the gather_atoms_by_tag function, and
    (3) re-apply the constraints, before using the
        atoms_too_close function.
    """
    a = atoms.copy()
    if use_tags:
        gather_atoms_by_tag(a)

    pbc = a.get_pbc()
    cell = a.get_cell()
    num = a.get_atomic_numbers()
    pos = a.get_positions()
    tags = a.get_tags()
    unique_types = sorted(list(set(num)))

    neighbours = []
    for i in range(3):
        if pbc[i]:
            neighbours.append([-1, 0, 1])
        else:
            neighbours.append([0])

    for nx, ny, nz in itertools.product(*neighbours):
        displacement = np.dot(cell.T, np.array([nx, ny, nz]).T)
        pos_new = pos + displacement
        distances = cdist(pos, pos_new)

        if nx == 0 and ny == 0 and nz == 0:
            if use_tags and len(a) > 1:
                x = np.array([tags]).T
                distances += 1e2 * (cdist(x, x) == 0)
            else:
                distances += 1e2 * np.identity(len(a))

        iterator = itertools.combinations_with_replacement(unique_types, 2)
        for type1, type2 in iterator:
            x1 = np.where(num == type1)
            x2 = np.where(num == type2)
            try:
                mindist = bl[(type1, type2)]
            except KeyError:
                mindist = bl[(type2, type1)]
            except:
                mindist = 0.0
            # if np.min(distances[x1].T[x2]) < bl[(type1, type2)]:
            if np.min(distances[x1].T[x2]) < mindist:
                return True

    return False