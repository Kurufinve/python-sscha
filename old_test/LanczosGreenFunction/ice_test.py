from __future__ import print_function

"""
In this simple example we test the Lanczos procedure on ice. To see if we get
some meaningful result or not.
"""
import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt

import cellconstructor as CC
import cellconstructor.Phonons

import sscha, sscha.Ensemble, sscha.DynamicalLanczos

T = 0
SUPERCELL = (1,1,1)
DATADIR = "../ensemble_data_test"
POPULATION = 2
NRANDOM = 10

# Where to store the progress?
#SAVE_DIR = "data_odd4_full_sym_cfast2"
#SAVE_DIR = "data_odd4_full_sym"
SAVE_DIR = "data"

# The frequencies/smearing for the dynamical responce
W_START = 0 #-5000/ CC.Phonons.RY_TO_CM
W_END = 3400 / CC.Phonons.RY_TO_CM
NW = 5000
SMEARING = 1 / CC.Phonons.RY_TO_CM


# The number of eigenvalues to return
N_VALS = 16
N_ITERS = 100

# If the data dir does not exist, create it
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load the original dynamical matrix
dyn = CC.Phonons.Phonons("%s/dyn" % DATADIR)
w_sscha, p = dyn.DyagDinQ(0)
w_sscha *= CC.Phonons.RY_TO_CM

# Load the original ensemble
ens = sscha.Ensemble.Ensemble(dyn, T, SUPERCELL)
ens.load(DATADIR, POPULATION, NRANDOM)

# Unwrap symmetries
ens.unwrap_symmetries()

# Compute the Lanczos matrix
LanczosResponce = sscha.DynamicalLanczos.Lanczos(ens)

# Ignore for now v3 and v4
LanczosResponce.ignore_v3 = True
LanczosResponce.ignore_v4 = True


# Get first the lowest eigenvalues
#LanczosResponce.set_max_frequency(3000/CC.Phonons.RY_TO_CM)


# Prepare the lanczos algorithm with a random vector
nat = dyn.structure.N_atoms
random_vector = np.random.uniform(size = 3*nat)
random_vector /= np.sqrt(random_vector.dot(random_vector))



LanczosResponce.prepare_perturbation(random_vector)
np.savetxt("psi0.dat", LanczosResponce.psi)

print ()
print (LanczosResponce.psi)
print ()

print("Preparation compleated.")
print("Running the Lanczos algorithm...")

step_file = "%s/LANCZOS_STEP10.npz" % (SAVE_DIR)#, N_ITERS-1)
if os.path.exists(step_file):
    LanczosResponce.load_status(step_file)
    print("Status loaded from file %s." % step_file)
    LanczosResponce.run(N_ITERS, SAVE_DIR)

    np.savetxt("a_coeffs_start.txt", LanczosResponce.a_coeffs)
    np.savetxt("b_coeffs_start.txt", LanczosResponce.b_coeffs)
else:
    t1 = time.time()
    #LanczosResponce.run_full_diag(N_VALS, n_iter = N_ITERS)
    print("A coeffs:", LanczosResponce.a_coeffs)
    LanczosResponce.run(N_ITERS, SAVE_DIR)
    t2 = time.time()
    np.savetxt("a_coeffs_continue.txt", LanczosResponce.a_coeffs)
    np.savetxt("b_coeffs_continue.txt", LanczosResponce.b_coeffs)

    print("Lanczos ended. Time elapsed = %.4f s" % (t2-t1))
    

# print()
# print("Eigenvalues found:", LanczosResponce.eigvals)

# # Now get the self-energy
# self_energy = LanczosResponce.GetFullSelfEnergy()

# print("Saving the self energy to 'self_energy.dat'")
# np.savetxt("self_energy.dat", self_energy)

# Now get the spectral function
w_array = np.linspace(W_START, W_END, NW)
spectral_function =LanczosResponce.get_spectral_function_from_Lenmann(w_array, SMEARING, False)
#static_odd = LanczosResponce.get_static_odd_fc(True)
#static_odd = LanczosResponce.get_statical_responce_from_scratch(n_iters = 100, thr = 1e-4)


# Plot the spectral function
plt.plot(w_array * CC.Phonons.RY_TO_CM, spectral_function)
plt.title("Spectral function")
plt.xlabel("Frequency [cm-1]")
plt.ylabel("Spectral function")

# Show the static and sscha phonons
for i in range(len(w_sscha)):
    w_s = w_sscha[i]

    plt.vlines(w_s, 0, np.max(spectral_function)*1.1, linestyle = "dotted", color ="k")

plt.tight_layout()
plt.show()