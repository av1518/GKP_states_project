#%%
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from thewalrus.quantum import state_vector, density_matrix

#%%

# Quantum Circuit Parameters
m1, m2 = 5, 7
params = np.array([-1.38155106, -1.21699567,  0.7798817,  1.04182349,
                   0.87702211, 0.90243916,  1.48353639,  1.6962906 ,
                   -0.24251599, 0.1958])
sq_r = params[:3]
bs_theta1, bs_theta2, bs_theta3 = params[3:6]
bs_phi1, bs_phi2, bs_phi3 = params[6:9]
sq_virt = params[9]

# Quantum Circuit
nmodes = 3
prog = sf.Program(nmodes)
eng = sf.Engine("gaussian")

with prog.context as q:
    for k in range(3):
        Sgate(sq_r[k]) | q[k]

    BSgate(bs_theta1, bs_phi1) | (q[0], q[1])
    BSgate(bs_theta2, bs_phi2) | (q[1], q[2])
    BSgate(bs_theta3, bs_phi3) | (q[0], q[1])

    Sgate(sq_virt) | q[2]

state = eng.run(prog).state
mu, cov = state.means(), state.cov()

cutoff = 25
psi = state_vector(mu, cov, post_select={0: m1, 1: m2}, normalize=False, cutoff=cutoff)
p_psi = np.linalg.norm(psi)
psi = psi / p_psi
# %%
