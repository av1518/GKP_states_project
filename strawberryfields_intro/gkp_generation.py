#%%
import numpy as np
from qutip import wigner, Qobj, wigner_cmap

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

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

print(np.round(mu, 10))
print(np.round(cov, 10))

cutoff = 25
psi = state_vector(mu, cov, post_select={0: m1, 1: m2}, normalize=False, cutoff=cutoff)
p_psi = np.linalg.norm(psi)
psi = psi / p_psi
print('The probability of successful heralding is {:.5f}.'.format(p_psi ** 2))

#%% Wigner function of heralded state

grid = 800
xvec = np.linspace(-5,5, grid)
Wp = wigner(Qobj(psi), xvec, xvec)
wmap = wigner_cmap(Wp)
sc1 = np.max(Wp)
nrm = mpl.colors.Normalize(-sc1, sc1)
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
plt1 = axes.contourf(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
axes.contour(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
axes.set_title("Wigner function of the heralded state");
cb1 = fig.colorbar(plt1, ax=axes)
fig.tight_layout()

x, y = np.meshgrid(xvec, xvec)
fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
ax1.set_ylabel(r'$p_i$')
ax1.set_xlabel(r'$x_i$')
ax1.set_zlabel(r"$W$")
threeD_plot = ax1.plot_surface(x, y, Wp, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig1.colorbar(threeD_plot, shrink=0.5)
plt.tight_layout()

#%% Plot photon number prob distribution

plt.figure(3)
plt.bar(np.arange(cutoff), np.abs(psi) ** 2)
plt.xlim(-1, 22)
plt.xticks(np.arange(0, 22, 2))
plt.xlabel('$i$')
plt.ylabel(r'$p_i$')

#%% Plot cut of Wigner function at p = 0

plt.figure(4)
plt.plot(xvec, Wp[grid//2,:])
plt.title(r"$W(x,0)$")
plt.xlabel(r"q")

plt.show()

# %%

#Ilan Tzitrin, J. Eli Bourassa, Nicolas C. Menicucci, and Krishna Kumar Sabapathy. Towards practical qubit computation using approximate error-correcting grid state. https://arxiv.org/abs/1910.03673