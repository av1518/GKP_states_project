#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
from qutip import wigner, Qobj, wigner_cmap, fidelity
import matplotlib as mpl
from matplotlib import cm
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from gkp_target import dm_target, cutoff, dB
import copy
from tqdm import tqdm

# Target state
target_dm_q = Qobj(dm_target)
target_dm = tf.constant(dm_target, dtype=tf.complex64)

a_vals = np.arange(0,7,0.2)
s_vals = np.arange(0,7,0.2)

s, a = np.meshgrid(a_vals, s_vals)
#%%
tr_i = []
tr = []

for i in tqdm(range(len(a_vals))):
    for j in tqdm(range(len(s_vals))):

        qnn = sf.Program(2)
    
        with qnn.context as q:
            Catstate(a_vals[i]) | q[0]
            Catstate(a_vals[i]) | q[1]

            Rgate(np.pi/2) | q[0]
            Rgate(np.pi/2) | q[0]

            BSgate() | (q[0], q[1])
            Sgate(s_vals[j]) | q[1] #GKP state mode
            MeasureHomodyne(np.pi/2, select = 0.0) | q[0]

        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
        state = eng.run(qnn).state
        dm = state.reduced_dm(1)
        dm_q = Qobj(dm.numpy())

        fid = fidelity(dm_q, target_dm_q)
        trace = 0.5*tf.linalg.trace(tf.abs(dm - target_dm))

        tr_i.append(trace)

        if j == len(s_vals)-1:
            c = 1
        if j != len(s_vals)-1:
            c = 0
        if c == 1:
            tr_j = np.array(copy.deepcopy(tr_i))
            tr.append(tr_j)
            tr_i.clear()
        
        eng.reset()

trs = np.array(tr)

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
ax1.set_ylabel(r'$a$')
ax1.set_xlabel(r'$s$')
ax1.set_zlabel(r"$Tr$")
threeD_plot = ax1.plot_surface(a, s, trs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig1.colorbar(threeD_plot, shrink=0.5)
plt.tight_layout()

grid = 800
xvec = np.linspace(-6,6, grid)
sc1 = np.max(trs)
nrm = mpl.colors.Normalize(-sc1, sc1)
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
plt1 = axes.contourf(a, s, trs, 60,  cmap=cm.RdBu, norm=nrm)
axes.contour(a, s, trs, 60,  cmap=cm.RdBu, norm=nrm)
axes.set_title(f'Trace distance for {dB} dB')
axes.set_xlabel('a')
axes.set_ylabel('s')
cb1 = fig.colorbar(plt1, ax=axes)
fig.tight_layout()

plt.show()