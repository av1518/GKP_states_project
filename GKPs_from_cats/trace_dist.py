#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import *
import matplotlib as mpl
from matplotlib import cm
from gkp_target import target_dm, cutoff, dB
import copy
from tqdm import tqdm

a_vals = np.arange(0,4.5,0.05)
s_vals = np.arange(0,4.5,0.05)

s, a = np.meshgrid(a_vals, s_vals)
#%%
tr_i = []
tr = []

for i in tqdm(range(len(a_vals))):
    for j in tqdm(range(len(s_vals))):

        prog = sf.Program(2)
    
        with prog.context as q:
            Catstate(a_vals[i]) | q[0]
            Catstate(a_vals[i]) | q[1]
            Sgate(s_vals[j]) | q[0] 
            Sgate(s_vals[j]) | q[1]
            BSgate() | (q[0], q[1])
            MeasureHomodyne(np.pi/2, select = 0.0) | q[0]

        eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})
        state = eng.run(prog).state
        dm = state.reduced_dm(1)
        trace = 0.5*np.trace(np.absolute(dm - target_dm))

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
np.save('trac_dist_explore', trs)

#%%
trac_dists = np.load('trac_dist_explore.npy',allow_pickle=True)

lists = []
for i in range(len(trac_dists)):
    lists.append(trac_dists[i])

trace_dista = np.vstack((lists))

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
ax1.set_xlabel(r'$a$')
ax1.set_ylabel(r'$s$')
ax1.set_zlabel(r"$Tr$")
threeD_plot = ax1.plot_surface(a, s, trace_dista, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig1.colorbar(threeD_plot, shrink=0.5)
plt.tight_layout()

ag = np.sqrt(np.pi)*np.exp(s_vals)

sgs = []
ags = []
for i in range(len(ag)):
    if ag[i] <= np.max(a_vals):
        ags.append(ag[i])
        sgs.append(s_vals[i])

grid = 800
xvec = np.linspace(-6,6, grid)
sc1 = np.max(trace_dista)
nrm = mpl.colors.Normalize(np.min(trace_dista), sc1)
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
plt1 = axes.contourf(a, s, trace_dista, 60,  cmap=cm.coolwarm, norm=nrm)
axes.contour(a, s, trace_dista, 60,  cmap=cm.coolwarm, norm=nrm)
axes.plot(ags, sgs, color = 'black')
axes.set_title(f'Parameter space for {dB} dB target')
axes.set_xlabel('a')
axes.set_ylabel('s')
cb1 = fig.colorbar(plt1, ax=axes)
fig.tight_layout()

plt.show()
# %%
