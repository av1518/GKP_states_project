#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import *
import matplotlib as mpl
from matplotlib import cm
import copy
from tqdm import tqdm
from cutoff_opt import optimal_cutoff, q_state

prob_threshold = 0.95
n = 1
a_vals = np.arange(0.7,7.7,0.2)
s_vals = np.log(a_vals/((np.sqrt(2)**(n-1))*np.sqrt(np.pi)))
dB_vals = np.arange(0.2,14,0.2)
dB, a = np.meshgrid(dB_vals, a_vals)

#%%
tr_i = []
tr = []

for i in tqdm(range(len(a_vals))):
    for j in tqdm(range(len(dB_vals))):

        st = q_state(a_vals[i], dB_vals[j])
        opt = optimal_cutoff(prob_threshold)
        cutoff = opt.min_cutoff(st)
        target_dm = st.target_d_matrix(cutoff)

        prog = sf.Program(2)
        with prog.context as q:
            Catstate(a_vals[i]) | q[0]
            Catstate(a_vals[i]) | q[1]
            Sgate(s_vals[i]) | q[0] 
            Sgate(s_vals[i]) | q[1]
            BSgate() | (q[0], q[1])
            MeasureHomodyne(np.pi/2, select = 0.0) | q[0]
        
        eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})
        state = eng.run(prog).state
        dm = state.reduced_dm(1)
        trace = 0.5*np.trace(np.absolute(dm - target_dm))

        if n > 1:
            for w in range(n-1):
                prog2 = sf.Program(2)

                with prog2.context as q:
                    DensityMatrix(dm) | q[0]
                    DensityMatrix(dm) | q[1]
                    BSgate() | (q[0],q[1])
                    MeasureHomodyne(np.pi/2,select=0.0) | q[1]

                eng2 = sf.Engine(backend="fock", backend_options={"cutoff_dim":cutoff})
                state2 = eng2.run(prog2).state
                dm = state2.reduced_dm(0)
                trace = 0.5*np.trace(np.absolute(dm - target_dm))

        tr_i.append(trace)
        if j == len(dB_vals)-1:
            c = 1
        if j != len(dB_vals)-1:
            c = 0
        if c == 1:
            tr_j = np.array(copy.deepcopy(tr_i))
            tr.append(tr_j)
            tr_i.clear()
        
        eng.reset()
        if n > 1:
            eng2.reset()

trs = np.array(tr)
np.save('trac_dist1_explore', trs)

#%%
trac_dists = np.load('trac_dist1_explore.npy',allow_pickle=True)

lists = []
for i in range(len(trac_dists)):
    lists.append(trac_dists[i])

trace_dista = np.vstack((lists))

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
ax1.set_xlabel(r'$dB$')
ax1.set_ylabel(r'$a$')
ax1.set_zlabel(r"$Tr$")
threeD_plot = ax1.plot_surface(dB, a, trace_dista, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig1.colorbar(threeD_plot, shrink=0.5)
plt.tight_layout()

nrm = mpl.colors.Normalize(np.min(trace_dista), np.max(trace_dista))
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
plt1 = axes.contourf(dB, a, trace_dista, 60,  cmap=cm.coolwarm, norm=nrm)
axes.contour(dB, a, trace_dista, 60,  cmap=cm.coolwarm, norm=nrm)
axes.set_xlabel('dB')
axes.set_ylabel('a')
fig.colorbar(plt1, ax=axes)
fig.tight_layout()

print(np.min(trace_dista))

plt.show()
