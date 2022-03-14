#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import *
import matplotlib as mpl
from matplotlib import cm
import copy
from tqdm import tqdm
from cutoff_opt import min_cutoff

s = 0.4
a = 2*np.sqrt(2)*np.sqrt(np.pi)*np.exp(s)
cat_cutoff = 25

n_vals = np.arange(0,10,1)
dB_vals = np.arange(6,17,1)
dB, ns = np.meshgrid(dB_vals, n_vals)

#%%
tr_i = []
tr = []

for i in tqdm(range(len(n_vals))):
    for j in tqdm(range(len(dB_vals))):

        gkp_cutoff = min_cutoff(dB_vals[j])

        if cat_cutoff < gkp_cutoff:
            cutoff = gkp_cutoff
        else:
            cutoff = cat_cutoff

        prog_gkp = sf.Program(1)
        e = 10**(-dB_vals[j]/10)
        with prog_gkp.context as q:
            GKP(epsilon = e) | q[0]
            
        eng_gkp = sf.Engine('fock', backend_options = {'cutoff_dim': cutoff})
        gkp = eng_gkp.run(prog_gkp).state
        target_state = gkp
        target_dm = gkp.dm()

        prog = sf.Program(2)
        with prog.context as q:
            Catstate(a) | q[0]
            Catstate(a) | q[1]
            Sgate(s) | q[0] 
            Sgate(s) | q[1]
            BSgate() | (q[0], q[1])
            MeasureHomodyne(np.pi/2, select = 0.0) | q[0]
        
        eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})
        state = eng.run(prog).state
        dm = state.reduced_dm(1)
        trace = 0.5*np.trace(np.absolute(dm - target_dm))

        if n_vals[i] != 0:
            for w in range(n_vals[i]):
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
        eng_gkp.reset()
        if n_vals[i] != 0:
            eng2.reset()

trs = np.array(tr)
np.save('trac_dist3_explore', trs)

#%%
trac_dists = np.load('trac_dist3_explore.npy',allow_pickle=True)

lists = []
for i in range(len(trac_dists)):
    lists.append(trac_dists[i])

trace_dista = np.vstack((lists))

centers = [dB_vals[0] , dB_vals[-1], n_vals[0] , n_vals[-1]]
dx, = np.diff(centers[:2])/(trace_dista.shape[1]-1)
dy, = -np.diff(centers[2:])/(trace_dista.shape[0]-1)
ext = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
nrm = mpl.colors.Normalize(np.min(trace_dista), np.max(trace_dista))

fig, axes = plt.subplots(1, 1, figsize = (5,4))
plt1 = axes.imshow(trace_dista, origin = 'lower', extent = ext, cmap=cm.coolwarm, norm=nrm)
axes.set_xlabel('dB')
axes.set_ylabel('n')
plt.colorbar(plt1,fraction=0.046, pad=0.04)

plt.show()
# %%
