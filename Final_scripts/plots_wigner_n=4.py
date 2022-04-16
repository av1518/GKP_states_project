#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from qutip import wigner, Qobj, fidelity
import matplotlib as mpl
from matplotlib import cm
from cutoff_opt import q_state, optimal_cutoff
import matplotlib.transforms as mtransforms

plot_params = {'axes.labelsize':30,
          'axes.titlesize':18,
          'font.size':27,
          'figure.figsize':[10,10],
          'xtick.major.size':30,
          'xtick.minor.size':20,
          'ytick.major.size': 30,
          'ytick.minor.size': 20,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix',
          'lines.linewidth': 2 }

mpl.rcParams.update(plot_params)

n = 4
s = 0.3559898602842482
a = 9.590231285601375
dB_target = 15.723376569770377
prob_threshold = 0.9

st = q_state(a, dB_target)
opt = optimal_cutoff(prob_threshold)
cutoff = opt.min_cutoff(st)
target_dm = st.target_d_matrix(cutoff)

#%% Circuit

prog = sf.Program(2)

with prog.context as q:
    ops.Catstate(a) | q[0]
    ops.Catstate(a) | q[1]
    ops.Sgate(s) | q[0]
    ops.Sgate(s) | q[1]
    ops.BSgate() | (q[0], q[1])
    ops.MeasureHomodyne(np.pi/2, select=0.0) | q[0]
    
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim":cutoff})
state = eng.run(prog).state
dm = state.reduced_dm(1)

xs = np.linspace(-30,30, 800)

if n > 1:
    for i in range(n-1):
        prog2 = sf.Program(2)

        with prog2.context as q:
            ops.DensityMatrix(dm) | q[0]
            ops.DensityMatrix(dm) | q[1]
            ops.BSgate() | (q[0],q[1])
            ops.MeasureHomodyne(np.pi/2,select=0.0) | q[1]

        eng2 = sf.Engine(backend="fock", backend_options={"cutoff_dim":cutoff})
        state2 = eng2.run(prog2).state
        dm = state2.reduced_dm(0)

trace = 0.5*np.trace(np.abs(dm - target_dm))
fid = fidelity(Qobj(target_dm), Qobj(dm))
print('Trace distance = ', trace)
print('Fidelity = ', fid)

with plt.style.context(['science']):

    fig, ax = plt.subplots(1, 2, figsize = (21,10), constrained_layout=True)
    xvec = np.linspace(-6,6, 800)

    Wp = wigner(Qobj(target_dm), xvec, xvec)
    sc = np.max(Wp)
    nrm = mpl.colors.Normalize(-sc, sc)
    plt1 = ax[0].contourf(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
    ax[0].contour(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
    ax[0].autoscale(tight=True)
    ax[0].set_ylabel(r'$p$')
    ax[0].set_xlabel(r'$x$')
    ax[0].set_yticks(np.arange(-6, 7, 2))

    Wp2 = wigner(Qobj(dm), xvec, xvec)
    sc2 = np.max(Wp2)
    nrm2 = mpl.colors.Normalize(-sc2, sc2)
    plt2 = ax[1].contourf(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
    ax[1].contour(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
    ax[1].autoscale(tight=True)
    ax[1].set_ylabel(r'$p$')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_yticks(np.arange(-6, 7, 2))

    fig.colorbar(plt2, ax=ax[1], ticks=[-0.15, 0, 0.15, 0.3])
    fig.savefig('figures/wigner_n=4_subplots.png', dpi=400)

#plt.show()