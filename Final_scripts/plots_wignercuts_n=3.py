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

n = 3
s = 0.3427212441660357
a = 6.696114053323244
dB_target = 13.5
prob_threshold = 0.9

st = q_state(a, dB_target)
opt = optimal_cutoff(prob_threshold)
cutoff = opt.min_cutoff(st)
target_dm = st.target_d_matrix(cutoff)

#%% Circuit

with plt.style.context(['science']):

    fig, ax = plt.subplots(1, 3, figsize = (20,7), constrained_layout=True)
    xvec = np.linspace(-8,8, 800)

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

    Wp = wigner(Qobj(dm), xvec, xvec)
    ax[0].plot(xvec, Wp[800//2,:], color = 'indianred', linewidth = 3)
    ax[0].set_ylabel(r"$W(q,0)$")
    ax[0].set_xlabel(r"$q$")
    ax[0].set_xticks(np.arange(-8, 9, 4))


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

            Wp = wigner(Qobj(dm), xvec, xvec)
            ax[i+1].plot(xvec, Wp[800//2,:], color = 'indianred', linewidth = 3)
            ax[i+1].set_ylabel(r"$W(q,0)$")
            ax[i+1].set_xlabel(r"$q$")
            ax[i+1].set_xticks(np.arange(-8, 9, 4))


    fig.savefig('figures/wignercuts_n=3.png', dpi=400)

#plt.show()