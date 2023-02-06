#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from qutip import wigner, Qobj, fidelity
import matplotlib as mpl
from matplotlib import cm
from cutoff_opt import q_state, optimal_cutoff

n = 3
s = 0.3427212441660357
a = 6.696114053323244
dB_target = 13.5
prob_threshold = 0.9

n = 2
s = 0.8506821934255078
a = 6.365468917869899
dB_target = 11.430964826931291
prob_threshold = 0.997

# n = 4
# s = 0.3559898602842482
# a = 9.590231285601375
# dB_target = 15.723376569770377
# prob_threshold = 0.9

st = q_state(a, dB_target)
opt = optimal_cutoff(prob_threshold)
cutoff = opt.min_cutoff(st)
target_dm = st.target_d_matrix(cutoff)
x = st.model_gkp()[0]
gk = st.model_gkp()[1]
cns = st.fock_coeffs(x,gk,cutoff)
fock_range = np.arange(0,cutoff,1)

with plt.style.context(['science']):
    fig, ax = plt.subplots()
    ax.plot(x, gk**2, label = f'{dB_target:.2f} dB', color = 'darkblue')
    ax.legend()
    ax.autoscale()
    ax.set_ylabel(r'Probability')
    ax.set_xlabel(r'$x$')
    #fig.savefig('figures/target_wf.png', dpi=400)

    plt.figure()
    plt.bar(fock_range,(cns)**2, label = f'Threshold = {prob_threshold}')
    plt.xlabel('Number basis')
    plt.ylabel('Fock probabilities')
    #plt.savefig('figures/fock_coeffs.png', dpi=400)

    xvec = np.linspace(-6,6, 800)
    Wp = wigner(Qobj(target_dm), xvec, xvec)
    sc = np.max(Wp)
    nrm = mpl.colors.Normalize(-sc, sc)
    figu, axes = plt.subplots(1, 1, figsize=(5, 4))
    plt1 = axes.contourf(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
    axes.contour(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
    figu.colorbar(plt1, ax=axes)
    axes.autoscale(tight=True)
    axes.set_ylabel(r'$p$')
    axes.set_xlabel(r'$x$')
    figu.savefig('figures/target_wigner.png', dpi=400)

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

# with plt.style.context(['science']):
#     fig, ax = plt.subplots(figsize = (5,4))
#     ax.plot(xs, state.x_quad_values(1,xs,xs), label = f'a = {a:.2f}', color = 'darkblue')
#     ax.legend()
#     ax.autoscale()
#     ax.set_ylabel(r'Probability')
#     ax.set_xlabel(r'x')
#     fig.savefig('figures/step1.png', dpi=400)

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

        xs = np.linspace(-30,30, 800)

        # with plt.style.context(['science']):
            # fig, ax = plt.subplots(figsize = (5,4))
            # ax.plot(xs, state2.x_quad_values(0,xs,xs), label = f'a = {a:.2f}', color = 'darkblue')
            # ax.legend()
            # ax.autoscale()
            # ax.set_ylabel(r'Probability')
            # ax.set_xlabel(r'$x$')
            # fig.savefig(f'figures/step{i+2}.png', dpi=400)

trace = 0.5*np.trace(np.abs(dm - target_dm))
fid = fidelity(Qobj(target_dm), Qobj(dm))
print('Trace distance = ', trace)
print('Fidelity = ', fid)

with plt.style.context(['science']):
    Wp2 = wigner(Qobj(dm), xvec, xvec)
    sc2 = np.max(Wp2)
    nrm2 = mpl.colors.Normalize(-sc2, sc2)
    fig2, axes2 = plt.subplots(1, 1, figsize=(5, 4))
    plt2 = axes2.contourf(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
    axes2.contour(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
    fig2.colorbar(plt2, ax=axes2)
    axes2.autoscale(tight=True)
    axes2.set_ylabel(r'$p$')
    axes2.set_xlabel(r'$x$')
    fig2.savefig('figures/circuit_wigner.png', dpi=400)

plt.show()