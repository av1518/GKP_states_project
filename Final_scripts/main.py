#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from qutip import wigner, Qobj, fidelity
import matplotlib as mpl
from matplotlib import cm
from cutoff_opt import q_state, optimal_cutoff

n = 4
s = 0.34269262263253114
a = 9.45275091223988
dB_target = 16.29953698342939
prob_threshold = 0.9

st = q_state(a, dB_target)
opt = optimal_cutoff(prob_threshold)
cutoff = opt.min_cutoff(st)
target_dm = st.target_d_matrix(cutoff)
x = st.model_gkp()[0]
gk = st.model_gkp()[1]

plt.figure()
plt.plot(x,gk**2, label = f'{dB_target}dB')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

xvec = np.linspace(-6,6, 800)
Wp = wigner(Qobj(target_dm), xvec, xvec)
sc = np.max(Wp)
nrm = mpl.colors.Normalize(-sc, sc)
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
plt1 = axes.contourf(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
axes.contour(xvec, xvec, Wp, 60,  cmap=cm.RdBu, norm=nrm)
fig.colorbar(plt1, ax=axes)
fig.tight_layout()

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

if n > 1:
    for i in range(n-1):
        prog2 = sf.Program(2)

        with prog2.context as q:
            ops.DensityMatrix(dm) | q[0]
            ops.DensityMatrix(dm) | q[1]
            ops.BSgate() | (q[0],q[1])
            ops.MeasureHomodyne(np.pi/2,select=0.0) | q[1]

        eng2 = sf.Engine(backend="fock", backend_options={"cutoff_dim":cutoff})
        state = eng2.run(prog2).state
        dm = state.reduced_dm(0)
        
# plt.figure()
# plt.plot(x, state.x_quad_values(1,x,x), label = f'a = {a}')
# plt.xlabel('x')
# plt.ylabel('Probability')
# plt.legend()

trace = 0.5*np.trace(np.abs(dm - target_dm))
fid = fidelity(Qobj(target_dm), Qobj(dm))
print('Trace distance = ', trace)
print('Fidelity = ', fid)

Wp2 = wigner(Qobj(dm), xvec, xvec)
sc2 = np.max(Wp2)
nrm2 = mpl.colors.Normalize(-sc2, sc2)
fig2, axes2 = plt.subplots(1, 1, figsize=(5, 4))
plt2 = axes2.contourf(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
axes2.contour(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
fig2.colorbar(plt2, ax=axes2)
fig2.tight_layout()

plt.show()