#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from gkp_target import target_dm, cutoff
from qutip import wigner, Qobj, wigner_cmap
import matplotlib as mpl
from matplotlib import cm

xvec = np.linspace(-12,12, 800)

#%%

prog = sf.Program(2)
s = 0.611
a = 3.278 #np.sqrt(np.pi)*np.exp(s)
n = 2

s = 0.4
a = 2*np.sqrt(2)*np.sqrt(np.pi)*np.exp(s)

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

trace = 0.5*np.trace(np.absolute(dm - target_dm))
print('Initial trace distance =', trace)

plt.figure()
plt.plot(xvec, state.x_quad_values(1,xvec,xvec), label = f'a = {a}')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

if n != 0:
    for i in range(n):
        prog2 = sf.Program(2)

        with prog2.context as q:
            ops.DensityMatrix(dm) | q[0]
            ops.DensityMatrix(dm) | q[1]
            ops.BSgate() | (q[0],q[1])
            ops.MeasureHomodyne(np.pi/2,select=0.0) | q[1]

        eng2 = sf.Engine(backend="fock", backend_options={"cutoff_dim":cutoff})
        state2 = eng2.run(prog2).state
        dm = state2.reduced_dm(0)

        plt.figure()
        plt.plot(xvec, state2.x_quad_values(0,xvec,xvec), label = f'a = {a}')
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.legend()
        
        eng2.reset()

trace = 0.5*np.trace(np.abs(dm - target_dm))
print('Final trace distance =', trace)

xvec = np.linspace(-6,6, 800)
Wp2 = wigner(Qobj(dm), xvec, xvec)
wmap2 = wigner_cmap(Wp2)
sc12 = np.max(Wp2)
nrm2 = mpl.colors.Normalize(-sc12, sc12)
fig2, axes2 = plt.subplots(1, 1, figsize=(5, 4))
plt2 = axes2.contourf(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
axes2.contour(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
cb12 = fig2.colorbar(plt2, ax=axes2)
fig2.tight_layout()

plt.show()
