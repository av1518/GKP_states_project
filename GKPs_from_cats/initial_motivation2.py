#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from gkp_target import target_dm

cutoff = 30
xvec = np.linspace(-12,12, 800)

#%%
prog = sf.Program(2)
s = 0.611#0.4
a = 3.278#2*np.sqrt(np.pi)*np.exp(s)

with prog.context as q:
    ops.Catstate(a) | q[0]
    ops.Catstate(a) | q[1]
    ops.Sgate(s) | q[0]
    ops.Sgate(s) | q[1]
    ops.BSgate() | (q[0], q[1])
    ops.MeasureHomodyne(np.pi/2, select=0.0) | q[0]
    
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim":cutoff})
state = eng.run(prog).state
dm = state.reduced_dm(1)

trace = 0.5*np.trace(np.absolute(dm - target_dm))
print(trace)

plt.figure()
plt.plot(xvec, state.x_quad_values(1,xvec,xvec), label = f'a = {a}')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

prog2 = sf.Program(2)

with prog2.context as q:
    ops.DensityMatrix(dm) | q[0]
    ops.DensityMatrix(dm) | q[1]
    ops.BSgate() | (q[0],q[1])
    ops.MeasureHomodyne(np.pi/2,select=0.0) | q[1]

eng2 = sf.Engine(backend="tf", backend_options={"cutoff_dim":cutoff})
state2 = eng2.run(prog2).state
dm2 = state2.reduced_dm(0)

plt.figure()
plt.plot(xvec, state2.x_quad_values(0,xvec,xvec), label = f'a = {a}')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

prog3 = sf.Program(2)

with prog3.context as q:
    ops.DensityMatrix(dm2) | q[0]
    ops.DensityMatrix(dm2) | q[1]
    ops.BSgate() | (q[0],q[1])
    ops.MeasureHomodyne(np.pi/2,select=0.0) | q[1]

eng3 = sf.Engine(backend="tf", backend_options={"cutoff_dim":cutoff})
state3 = eng3.run(prog3).state
dm3 = state3.reduced_dm(0)

plt.figure()
plt.plot(xvec, state3.x_quad_values(0,xvec,xvec), label = f'a = {a}')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

plt.show()