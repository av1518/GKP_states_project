#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops

cutoff = 30
xvec = np.linspace(-6,6, 800)

#%%
prog = sf.Program(2)
a = 2.0

with prog.context as q:
    ops.Catstate(a) | q[0]
    ops.Catstate(a) | q[1]
    ops.Rgate(np.pi/2) | q[0]
    ops.Rgate(np.pi/2) | q[1]
    ops.BSgate() | (q[0], q[1])
    ops.MeasureHomodyne(np.pi/2, select=0.0) | q[0]
    
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim":cutoff})
state = eng.run(prog).state
print(state)

plt.figure()
plt.plot(xvec, state.x_quad_values(1,xvec,xvec), label = f'a = {a}')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

#%%
prog = sf.Program(2)
a = 3.0

with prog.context as q:
    ops.Catstate(a) | q[0]
    ops.Catstate(a) | q[1]
    ops.Rgate(np.pi/2) | q[0]
    ops.Rgate(np.pi/2) | q[1]
    ops.BSgate() | (q[0], q[1])
    ops.MeasureHomodyne(np.pi/2, select=0.0) | q[0]
    
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim":cutoff})
state = eng.run(prog).state
print(state)

plt.figure()
plt.plot(xvec, state.x_quad_values(1,xvec,xvec), label = f'a = {a}')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

#%%
prog = sf.Program(2)
a = 4.0

with prog.context as q:
    ops.Catstate(a) | q[0]
    ops.Catstate(a) | q[1]
    ops.Rgate(np.pi/2) | q[0]
    ops.Rgate(np.pi/2) | q[1]
    ops.BSgate() | (q[0], q[1])
    ops.MeasureHomodyne(np.pi/2, select=0.0) | q[0]
    
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim":cutoff})
state = eng.run(prog).state
print(state)

plt.figure()
plt.plot(xvec, state.x_quad_values(1,xvec,xvec), label = f'a = {a}')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

#%%
prog = sf.Program(2)
a = 4.0
cutoff = 60

with prog.context as q:
    ops.Catstate(a) | q[0]
    ops.Catstate(a) | q[1]
    ops.Rgate(np.pi/2) | q[0]
    ops.Rgate(np.pi/2) | q[1]
    ops.BSgate() | (q[0], q[1])
    ops.MeasureHomodyne(np.pi/2, select=0.0) | q[0]
    
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim":cutoff})
state = eng.run(prog).state
print(state)

plt.figure()
plt.plot(xvec, state.x_quad_values(1,xvec,xvec), label = f'a = {a}')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

plt.show()