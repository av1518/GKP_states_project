# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 13:57:13 2022

@author: gefae
"""

import strawberryfields as sf
from strawberryfields import ops
import numpy as np
import matplotlib.pyplot as plt

#%%

# create a 3-mode quantum program
prog = sf.Program(3)

with prog.context as q:
    ops.Sgate(0.54) | q[0]
    ops.Sgate(0.54) | q[1]
    ops.Sgate(0.54) | q[2]
    ops.BSgate(0.43, 0.1) | (q[0], q[2])
    ops.BSgate(0.43, 0.1) | (q[1], q[2])
    ops.MeasureFock() | q
    
#prog.print()

# initialize the fock backend with a
# Fock cutoff dimension (truncation) of 5
eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
result = eng.run(prog)
print(result.samples)

#%%

res1 = []
res2 = []
res = []

for i in range(0,10000):
    prog = sf.Program(2)
    
    with prog.context as q:
        ops.Squeezed(0.54, np.pi/4) | q[0]
        ops.Squeezed(0.54, 2.1) | q[1]
        ops.BSgate(0.43, 0.1) | (q[0], q[1])
        ops.MeasureFock() | q
        
    #prog.print()
    
    # initialize the fock backend with a
    # Fock cutoff dimension (truncation) of 5
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
    result = eng.run(prog)
    res1.append(result.samples[0][0])
    res2.append(result.samples[0][1])
    res.append(result.samples[0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

hist, xedges, yedges = np.histogram2d(res1, res2, bins=5, range=[[0, 5], [0, 5]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()

plt.figure()
plt.plot(res1,res2,'ro')
plt.show()

#%%

res1 = []
res2 = []
res = []

for i in range(0,10000):
    prog = sf.Program(2)
    
    with prog.context as q:
        ops.Squeezed(0.1, np.pi/4) | q[0]
        ops.Squeezed(0.9, 5.1) | q[1]
        ops.BSgate(0.9, 0.1) | (q[0], q[1])
        ops.MeasureFock() | q
        
    #prog.print()
    
    # initialize the fock backend with a
    # Fock cutoff dimension (truncation) of 5
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
    result = eng.run(prog)
    res1.append(result.samples[0][0])
    res2.append(result.samples[0][1])
    res.append(result.samples[0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

hist, xedges, yedges = np.histogram2d(res1, res2, bins=5, range=[[0, 5], [0, 5]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()

plt.figure()
plt.plot(res1,res2,'ro')
plt.show()
















