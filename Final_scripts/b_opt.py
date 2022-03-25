#%%
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
import strawberryfields.ops as ops
from cutoff_opt import q_state, optimal_cutoff
import matplotlib.pyplot as plt
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import json
import matplotlib as mpl
from matplotlib import cm
from qutip import Qobj, fidelity

n = 4
prob_threshold = 0.9
iterations = 500
explore = int(iterations*0.2)

def bayesian_opt(function, pbounds):
    """
    runs function to find optimal parameters

    output:
        result : dict
            for example, {'target': 123, 'params': {'x': 123, 'y': 123}}
            where target = function(params)
    """
    print('====== start bayesian optimisation ======')
    optimizer = BayesianOptimization(function, pbounds, verbose=2, random_state=1,)
    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points = explore, n_iter = iterations,)
    result = optimizer.max
    print('====== end bayesian optimisation ======')

    return result

def cost(a, s, dB):

    prog = sf.Program(2)

    st = q_state(a, dB)
    opt = optimal_cutoff(prob_threshold)
    cutoff = opt.min_cutoff(st)
    target_dm = st.target_d_matrix(cutoff)
    
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
            state2 = eng2.run(prog2).state
            dm = state2.reduced_dm(0)

    #trace_dist = 0.5*np.trace(np.abs(dm - target_dm))
    fid = fidelity(Qobj(target_dm),Qobj(dm))

    eng.reset()
    if n > 1:
        eng2.reset()

    return fid

pbounds1 = {'a': (1.5, 5.0), 's': (0.1,1.1), 'dB': (7.5, 11.0)}
pbounds2 = {'a': (3.5, 7.5), 's': (0.1,1.3), 'dB': (10.5, 13.5)}
pbounds3 = {'a': (4.5, 9.0), 's': (0.1,2.1), 'dB': (13.5, 16.5)}
pbounds4 = {'a': (5.5, 10.0), 's': (0.1,2.5), 'dB': (15.0, 17.0)}

if n==1:
    soltn = bayesian_opt(cost, pbounds1)
    print(soltn)

if n==2:
    soltn = bayesian_opt(cost, pbounds2)
    print(soltn)

if n==3:
    soltn = bayesian_opt(cost, pbounds3)
    print(soltn)

if n==4:
    soltn = bayesian_opt(cost, pbounds4)
    print(soltn)

data = []
for line in open('logs.json', 'r'):
    data.append(json.loads(line))

cost_prog = []
for i in data:
    cost_prog.append(abs(i['target']))

a_prog = []
for i in data:
    a_prog.append(i['params']['a'])

a_vals = np.sort(a_prog)

s_prog = []
for i in data:
    s_prog.append(i['params']['s'])

dB_prog = []
for i in data:
    dB_prog.append(i['params']['dB'])

from scipy.interpolate import griddata
xi = np.linspace(np.min(a_prog), np.max(a_prog), 100)
yi = np.linspace(np.min(s_prog), np.max(s_prog), 100)
zi = griddata((a_prog, s_prog), cost_prog, (xi[None, :], yi[:, None]),
                method='linear')

nrm = mpl.colors.Normalize(np.min(cost_prog), np.max(cost_prog))
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
plt1 = axes.contourf(xi, yi, zi, 100,  cmap=cm.coolwarm, norm=nrm)
axes.contour(xi, yi, zi, 100,  cmap=cm.coolwarm, norm=nrm)
axes.plot(a_prog,s_prog, 'ko', markersize = 2.0)
axes.set_xlabel('a')
axes.set_ylabel('s')
fig.colorbar(plt1, ax=axes)
fig.tight_layout()

# data = []
# for line in open('Figures_and_data/bayes_opt/n=4_100it/logs_n=4_200it.json', 'r'):
#     data.append(json.loads(line))

# cost_prog = []
# for i in data:
#     cost_prog.append(abs(i['target']))

# a_prog = []
# for i in data:
#     a_prog.append(i['params']['a'])

# a_vals = np.sort(a_prog)

# s_prog = []
# for i in data:
#     s_prog.append(i['params']['s'])

# dB_prog = []
# for i in data:
#     dB_prog.append(i['params']['dB'])


fig2 = plt.figure()
nrm = mpl.colors.Normalize(np.min(cost_prog), np.max(cost_prog))
ax2 = fig2.add_subplot(111, projection='3d')
img = ax2.scatter(a_prog, s_prog, dB_prog, c=cost_prog, cmap=cm.coolwarm, norm=nrm)
plt.colorbar(img,shrink=0.7)
plt.yticks(np.linspace(np.min(s_prog),np.max(s_prog),6))
ax2.set_xlabel(r'$a$')
ax2.set_ylabel(r'$s$')
ax2.set_zlabel(r"$dB$")

plt.show()