#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields import ops
from qutip import wigner, Qobj, fidelity
import matplotlib as mpl
from matplotlib import cm
from cutoff_opt import q_state, optimal_cutoff
from tqdm import tqdm

mpl.rcParams.update(mpl.rcParamsDefault)

n = 3
s = 0.3427212441660357
a = 6.696114053323244
dB_target = 13.5
prob_threshold = 0.9
iterations = 500

# n = 4
# s = 0.3559898602842482
# a = 9.590231285601375
# dB_target = 15.723376569770377
# prob_threshold = 0.9

st = q_state(a, dB_target)
opt = optimal_cutoff(prob_threshold)
cutoff = opt.min_cutoff(st)
target_dm = st.target_d_matrix(cutoff)

xvec = np.linspace(-6,6, 800)
    
fids = []

for j in tqdm(range(iterations)):
    prog = sf.Program(2)

    with prog.context as q:
        ops.Catstate(a) | q[0]
        ops.Catstate(a) | q[1]
        ops.Sgate(s) | q[0]
        ops.Sgate(s) | q[1]
        ops.BSgate() | (q[0], q[1])
        ops.MeasureHomodyne(np.pi/2) | q[0]
        
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim":cutoff})
    state = eng.run(prog).state
    dm = state.reduced_dm(1)

    # xs = np.linspace(-30,30, 800)

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
                ops.MeasureHomodyne(np.pi/2) | q[1]

            eng2 = sf.Engine(backend="fock", backend_options={"cutoff_dim":cutoff})
            state2 = eng2.run(prog2).state
            dm = state2.reduced_dm(0)

            # xs = np.linspace(-30,30, 800)

            # with plt.style.context(['science']):
            #     fig, ax = plt.subplots(figsize = (5,4))
            #     ax.plot(xs, state2.x_quad_values(0,xs,xs), label = f'a = {a:.2f}', color = 'darkblue')
            #     ax.legend()
            #     ax.autoscale()
            #     ax.set_ylabel(r'Probability')
            #     ax.set_xlabel(r'$x$')
            #     fig.savefig(f'figures/step{i+2}.png', dpi=400)

    fid = fidelity(Qobj(target_dm), Qobj(dm))
    fids.append(fid)
    print('Fidelity = ', fid)

    # with plt.style.context(['science']):
    #     Wp2 = wigner(Qobj(dm), xvec, xvec)
    #     sc2 = np.max(Wp2)
    #     nrm2 = mpl.colors.Normalize(-sc2, sc2)
    #     fig2, axes2 = plt.subplots(1, 1, figsize=(5, 4))
    #     plt2 = axes2.contourf(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
    #     axes2.contour(xvec, xvec, Wp2, 60,  cmap=cm.RdBu, norm=nrm2)
    #     fig2.colorbar(plt2, ax=axes2)
    #     axes2.autoscale(tight=True)
    #     axes2.set_ylabel(r'$p$')
    #     axes2.set_xlabel(r'$x$')
    #     fig2.savefig('figures/circuit_wigner.png', dpi=400)

fids = np.array(fids)
np.save('fide', fids)

import numpy as np
import matplotlib.pyplot as plt
fid_progress = np.load('fide_n=3_500it_random_meas.npy', allow_pickle=True)

with plt.style.context(['science']):
    plt.figure()
    plt.hist(fid_progress, bins=50, color = 'darkblue')
    plt.ylabel('Counts')
    plt.xlabel('Fidelity')
    plt.savefig('figures/meas_hist.png', dpi=400)

high_fids = []
for i in range(len(fid_progress)):
    if fid_progress[i] > 0.8:
        high_fids.append(fid_progress[i])

print(len(high_fids))

plt.show()
