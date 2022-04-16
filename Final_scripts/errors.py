
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from qutip import Qobj, fidelity
from cutoff_opt import q_state, optimal_cutoff

#%%
dir1 = 'b_opt_data/logs_n=1_thres0.9_500it.json'
dir2 = 'b_opt_data/logs_n=2_thres0.9_500it.json'
dir3 = 'b_opt_data/logs_n=3_thres0.9_500it.json'
dir4 = 'b_opt_data/logs_n=4_thres0.9_500it.json'

dir_list = [dir1,dir2,dir3,dir4]

fid_threshold = 0.9798

for j in range(len(dir_list)):
    data = []
    for line in open(dir_list[j], 'r'):
        data.append(json.loads(line))

    cost_prog = []
    for i in data:
        cost_prog.append(abs(i['target']))

    a_prog = []
    for i in data:
        a_prog.append(i['params']['a'])

    s_prog = []
    for i in data:
        s_prog.append(i['params']['s'])

    dB_prog = []
    for i in data:
        dB_prog.append(i['params']['dB'])

    cost98 = np.copy(cost_prog)
    cost98indices = []

    for i in range(len(cost98)):
        if cost98[i] < fid_threshold:
            cost98[i] = 0
        else:
            cost98indices.append(i)
    
    max_value = max(cost_prog)
    max_index = cost_prog.index(max_value)
    print(

    )
    print(f'n = {j+1}')
    print(f'max a = {a_prog[max_index]}')
    print(f'max s = {s_prog[max_index]}')
    print(f'max dB = {dB_prog[max_index]}')
    print(f'max fidelity = {max_value}')

    print(f'Number of values with fidelity > {fid_threshold} = ', len(cost98indices))

    cost_prog = [cost_prog[i] for i in cost98indices]
    a_prog = [a_prog[i] for i in cost98indices]
    s_prog = [s_prog[i] for i in cost98indices]
    dB_prog = [dB_prog[i] for i in cost98indices]

    abar = np.mean(a_prog)
    sbar = np.mean(s_prog)
    dbbar = np.mean(dB_prog)

    astd = np.std(a_prog)
    sstd = np.std(s_prog)
    dbstd = np.std(dB_prog)

    print(f'mean a = {abar} +/- {astd}')
    print(f'mean s = {sbar} +/- {sstd}')
    print(f'mean dB = {dbbar} +/- {dbstd}')

    n = j+1
    s = sbar
    a = abar
    dB_target = dbbar
    prob_threshold = 0.9

    st = q_state(a, dB_target)
    opt = optimal_cutoff(prob_threshold)
    cutoff = opt.min_cutoff(st)
    target_dm = st.target_d_matrix(cutoff)

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
            state2 = eng2.run(prog2).state
            dm = state2.reduced_dm(0)

    trace = 0.5*np.trace(np.abs(dm - target_dm))
    fid = fidelity(Qobj(target_dm), Qobj(dm))
    #print('trace distance = ', trace)
    print('new fidelity = ', fid)