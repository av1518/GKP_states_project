#%%
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
import strawberryfields.ops as ops
from tqdm import tqdm
from cutoff_opt import optimal_cutoff, q_state
import matplotlib.pyplot as plt

#%%
iterations = 50
initial_temp = 0.1
final_temp = 0.0
temp_step = -0.01
prob_threshold = 0.95

a_i = 3.0
s_i = 0.4
dB_i = 7
n = 2

sigma_a = 0.2
sigma_s = 0.2
sigma_dB = 0.7

def newcost(a, s, dB):

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

    trace_dist = 0.5*np.trace(np.abs(dm - target_dm))

    eng.reset()
    if n > 1:
        eng2.reset()

    return trace_dist

def acceptancefunction(dE, T):
    
    if dE <= 0.0:
        return 1
    
    print(f'dE = {dE}')
    print(f' Probability of still accepting = ', np.exp(-1.0 * dE/T))
    
    if np.random.uniform(0,1,1) < np.exp(-1.0 * dE/T): 
        return 1
    return 0

def ProposalFunction(u,sigma_a,sigma_s, sigma_db):
    x = np.abs(np.random.normal(0.0, sigma_a) + u[0])
    y = np.abs(np.random.normal(0.0, sigma_s) + u[1])
    z = np.abs(np.random.normal(0.0, sigma_db) + u[2])

    if z > 6.5 and x > 1.2:
        return np.array([x,y,z])
    else:
        x = np.abs(sigma_a + u[0])
        z = np.abs(sigma_db + u[2])
        return np.array([x,y,z])

def runchain3d(f, a, s, db, sigma_a, sigma_s, sigma_db):

    temps = np.arange(initial_temp,final_temp,temp_step)
    values = [a,s,db]
    
    listofvalues = []
    cost_progress = []
    
    for T in temps:
        zeros = np.zeros((iterations+1,len(values)))
        zeros[0,:] = values 

        for j in range(iterations):
            params_now = zeros[j,:]
            print('a:', params_now[0], 's:', params_now[1], 'dB:', params_now[2])
            params_next = ProposalFunction(params_now, sigma_a, sigma_s, sigma_db)
            while params_next[0] < 0:
                newchoice =  ProposalFunction(params_now, sigma_a, sigma_s, sigma_db)
                params_next[0] = newchoice[0]
            while params_next[1] < 0:
                newchoice =  ProposalFunction(params_now, sigma_a, sigma_s, sigma_db)
                params_next[1] = newchoice[1]
            while params_next[2] < 0 or params_next[2] >16:
                newchoice =  ProposalFunction(params_now, sigma_a, sigma_s, sigma_db)
                params_next[2] = newchoice[2]
            cost_now = f(params_now[0],params_now[1], params_now[2])
            
            cost_next = f(params_next[0],params_next[1], params_next[2])
            dE = cost_next - cost_now
            
            acceptstep = acceptancefunction(dE,T)
            if acceptstep == 1:
                zeros[j+1,:] = params_next
            if acceptstep == 0:
                zeros[j+1,:] = params_now

            print('Cost:', cost_now)
            cost_progress.append(cost_now)
        values = zeros[-1,:]
        listofvalues.append(zeros)
        
    return listofvalues, cost_progress

t,cost_progress = runchain3d(newcost, a_i, s_i, dB_i, sigma_a, sigma_s, sigma_dB)
np.save('anneal', t)
np.save('cost_prog', cost_progress)

#%%
t = np.load('anneal.npy',allow_pickle=True)
cost_progress = np.load('cost_prog.npy', allow_pickle=True)

lists = []
for i in range(len(t)):
    lists.append(t[i])

anneal= np.vstack((lists))

a_vals = anneal[:,0]
plt.figure()
plt.plot(np.arange(0,len(anneal),1),anneal[:,0], color = 'grey')
plt.ylabel(r'$a$')
plt.xlabel('Iteration number')
plt.tight_layout()

s_vals = anneal[:,1]
plt.figure()
plt.plot(np.arange(0,len(anneal),1),anneal[:,1], color = 'grey')
plt.ylabel(r'$s$')
plt.xlabel('Iteration number')
plt.tight_layout()

dB_vals = anneal[:,2]
plt.figure()
plt.plot(np.arange(0,len(anneal),1),anneal[:,2], color = 'grey')
plt.ylabel(r'$dB$')
plt.xlabel('Iteration number')
plt.tight_layout()

plt.figure()
plt.plot(range(len(cost_progress)), cost_progress, color = 'grey')
plt.xlabel('Iteration number')
plt.ylabel('Cost Progress')
plt.tight_layout()

optimal_i = np.argmin(cost_progress)
optimal_val = [a_vals[optimal_i],s_vals[optimal_i], dB_vals[optimal_i]]

print(f'Optimal parameters = {optimal_val}')
print('Optimal cost =', np.min(cost_progress))

plt.show()