#%%
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as op
import numpy as np
from cutoff_opt import optimal_cutoff, q_state

def exp_fit(x,a,b):
    y = a*np.exp(b*x)
    return y

dummy_a = 1.0
prob_threshold = 0.95
deltas = np.arange(6,20,0.5)
mins = []

for i in tqdm(range(len(deltas))):
    st = q_state(dummy_a, deltas[i])
    opt = optimal_cutoff(prob_threshold)
    mini = opt.gkp_min_cutoff(st)
    mins.append(mini)

mins = np.array(mins)

init_guess=[1.53,0.23]
fit,fit_cov = op.curve_fit(exp_fit,deltas,mins,init_guess)
perr = np.sqrt(np.diag(np.absolute(fit_cov)))
print('Error of exponential fit:',perr)
print(fit)
d = np.arange(6,25,0.01)

plt.figure()
plt.scatter(deltas, mins, marker = 'X', color = 'darkblue', label = f'Threshold = {prob_threshold}')
plt.plot(d,exp_fit(d,fit[0],fit[1]), 'b-', label = 'Fit')
plt.legend()
plt.xlabel('Delta (dB)')
plt.ylabel('Minimum cutoff')
plt.tight_layout()

#%%
dummy_dB = 7.0
aa = np.arange(1,15,0.5)
mins = []

for i in tqdm(range(len(aa))):
    st = q_state(aa[i], dummy_dB)
    opt = optimal_cutoff(prob_threshold)
    mini = opt.cat_min_cutoff(st)
    mins.append(mini)

mins = np.array(mins)

init_guess=[9.0,0.2]
fit,fit_cov = op.curve_fit(exp_fit,aa,mins,init_guess)
perr = np.sqrt(np.diag(np.absolute(fit_cov)))
print('Error of exponential fit:',perr)
print(fit)
d = np.arange(1,20,0.01)

plt.figure()
plt.scatter(aa, mins, marker = 'X', color = 'darkblue', label = f'Threshold = {prob_threshold}')
plt.plot(d,exp_fit(d,fit[0],fit[1]), 'b-', label = 'Fit')
plt.legend()
plt.xlabel('a')
plt.ylabel('Minimum cutoff')
plt.tight_layout()

plt.show()