#%%
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as op
import numpy as np
from cutoff_opt import optimal_cutoff, q_state
import matplotlib as mpl

plot_params = {'axes.labelsize':30,
          'axes.titlesize':18,
          'font.size':27,
          'figure.figsize':[10,10],
          'xtick.major.size':30,
          'xtick.minor.size':20,
          'ytick.major.size': 30,
          'ytick.minor.size': 20,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix',
          'lines.linewidth': 2 }

mpl.rcParams.update(plot_params)

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

#%%
dummy_dB = 7.0
aa = np.arange(1,15,0.5)
mins1 = []

for i in tqdm(range(len(aa))):
    st = q_state(aa[i], dummy_dB)
    opt = optimal_cutoff(prob_threshold)
    mini = opt.cat_min_cutoff(st)
    mins1.append(mini)

mins1 = np.array(mins1)

init_guess=[9.0,0.2]
fit1,fit_cov1 = op.curve_fit(exp_fit,aa,mins1,init_guess)
perr1 = np.sqrt(np.diag(np.absolute(fit_cov1)))
print('Error of exponential fit:',perr1)
print(fit1)
d1 = np.arange(1,20,0.01)

with plt.style.context(['science']):
    fig, ax = plt.subplots(1,2,  figsize = (20,7), constrained_layout=True)
    ax[0].scatter(deltas, mins, marker = 'X', s=100, color = 'darkblue', label = f'Threshold = {prob_threshold}')
    ax[0].plot(d,exp_fit(d,fit[0],fit[1]), 'b-', label = 'Fit', linewidth = 2)
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel('dB')
    ax[0].set_ylabel('Minimum cutoff')

    ax[1].scatter(aa, mins1, marker = 'X', s=100, color = 'darkblue', label = f'Threshold = {prob_threshold}')
    ax[1].plot(d1,exp_fit(d1,fit1[0],fit1[1]), 'b-', label = 'Fit', linewidth = 2)
    ax[1].set_xlabel('a')
    ax[1].grid()
    ax[1].set_ylabel('Minimum cutoff')
    
    fig.savefig('figures/opt_cutoff_two_views_cut.png', dpi=400)

#plt.show()