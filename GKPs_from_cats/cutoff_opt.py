import numpy as np
import scipy.integrate as integrate
import numpy.polynomial.hermite as Herm
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as op

m=1.
w=1.
hbar=1.

cutoff_list = np.arange(2,300,1)
prob_threshold = 0.99

eps = np.finfo(float).eps

def gauss(x,sigma,mu):
    a = 1/(np.sqrt(2*np.pi)*sigma)
    y = a*np.exp(-0.5*((x-mu)**2)/sigma**2)
    return y

def model_gkp(delta_dB):
    delta = 10**(-delta_dB/20)
    lim = 7/delta
    alpha = 2*np.sqrt(np.pi) #separation between peaks
    n_p = round(2*lim/alpha)
    if n_p%2 == 0:
        n_p -=1
    xs = []
    for i in range(int(-(n_p-1)/2),0):
        x_neg = alpha*(i)
        xs.append(x_neg)
    for i in range(0,int((n_p+1)/2)):
        x_pos = alpha*(i)
        xs.append(x_pos)
    xs = np.array(xs)
    q = np.linspace(-lim,lim, 300*n_p)
    for i in range(0,n_p):
        if i==0:
            g = gauss(q,delta,xs[0])
        else:
            g += gauss(q,delta,xs[i])

    big_g = gauss(q,1/delta,0)
    g = g*big_g
    norm = 1/np.sqrt(integrate.trapz(np.absolute(g)**2, q))
    gkp = norm*g

    return q, gkp

def hermite(x, n):
    xi = np.sqrt(m*w/hbar)*x
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)
  
def hm_eigenstate(x,n):
    xi = np.sqrt(m*w/hbar)*x
    k = 1./math.sqrt(2.**n * math.factorial(n)) * (m*w/(np.pi*hbar))**(0.25)
    psi = k * np.exp(- xi**2 / 2) * hermite(x,n)
    return psi

def fock_coeffs(x,gkp_wf,cutoff):
    cs = []
    for i in range(cutoff):
        fi = np.conjugate(hm_eigenstate(x,i))*gkp_wf
        c_i = integrate.trapz(fi, x)
        cs.append(c_i)
    for i in range(cutoff):
        if abs(cs[i])<eps:
            cs[i]=0.0
    return np.array(cs)

def min_cutoff(delta):
    i = -1
    param = 0
    x = model_gkp(delta)[0]
    gk = model_gkp(delta)[1]

    if i < len(cutoff_list):
        while param == 0:
            i += 1
            cns = fock_coeffs(x,gk,cutoff_list[i])
            fock_probs = cns**2
            suma = sum(fock_probs)
            if suma >= prob_threshold:
                param = 1
    
    return int(cutoff_list[i])

def exp_fit(x,a,b):
    y = a*np.exp(b*x)
    return y

# deltas = np.arange(6,20,0.5)
# mins = []

# for i in tqdm(range(len(deltas))):
#     mini = min_cutoff(deltas[i])
#     mins.append(mini)

# mins = np.array(mins)

# init_guess=[1.53,0.23]
# fit,fit_cov = op.curve_fit(exp_fit,deltas,mins,init_guess)
# perr = np.sqrt(np.diag(np.absolute(fit_cov)))
# print('Error of exponential fit:',perr)
# print(fit)
# d = np.arange(6,25,0.01)

# fig = plt.figure()
# plt.scatter(deltas, mins, marker = 'X', color = 'darkblue', label = f'Threshold = {prob_threshold}')
# plt.plot(d,exp_fit(d,fit[0],fit[1]), 'b-', label = 'Fit')
# plt.legend()
# plt.xlabel('Delta (dB)')
# plt.ylabel('Minimum cutoff')
# fig.tight_layout()

# plt.show()