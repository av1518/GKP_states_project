
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy.polynomial.hermite as Herm
import math

m=1.
w=1.
hbar=1.

dB = 24
cutoff = 15

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

x = model_gkp(dB)[0]
gk = model_gkp(dB)[1]

plt.figure()
plt.plot(x,gk, label = f'{dB}dB')
plt.xlabel('q')
plt.ylabel('psi(q)')
plt.legend()

# %%

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

cns = fock_coeffs(x,gk,cutoff)
fock_range = np.arange(0,cutoff,1)

# plt.figure()
# plt.bar(fock_range,cns)
# plt.xlabel('Number basis')
# plt.ylabel('Fock amplitudes')

plt.figure()
plt.bar(fock_range,(np.absolute(cns))**2)
plt.xlabel('Number basis')
plt.ylabel('Fock probabilities')

def adj(v):
    adjo = []
    for i in range(len(v)):
        adjo.append([v[i]])
    return np.array(adjo)

ad_cns = adj(cns)
dm_target = ad_cns*cns

#plt.show()