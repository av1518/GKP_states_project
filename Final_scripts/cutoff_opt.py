#%%
import numpy as np
import scipy.integrate as integrate
import numpy.polynomial.hermite as Herm
import math

class q_state():
    '''
    Input a and dB values
    '''
    def __init__(self, a, dB):
        self._a = a
        self._dB = dB

    def gauss(self,x,sigma,mu):
        a = 1/((np.sqrt(2*np.pi))*sigma)
        y = a*np.exp(-0.5*((x-mu)**2)/(sigma**2))
        return y

    def model_gkp(self):
        delta = 10**(-self._dB/20)
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
                g = self.gauss(q,delta,xs[0])
            else:
                g += self.gauss(q,delta,xs[i])

        big_g = self.gauss(q,1/delta,0)
        g = g*big_g
        norm = 1/np.sqrt(integrate.simpson(np.absolute(g)**2, q))
        gkp = norm*g

        return q, gkp
    
    def model_cat(self):
        delta = 1
        lim = self._a + 5*delta
        q = np.linspace(-lim,lim, int(400*self._a))
        g = self.gauss(q, 1, -self._a) + self.gauss(q, 1, self._a)
        norm = 1/np.sqrt(integrate.simpson(np.absolute(g)**2, q))
        cat = norm*g

        return q, cat
    
    m=1.
    w=1.
    hbar=1.
    eps = np.finfo(float).eps

    def hermite(self, x, n):
        xi = np.sqrt(self.m*self.w/self.hbar)*x
        herm_coeffs = np.zeros(n+1)
        herm_coeffs[n] = 1
        return Herm.hermval(xi, herm_coeffs)
    
    def hm_eigenstate(self,x,n):
        xi = np.sqrt(self.m*self.w/self.hbar)*x
        k = 1./math.sqrt(2.**n * math.factorial(n)) * (self.m*self.w/(np.pi*self.hbar))**(0.25)
        psi = k * np.exp(- xi**2 / 2) * self.hermite(x,n)
        return psi

    def fock_coeffs(self,x,gkp_wf,cutoff):
        cs = []
        for i in range(cutoff):
            fi = np.conjugate(self.hm_eigenstate(x,i))*gkp_wf
            c_i = integrate.simpson(fi, x)
            cs.append(c_i)
        for i in range(cutoff):
            if abs(cs[i])<self.eps:
                cs[i]=0.0
        return np.array(cs)
    
    def adj(self,v):
        adjo = []
        for i in range(len(v)):
            adjo.append([v[i]])
        return np.array(adjo)
    
    def target_d_matrix(self, cutoff):
        x = self.model_gkp()[0]
        gk = self.model_gkp()[1]
        cns = self.fock_coeffs(x,gk,cutoff)
        ad_cns = self.adj(cns)
        target_dm = ad_cns*cns/np.trace(ad_cns*cns)
        return target_dm


class optimal_cutoff():
    """
    Input the probability threshold value
    """
    def __init__(self, thres):
        self._thres = thres
    
    cutoff_list = np.arange(2,300,1)
    cut_threshold = 13

    def gkp_min_cutoff(self, state):
        i = -1
        param = 0
        x = state.model_gkp()[0]
        gk = state.model_gkp()[1]

        if i < len(self.cutoff_list):
            while param == 0:
                i += 1
                cns = state.fock_coeffs(x,gk,self.cutoff_list[i])
                fock_probs = cns**2
                suma = sum(fock_probs)
                if suma >= self._thres:
                    param = 1
        
        return int(self.cutoff_list[i])

    def cat_min_cutoff(self, state):
        i = -1
        param = 0
        x = state.model_cat()[0]
        cat = state.model_cat()[1]

        if i < len(self.cutoff_list):
            while param == 0:
                i += 1
                cns = state.fock_coeffs(x,cat,self.cutoff_list[i])
                fock_probs = cns**2
                suma = sum(fock_probs)
                if suma >= self._thres:
                    param = 1
        
        return int(self.cutoff_list[i])

    def min_cutoff(self,state):
        min1 = self.cat_min_cutoff(state)
        min2 = self.gkp_min_cutoff(state)
        if min1 > min2:
            mini = min1
        else:
            mini = min2
        if mini > self.cut_threshold:
            return mini
        else:
            return self.cut_threshold

