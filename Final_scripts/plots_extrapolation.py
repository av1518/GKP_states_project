#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import matplotlib as mpl
import matplotlib.transforms as mtransforms

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

'''
optimal value for fault tolerant = 23.53db
'''

dBs = [9.738497838814267, 12.625016407139775, 13.5, 15.723376569770377]
dberrors = [ 0.9372402553997963,0.6332863231816144,0.7393711910205082, 0.24922322864378932]
ns = np.array([1,2,3,4])
s = np.array([ 0.3414662590968012,0.6341353578786608, 0.3427212441660357, 0.34356460571222])
a = [3.396839454409047,5.841414593366296,6.696114053323244,9.524976161663647]
aerror = [0.3871588158677099,0.5532373007943082, 0.794590181985177,0.050363073061547005]

def linear_fit(x,m,c):
    y = m*x + c
    return y

init_guess=[1.0,0.5]
fit,fit_cov = op.curve_fit(linear_fit,ns,dBs,init_guess)
perr = np.sqrt(np.diag(np.absolute(fit_cov)))
d = np.arange(1, 8.5,0.01)

nstd = 1 # to draw 2-sigma intervals

m_up = fit[0] + nstd * perr[0]
h_up = fit[1] + nstd * perr[1]
m_dw = fit[0] - nstd * perr[0]
h_dw = fit[1] - nstd * perr[1]

fit_up = linear_fit(d,m_up,h_up)
fit_dw = linear_fit(d,m_dw,h_dw)

x_cor = 7
y_cor = 23.53 #linear_fit(x_cor,fit[0],fit[1])
x_line = [0, x_cor, x_cor]
y_line = [y_cor, y_cor, 8]
y_cor2 = (np.sqrt(2)**(x_cor-1))*np.sqrt(np.pi)*np.exp(-0.108*x_cor + 0.96)
y_line2 = [y_cor2, y_cor2, 0]
y_cor3 = (np.sqrt(2)**(x_cor-1))*np.sqrt(np.pi)*np.exp(0.6)
y_line3 = [y_cor3, y_cor3, 0]

with plt.style.context(['science']):
    fig, ax = plt.subplots(1, 3, figsize = (20,7), constrained_layout=True)

    ax[0].plot(ns, dBs, 'ko', markersize = 6)
    ax[0].plot(d, linear_fit(d,fit[0],fit[1]), color='darkblue', linestyle = '-', label = 'Fit', linewidth = 2)
    ax[0].plot(x_line, y_line, linestyle = '--', color = 'grey', linewidth = 2)
    ax[0].fill_between(d, fit_up, fit_dw, color = 'c', alpha=.2, interpolate = True, label = r'$\sigma_{fit}$')
    ax[0].plot(7, 23.53, 'v', color= 'darkorange', markersize = 10)
    ax[0].autoscale(tight= True)
    ax[0].errorbar(ns, dBs, yerr = dberrors, linestyle = 'none', capsize= 8, color = 'black', label = r'$\sigma_{dB}$')
    ax[0].set_xticks(np.arange(1,np.max(d),2))
    ax[0].set_yticks(np.arange(8, 28, 3))
    ax[0].set_ylabel(r'$dB$', size = 30)
    ax[0].set_xlabel(r'$n$')
    ax[0].tick_params(which='both', width=1)
    ax[0].tick_params(which='major', length= 6)
    ax[0].tick_params(which='minor', length=3)
    ax[0].legend()

    ax[1].plot(ns, a, 'ko', markersize = 6)
    ax[1].plot(d, (np.sqrt(2)**(d-1))*np.sqrt(np.pi)*np.exp(0.6), color='darkblue', linestyle = '-', label = 'Prediction $s$ = 0.6', linewidth = 2)
    ax[1].plot(x_line, y_line3, linestyle = '--', color = 'grey', linewidth = 2)
    ax[1].plot(7, y_cor3, 'v', color= 'darkorange', markersize = 10)
    ax[1].autoscale(tight= True)
    ax[1].errorbar(ns, a, yerr = aerror, linestyle = 'none', capsize= 8, color = 'black', label = r'$\sigma_{a}$')
    ax[1].set_xticks(np.arange(1,np.max(d),2))
    ax[1].set_yticks(np.arange(0, 48, 8))
    ax[1].set_ylabel(r'$a$', size = 30)
    ax[1].set_xlabel(r'$n$')
    ax[1].tick_params(which='both', width=1)
    ax[1].tick_params(which='major', length= 6)
    ax[1].tick_params(which='minor', length=3)
    ax[1].legend(loc = 'upper left')

    ax[2].plot(ns, a, 'ko', markersize = 6)
    ax[2].plot(d, (np.sqrt(2)**(d-1))*np.sqrt(np.pi)*np.exp(-0.108*d+0.96), color='darkblue', linestyle = '-', label = r'Prediction $\downarrow$ s', linewidth = 2)
    ax[2].plot(x_line, y_line2, linestyle = '--', color = 'grey', linewidth = 2)
    ax[2].plot(7, y_cor2, 'v', color= 'darkorange', markersize = 10)
    ax[2].autoscale(tight= True)
    ax[2].errorbar(ns, a, yerr = aerror, linestyle = 'none', capsize= 8, color = 'black', label = r'$\sigma_{a}$')
    ax[2].set_xticks(np.arange(1,np.max(d),2))
    ax[2].set_yticks(np.arange(0, 24, 4))
    ax[2].set_ylabel(r'$a$', size = 30)
    ax[2].set_xlabel(r'$n$')
    ax[2].tick_params(which='both', width=1)
    ax[2].tick_params(which='major', length= 6)
    ax[2].tick_params(which='minor', length=3)
    ax[2].legend(loc = 'upper left')

    labels = ['a)', 'b)', 'c)']
    for j in range(len(ax)):
    # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax[j].text(0.0, 1.0, labels[j], transform=ax[j].transAxes + trans,
                fontsize='medium', va='bottom', fontfamily='serif')

    fig.savefig('figures/three_extrapolation.png', dpi=400)
    
plt.show()