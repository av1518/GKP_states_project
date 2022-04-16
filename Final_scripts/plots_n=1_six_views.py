import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import cm
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
#%%
dir1 = 'b_opt_data/logs_n=1_thres0.9_500it.json'
dir2 = 'b_opt_data/logs_n=2_thres0.9_500it.json'
dir3 = 'b_opt_data/logs_n=3_thres0.9_500it.json'
dir4 = 'b_opt_data/logs_n=4_thres0.9_500it.json'
dir5 = 'b_opt_data/logs_n=1_thres0.95_500it.json'
dir6 = 'b_opt_data/logs_n=1_thres0.99_500it.json'
dir7 = 'b_opt_data/logs_n=1_thres0.997_500it.json'

dir_list = [dir1,dir5,dir7]

#%%
with plt.style.context(['science']):

    fig2, axes2 = plt.subplots(3, 3, figsize = (21,13), constrained_layout=True)
    
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

        fid_threshold = 0.995
        for i in range(len(cost98)):
            if cost98[i] < fid_threshold:
                cost98[i] = 0
            else:
                cost98indices.append(i)

        bestcost = [cost_prog[i] for i in cost98indices]
        besta = [a_prog[i] for i in cost98indices]
        bests = [s_prog[i] for i in cost98indices]
        bestdb = [dB_prog[i] for i in cost98indices]

        nrm = mpl.colors.Normalize(np.min(cost_prog), np.max(cost_prog))
        n = 1
        s_arr = np.arange(np.min(s_prog), np.max(s_prog), 0.005)
        a_arr = (np.sqrt(2)**(n-1))*np.sqrt(np.pi)*np.exp(s_arr)

        a_lis = []
        s_lis = []
        for i in range(len(a_arr)):
            if a_arr[i] < np.max(a_prog) and a_arr[i]>np.min(a_prog):
                a_lis.append(a_arr[i])
                s_lis.append(s_arr[i])

        a_lis = np.array(a_lis)
        s_lis = np.array(s_lis)

        img = axes2[j,0].scatter(a_prog, s_prog, c=cost_prog, cmap=cm.coolwarm, norm=nrm, s = 25)
        plt1 = axes2[j,0].plot(a_lis, s_lis, 'k-')
        img2 = axes2[j,1].scatter(s_prog, dB_prog, c=cost_prog, cmap=cm.coolwarm, norm=nrm, s = 25)
        img3 = axes2[j,2].scatter(a_prog, dB_prog, c=cost_prog, cmap=cm.coolwarm, norm=nrm, s = 25)
        imgerrors = axes2[j,0].scatter(besta, bests, color = 'fuchsia')
        img2errors =  axes2[j,1].scatter(bests, bestdb, color = 'fuchsia')
        img3errors = axes2[j,2].scatter(besta, bestdb, color = 'fuchsia', label = r'Fidelity $ > 0.997$ ')
        
        axes2[j,0].autoscale(tight=True)
        axes2[j,1].autoscale(tight=True)
        axes2[j,2].autoscale(tight=True)
        axes2[j,0].set_xlabel(r'$a$')
        axes2[j,0].set_ylabel(r'$s$')
        axes2[j,1].set_xlabel(r'$s$')
        axes2[j,1].set_ylabel(r'$dB$')
        axes2[j,2].set_xlabel(r'$a$')
        axes2[j,2].set_ylabel(r'$dB$')

    fig2.colorbar(img3, ax=axes2.ravel().tolist(), location = 'right', aspect=50)
    fig2.savefig('figures/n=1_six_views.png', dpi=400)

#plt.show()