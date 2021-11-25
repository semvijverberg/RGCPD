# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
user_dir = os.path.expanduser('~')
os.chdir(os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/paper_Raed/'))
curr_dir = os.path.join(user_dir, 'surfdrive/Scripts/RGCPD/RGCPD/')
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')

if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)
import functions_pp

path_output = os.path.join("/Users/semvijverberg/surfdrive/output_paper3/extra_plots_paper/")
path_input  = '/Users/semvijverberg/Desktop/cluster/surfdrive/output_paper3/USDA_Soy'

#%% Collect different splits continuous forecast

methods = ['random_5', 'random_10', 'random_20']
seeds = [1, 2, 3, 4]

orientation = 'horizontal'
alpha = .05
metrics_cols = ['corrcoef', 'MAE', 'RMSE']
rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
            'MAE':'MAE-SS', 'CRPSS':'CRPSS'}

cs = ["#a4110f","#f7911d","#fffc33","#9bcd37","#1790c4"]


month = 'April'

combinations = np.array(np.meshgrid(methods, seeds)).T.reshape(-1,2)
metrics = ['corrcoef', 'MAE', 'RMSE']
np_out = np.zeros( (len(metrics),combinations.shape[0], 4))
for i, (method, s) in enumerate(combinations):
    path = os.path.join(path_input, f'{method}')

    hash_str = f'cond_fc_{method}_s{s}.h5'
    f_name = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if re.findall(f'{hash_str}', file):
                print(f'Found file {file}')
                f_name = file


    if f_name is not None:
        d_dfs = functions_pp.load_hdf5(os.path.join(path,
                                                    f's{s}',
                                                    f_name))['df_cond_fc']
        np_out[0][i] = d_dfs.loc[metrics[0]].loc[month]
        np_out[1][i] = d_dfs.loc[metrics[1]].loc[month]
        np_out[2][i] = d_dfs.loc[metrics[2]].loc[month]

#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
save = False
df_cond_fc = pd.DataFrame(np_out.reshape((-1, np_out.shape[-1])),
                          index=pd.MultiIndex.from_product([metrics, [c[0]+'_'+c[1] for c in combinations]]),
                          columns=d_dfs.columns)
plot_cols = ['strong 50%', 'weak 50%']
rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
            'MAE':'MAE-SS', 'CRPSS':'CRPSS'}

seeds = [1,2,3,4]
seed_legend = False
if seed_legend:
    markers = ['*', '^', 's', 'o']
else:
    markers = ['s'] * len(seeds)
CV_legend = False
add_boxplot = False


metric = metrics[0]

f, axes = plt.subplots(1,len(metrics), figsize=(5*len(metrics_cols), 4),
                     sharex=True) ;

percentages = []
for iax, metric in enumerate(metrics):
    ax = axes[iax]
    # ax.set_facecolor('white')
    data = df_cond_fc.loc[metric][plot_cols]
    # data.plot(kind='box', ax=ax)
    CV_types = ['random_5', 'random_10', 'random_20']
    if CV_legend:
        colors = ["orange","yellow", "blue"]
    else:
        colors = ["grey"]*len(CV_types)
    perc_incr = (data[plot_cols[0]] - data[plot_cols[1]]) / data[plot_cols[1]]
    if add_boxplot:
        if metric == 'corrcoef':
            data['Factor increase'] = perc_incr
        else:
            data['Factor increase'] = 0.25*perc_incr
        nlabels = plot_cols.copy() ; nlabels.append('Factor increase')
        widths=(.5,.5,.5)
    else:
        nlabels = plot_cols.copy() ; widths=(.5,.5)

    ax.boxplot(data, labels=[l.replace(' ', '\n ').capitalize() for l in nlabels],
               widths=widths, whis=.95)


    for isd, seed in enumerate(seeds):
        data_s = data.loc[[i for i in data.index if int(i.split('_')[-1])==seed ]]
        for j,CV in enumerate(CV_types):
            for i,d in enumerate(data):

                y = data_s[d].loc[[i for i in data_s[d].index if CV in i]]
                x = np.random.normal(i+1, 0.04, len(y))
                ax.plot(x, y, color= colors[j], mec='k', ms=7,
                        marker=markers[isd],
                        linestyle="None", label=None, alpha=.2)

    if add_boxplot:
        ax2=ax.twinx()
        ax.fill_betweenx(y=[-.4,1], x1=2.5,x2=3.5, facecolor="pink", alpha=.6)
    else:
        ax.text(0.98, 0.98,f'+{int(100*perc_incr.mean(0))}%',
                horizontalalignment='right',
                verticalalignment='top',
                transform = ax.transAxes,
                fontsize=15)
    if metric == 'corrcoef':
        ax.set_ylim(0,1) ; steps = 1
        yticks = np.round(np.arange(0,1.01,.2), 2)
        ax.set_yticks(yticks[::steps])
        ax.set_yticks(yticks, minor=True)
        ax.tick_params(which='minor', length=0)
        ax.set_yticklabels(yticks[::steps])
        if add_boxplot:
            ax2.set_ylim(0, 1)
            ax2.set_yticks(yticks[::steps])
            ax2.set_yticks(yticks, minor=True)
            ax2.tick_params(which='minor', length=0)
            ax2.set_yticklabels(np.round(1+ yticks[::steps] , 1),
                                fontsize=15)
            ax2.grid(False)
    else:
        yticks = np.round(np.arange(0,.51,.1), 1)
        ax.set_ylim(0,.5) ; steps = 1
        ax.set_yticks(yticks[::steps])
        ax.set_yticks(yticks, minor=True)
        ax.tick_params(which='minor', length=0)
        ax.set_yticklabels(yticks[::steps])
        if add_boxplot:
            ax2.set_ylim(0, .5)
            ax2.set_yticks(yticks[::steps])
            ax2.set_yticks(yticks, minor=True)
            ax2.tick_params(which='minor', length=0)
            ax2.set_yticklabels(np.round(1 + yticks[::steps] * 4, 1),
                                fontsize=15)
            ax.axhline(y=0, color='black', linewidth=1)




    ax.tick_params(which='both', grid_ls='-', grid_lw=1,width=1,
                   labelsize=16, pad=6, color='black')
    ax.grid(which='both')
    ax.set_ylabel(rename_m[metric], fontsize=18, labelpad=2)

if CV_legend:
    CV_types = [CV.replace('random_','').capitalize()+'-fold CV' for CV in CV_types]

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], edgecolor='w',
                             label=CV_types[0]),
                       Patch(facecolor=colors[1], edgecolor='w',
                             label=CV_types[1]),
                       Patch(facecolor=colors[2], edgecolor='w',
                             label=CV_types[2])]
    axes[0].legend(handles=legend_elements, loc='lower left',
                   framealpha=1, facecolor='w', prop={'size': 12})
if seed_legend:
    import matplotlib.lines as mlines
    ms = 7
    s1 = mlines.Line2D([], [], color='black', marker=markers[0],
                       linestyle='None', ms=ms, label='Seed 1')
    s2 = mlines.Line2D([], [], color='black', marker=markers[1],
                       linestyle='None', ms=ms, label='Seed 2')
    s3 = mlines.Line2D([], [], color='black', marker=markers[2],
                       linestyle='None', ms=ms, label='Seed 3')
    s4 = mlines.Line2D([], [], color='black', marker=markers[3],
                       linestyle='None', ms=ms, label='Seed 4')

    axes[1].legend(handles=[s1, s2, s3, s4], loc='lower left',
                   framealpha=1, facecolor='w')




f.subplots_adjust(hspace=.1)
if add_boxplot:
    f.subplots_adjust(wspace=.5)
else:
    f.subplots_adjust(wspace=.3)
title = f'Improved skill during Strong Boundary Forcing [forecast in {month}]'
# additional statistics on distributions
# perc =


f.suptitle(title, y=1.0, fontsize=18)
f_name = f'Strong_weak_forcing_box_plots_{month}'
fig_path = os.path.join(path_output, f_name)+'.pdf'
if save:
    plt.savefig(fig_path, bbox_inches='tight')




# sns.stripplot(x='sex', y='age', data=dataset, jitter=True, hue='survived', split=True)
#         c1, c2 = '#3388BB', '#EE6666'
#         for i, m in enumerate(metrics_cols):
#             # normal SST

#             labels = d_dfs['df_scores'].columns.levels[0]
#             ax[i].plot(labels, d_dfs['df_scores'].reorder_levels((1,0), axis=1).loc[0][m].T,
#                     label=f'seed: {s}',
#                     color=cs[s],
#                     linestyle='solid')
#             ax[i].fill_between(labels,
#                                 d_dfs['df_boot'].reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
#                                 d_dfs['df_boot'].reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
#                                 edgecolor=cs[s], facecolor=cs[s], alpha=0.3,
#                                 linestyle='solid', linewidth=2)

#             if m == 'corrcoef':
#                 ax[i].set_ylim(-.2,1)
#             else:
#                 ax[i].set_ylim(-.2,.6)
#             ax[i].axhline(y=0, color='black', linewidth=1)
#             ax[i].tick_params(labelsize=16, pad=6)
#             if i == len(metrics_cols)-1 and orientation=='vertical':
#                 ax[i].set_xlabel('Forecast month', fontsize=18)
#             elif orientation=='horizontal':
#                 ax[i].set_xlabel('Forecast month', fontsize=18)
#             if i == 0:
#                 ax[i].legend(loc='lower right', fontsize=14)
#             ax[i].set_ylabel(rename_m[m], fontsize=18, labelpad=-4)


# f.subplots_adjust(hspace=.1)
# f.subplots_adjust(wspace=.22)
# title = 'Verification Soy Yield forecast'
# if orientation == 'vertical':
#     f.suptitle(title, y=.92, fontsize=18)
# else:
#     f.suptitle(title, y=.95, fontsize=18)
# f_name = f'{method}_{seed}_PacAtl_seeds'
# fig_path = os.path.join(path_output, f_name)+rg.figext
# if save:
#     plt.savefig(fig_path, bbox_inches='tight')