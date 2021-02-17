#!/usr/bin/env python
# coding: utf-8

# # Clustering



import os, inspect, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
print(main_dir)
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering')
if RGCPD_func not in sys.path:
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(main_dir)

import clustering_spatial as cl
from RGCPD import RGCPD
import plot_maps
rg = RGCPD()

rg.pp_precursors()

rg.list_precur_pp


var_filename = rg.list_precur_pp[0][1]
mask = [145.0, 230.0, 20.0, 50.0]
for q in [85, 95]:
    xrclustered, results = cl.dendogram_clustering(var_filename, mask=mask, kwrgs_clust={'q':q, 'n_clusters':3})
    plot_maps.plot_labels(xrclustered)

