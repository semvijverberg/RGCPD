# -*- coding: utf-8 -*-
"""Documentation about RGCPD"""
import sys, os, inspect
RGCPD_func = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
main_dir = '/'.join(RGCPD_func.split('/')[:-1])
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
df_ana_dir = os.path.join(main_dir, 'df_analysis/')
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')

if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)
    sys.path.append(df_ana_dir)


# sys.path.append('./forecasting')
from class_RGCPD import RGCPD
# from func_fc import fcev
from class_EOF import EOF
from class_BivariateMI import BivariateMI
from df_ana_class import DFA




__version__ = '0.1'

__author__ = 'Sem Vijverberg '
__email__ = 'sem.vijverberg@vu.nl'

