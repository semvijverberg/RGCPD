# -*- coding: utf-8 -*-
"""Documentation about RGCPD"""
import sys, os, inspect
if 'win' in sys.platform and 'dar' not in sys.platform:
    sep = '\\' # Windows folder seperator
else:
    sep = '/' # Mac/Linux folder seperator

RGCPD_func = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
main_dir = sep.join(RGCPD_func.split(sep)[:-1])
assert main_dir.split(sep)[-1] == 'RGCPD', 'main dir is not RGCPD dir'
df_ana_dir = os.path.join(main_dir, 'df_analysis', 'df_analysis')
cluster_func = os.path.join(main_dir, 'clustering')
fc_dir = os.path.join(main_dir, 'forecasting')

if df_ana_dir not in sys.path:
    sys.path.append(df_ana_dir)
if main_dir not in sys.path:
    sys.path.append(main_dir)
if RGCPD_func not in sys.path:
    sys.path.append(RGCPD_func)
if cluster_func not in sys.path:
    sys.path.append(cluster_func)
if fc_dir not in sys.path:
    sys.path.append(fc_dir)



from class_RGCPD import RGCPD
from class_EOF import EOF
from class_BivariateMI import BivariateMI




__version__ = '0.1'

__author__ = 'Sem Vijverberg '
__email__ = 'sem.vijverberg@vu.nl'

