# -*- coding: utf-8 -*-
"""Documentation about RGCPD"""
import sys, os
RGCPD_dir = './RGCPD'
sys.path.append(RGCPD_dir)
main_dir = '/'.join(RGCPD_dir.split('/')[:-1])
df_ana_dir = os.path.join(main_dir, 'df_analysis')
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

