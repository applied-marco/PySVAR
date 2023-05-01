import scipy.io as spio
import numpy as np
import pandas as pd
from identification.VAR import VAR

# EX 1: estimate the VAR model
data = spio.loadmat('/Users/fangli/PySVAR/data/estimation_test.mat')
data = data['Y']
data = data[:100, [0, 6]]
names = ['Output', 'Inflation']
var = VAR(data=data, var_names=names, data_frequency='Q')
var.fit()
h = 20

# estimate the IRF and VD
irf = var.irf(h=h)
vd = var.vd(h=h)
var.bootstrap(h=h)

# tested
var.plot_irf(sigs=[68, 80], with_ci=True)
# var.plot_vd(save_path='/Users/fangli/PySVAR/graphs')


# EX2: replicate Kilian
oil = spio.loadmat('/Users/fangli/PySVAR/data/oil.mat')
oil = oil['data']
names = ['OilProd', 'REA', 'OilPrice']
shocks = ['Supply', 'Agg Demand', 'Specific Demand']
exm = VAR(data=oil, var_names=names, data_frequency='Q')
exm.fit()

# estimate the IRF and VD
h = 20
exm.irf(h=h)
exm.vd(h=h)
exm.bootstrap(h=h)
exm.irf_point_estimate = np.cumsum(exm.irf_point_estimate, axis=1)
exm.plot_irf(var_list=names, sigs=[68, 80], with_ci=False)

# EX3. oil price uncertainty shock
data = pd.read_csv('/Users/fangli/PySVAR/data/data_comp.csv')
data.drop(columns='Unnamed: 0', inplace=True)
var_names = ['VIX_avg', 'OVX_avg', 'WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory']
var = VAR(data=np.array(data[var_names]), var_names=var_names, data_frequency='Q')
var.fit()

var.irf(h=20)
var.vd(h=20)
var.bootstrap(h=20)
var.plot_irf(var_list=var_names, shock_list=[1], sigs=[68, 80], with_ci=True)
var.plot_vd(var_list=var.var_names, shock_list=[0, 1])
