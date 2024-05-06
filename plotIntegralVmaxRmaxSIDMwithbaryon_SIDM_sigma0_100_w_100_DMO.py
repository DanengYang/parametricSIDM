import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import math
import statistics
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from scipy.optimize import minimize, rosen, rosen_der
import matplotlib
from parametricCoredDZ import *

plt.style.use('classic')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 1.4 # border width
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.4
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1.4
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1.4
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.width'] = 1.4

plt.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.family'] = 'Times New Roman'

data = np.genfromtxt('output_SIDM_sigma0_100_w_100_baryon_n1000_integral_DMO.txt', names=True)

GG=4.30073*1e-6
h=0.7
def rhos2(Vmax,rs):
      return pow(Vmax/1.648/rs,2)/GG

#------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.set_xlabel(r"$V_{\rm max,DM}\rm \ (km/s)$",fontsize=20)
ax.set_ylabel(r"$R_{\rm max,DM}\rm \ (kpc)$",fontsize=20)

#-------------------------------------------

plt.scatter(data['vmax'],   data['rmax'],  s=32,c=data['tr'],cmap='coolwarm',linewidth=0.1)
plt.clim(0, 1.1)
clb=plt.colorbar()
clb.ax.tick_params(labelsize=18)
clb.set_label(r'$\tau=10~{\rm Gyr}/t_{c,b}$', labelpad=0, size=20)

ax.set_xlim([10, 1200])
ax.set_ylim([0.5, 2000])

ax.set_xscale('log')
ax.set_yscale('log')

ax.text(0.05, 0.9, r"SIDM+baryon with a flat HMF, integral", transform=ax.transAxes, fontsize=22,verticalalignment='bottom')
ax.text(0.05, 0.8, r"$\sigma_0/m=100\ {\rm cm^2/g},\ w=100\ \rm km/s$", transform=ax.transAxes, fontsize=18,verticalalignment='bottom')

fig.set_size_inches(9,8,forward='True')

plt.show()
fig.savefig('figs/fig_vmax_rmax_SIDM_baryon_flat_HMF_sigma0_100_w_100_integral_DMO.png', bbox_inches='tight')

