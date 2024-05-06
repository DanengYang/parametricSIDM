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
import multiprocessing
from multiprocessing import Pool
#matplotlib.use('Agg')
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

hvmax = np.loadtxt('samples/tab_vmax_rmax_lognormal_flathmf_lognormalc_n1000.dat',usecols=0)
hrvmax =np.loadtxt('samples/tab_vmax_rmax_lognormal_flathmf_lognormalc_n1000.dat',usecols=1)
hcvir = np.loadtxt('samples/tab_vmax_rmax_lognormal_flathmf_lognormalc_n1000.dat',usecols=2)
hmvir = np.loadtxt('samples/tab_vmax_rmax_lognormal_flathmf_lognormalc_n1000.dat',usecols=3)
hrvir = np.loadtxt('samples/tab_vmax_rmax_lognormal_flathmf_lognormalc_n1000.dat',usecols=4)
hc200 = np.loadtxt('samples/tab_vmax_rmax_lognormal_flathmf_lognormalc_n1000.dat',usecols=7)

GG=4.30073*1e-6
h=0.7
def rhos2(Vmax,rs):
      return pow(Vmax/1.648/rs,2)/GG

def v(z):
    return np.exp(-4 * (1 / (1 + z))**2)

def eps(z):
    return 10**(-1.777 - 0.006 * (1 / (1 + z) - 1) * v(z) - 0.119 * (1 / (1 + z) - 1))

def M1(z):
    return 10**(11.514 - 1.793 * (1 / (1 + z) - 1) * v(z) - 0.251 * z * v(z))

def alpha(z):
    return -1.412 + 0.731 * (1 / (1 + z) - 1) * v(z)

def delta(z):
    return 3.508 + 2.608 * (1 / (1 + z) - 1) * v(z) - 0.043 * z * v(z)

def gamma(z):
    return 0.316 + 1.319 * (1 / (1 + z) - 1) * v(z) + 0.279 * z * v(z)

def fx(x, z):
    return -np.log10(10**(alpha(z) * x) + 1) + delta(z) * (np.log10(1 + np.exp(x)))**gamma(z) / (1 + np.exp(10**-x))

def Mstar(Mh, z):
    return 10**(np.log10(eps(z) * M1(z)) + fx(np.log10(Mh / M1(z)), z) - fx(0, z))

hvmax1=[]
hrvmax1=[]
hvfid1=[]
htr1=[]
hcs=[]

c=0.060999954122183286
b=1.1155471801757812
counts=0
hrc1=[]
m1=pow(10,11.59)
np.random.seed(123456)

################################################################################
def calc_values(Vmax, Rvmax, c200, cvir, Mvir, horder): 
   md=Mvir
   msigma=np.random.normal(0, 0.15)
   resigma=np.random.normal(0, 0.1)
   mb=md*2*0.0351/(pow(md/m1,-1.376)+pow(md/m1,0.608))*pow(10,msigma)
   rH=0.0938884074712349*pow(1 + 4.329004329004329e-11*mb,0.65)*pow(mb,0.1)*pow(10,resigma)
   rhoH=mb/(2.*3.1415923*pow(rH,3))
   rss=Rvmax/2.1626
   rhoss=rhos2(Vmax,rss)
   tr=0 # turn off SIDM for now.  
   htr1 = tr
   vmax1=fvmaxb(0,c200,rss,rhoH,rH)
   rmax1=frmaxb(0,c200,rss,rhoH,rH)
   shmr=mb/Mvir
   return horder, vmax1, rmax1, cvir, c200, shmr
######################################################################################

pool = Pool(processes=14)
np.random.seed(123456)
horder = np.arange(1, len(hvmax) + 1)

results = pool.starmap(
     calc_values,
     [(hvmax[i], hrvmax[i], hc200[i], hcvir[i], hmvir[i], horder[i]) for i in range(len(hvmax))]
     #[(hvmax[i], hrvmax[i], hc200[i], hcvir[i], hmvir[i], horder[i]) for i in range(56)]
)
horder1, hvmax1, hrvmax1, hcvir1, hc2001, hshmr = zip(*results)

with open('output_CDM_baryon_n1000_DMO.txt', 'w') as file:
    # Write the header to the file
    file.write('# order vmax rmax cvir c200 shmr\n')
    
    # Iterate through the lists and write each line to the file
    for i in range(len(hvmax1)):
        # Create a formatted string with the data for the current row
        line = f"{horder1[i]} {hvmax1[i]} {hrvmax1[i]} {hcvir1[i]} {hc2001[i]} {hshmr[i]}\n"
        
        # Write the formatted string to the file
        file.write(line)

#print('# order vmax rmax cvir c200')
#for i in range(len(hvmax1)):
#   print(horder1[i],hvmax1[i], hrvmax1[i], hcvir1[i], hc2001[i])

#------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.set_xlabel(r"$V_{\rm max}\rm \ (km/s)$",fontsize=20)
ax.set_ylabel(r"$R_{\rm max}\rm \ (kpc)$",fontsize=20)

#-------------------------------------------
#mmin=3.64755609660833e+8
mmin=0 
h=0.6774
#-------------------------------------------

#plt.scatter(hvmax1,   hrvmax1,  s=32,c=hcvir1,cmap='viridis_r',linewidth=0.1)
plt.scatter(hvmax1,   hrvmax1,  s=32,c=hshmr,cmap='hot_r',linewidth=0.1)
plt.clim(0, 120)
clb=plt.colorbar()
clb.ax.tick_params(labelsize=18)
clb.set_label(r'$c_{\rm vir}$', labelpad=-1, size=20)

ax.set_xlim([10, 1000])
ax.set_ylim([1, 2000])

ax.set_xscale('log')
ax.set_yscale('log')

ax.text(0.05, 0.9, r"CDM+baryon with a flat HMF", transform=ax.transAxes, fontsize=22,verticalalignment='bottom')
#ax.text(0.1, 0.8, r"$\rm generated\ with\ baryons$", transform=ax.transAxes, fontsize=22,verticalalignment='bottom')

fig.set_size_inches(9,8,forward='True')

plt.show()
#fig.savefig('fig_vmax_rmax_CDM_baryon_flat_HMF.png', bbox_inches='tight')

