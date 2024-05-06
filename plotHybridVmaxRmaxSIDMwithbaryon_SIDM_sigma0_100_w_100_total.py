# For the hybrid approach, one may directly compute the total Rmax Vmax without using this code... 
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
from scipy.optimize import fsolve
import os
import multiprocessing
from multiprocessing import Pool


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

data = np.genfromtxt('output_SIDM_sigma0_100_w_100_baryon_n1000_DMO.txt', names=True)

GG=4.30073*1e-6
h=0.7
def rhos2(Vmax,rs):
      return pow(Vmax/1.648/rs,2)/GG

#------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.set_xlabel(r"$V_{\rm max}\rm \ (km/s)$",fontsize=20)
ax.set_ylabel(r"$R_{\rm max}\rm \ (kpc)$",fontsize=20)

#-------------------------------------------

def equation(c200,rhoss):
   if(c200<1.5 or c200>120): return -1e10
   gx=1.0/(-(c200/(1 + c200)) + np.log(1 + c200))
   val = np.log(rho200c * pow(c200, 3) * gx / 3) - np.log(rhoss)
   return val

def eqfit(vars, vmaxx, rmaxx, rhoH, rH):
   c200_fic, rss_fic = vars
   if(c200_fic<1.5 or c200_fic>120):  return [-np.inf,rss_fic]
   gx=1.0/(-(c200_fic/(1 + c200_fic)) + np.log(1 + c200_fic))
   rhoss_fic=rho200c*pow(c200_fic,3)*gx/3
   r200_fic=rss_fic*c200_fic
   Mh=MtotNFW(r200_fic,rhoss_fic,rss_fic)

   lnpos=np.linspace(np.log(max(0.1,min(0.2*rss_fic,0.2*rH))),np.log(max(rss_fic*4,rH*5)),140)
   at=[]
   for j in range(len(lnpos)):
      x,y = MrContra2(np.exp(lnpos[j]), rhoss_fic, rss_fic, rhoH, rH, r200_fic, Mh)
      rf = max(0.01,x)
      mf = max(1e-3,y)
      at.append(np.sqrt(GG*mf/rf))
   if(len(at)==0): return 0
   vmax1 = np.max(at)
   rmax1 = np.exp(lnpos[np.argmax(at)])

   eq1 = np.log(vmax1/vmaxx)  
   eq2 = np.log(rmax1/rmaxx)  
   return [eq1, eq2]

#-- dependent information -------------------------------
def process_data(i):
   rho200c=27238.4 # consistent with used for BM2
   c200=data['c200'][i]
   gx=1.0/(-(c200/(1 + c200)) + np.log(1 + c200))
   rhoss=rho200c*pow(c200,3)*gx/3
   rss=data['rss'][i]
   rhoH=data['rhoH'][i]
   rH=data['rH'][i]
   Mb = (2 * np.pi * rhoH * pow(rH, 3))
   r200=c200*rss 
   Mh=MtotNFW(r200,rhoss,rss)
   if Mb > Mh:
       print("fb>1, bypass...")
       return -1,-1,-1 
   tr=data['tr'][i]
   vmaxp,rmaxp=fvrmaxb(tr,rhoss,rss,rhoH,rH)
   rmaxx=data['rmax'][i]/(rmaxp/(rss*2.1626))
   vmaxx=data['vmax'][i]/(vmaxp/(1.648*rss*np.sqrt(GG*rhoss)))
   # need to undo the effect of adiabatic contraction in rmaxx and vmaxx ......
   solution = fsolve(eqfit, [c200,rss], args=(vmaxx, rmaxx, rhoH, rH ))
   c200_fic = solution[0] 
   rss_fic = solution[1]  
   if(not np.isreal(c200_fic)): 
      print('c200_fic not a real number.......')
      return -1,-1,-1 
   #print('check args: ',c200_fic/c200,rss_fic/rss,tr,rhoH,rH)
   f= lambda lnx:4*3.141593*(rhoDZcore(np.exp(lnx),c200_fic,rss_fic,tr,rhoH,rH)+fHern(np.exp(lnx),rhoH,rH))*pow(np.exp(lnx),3) # one np.exp(lnx) from change of variable
   at=[]
   lnpos=np.linspace(np.log(max(0.1,min(0.2*rss,0.2*rH))),np.log(max(rss*4,rH*5)),200)
   rlow=0.2*min(np.exp(lnpos))
   for i in lnpos:
      at.append(np.sqrt(GG*(scipy.integrate.quad(f, np.log(rlow), i)[0])/np.exp(i)))
   at = [x for x in at if not math.isnan(x)]
   if(len(at)==0): 
      print('err finding vmax rmax')
      return -1,-1,-1 
   return np.max(at), np.exp(lnpos[np.argmax(at)]), tr 

######################################################################################


num_cpus = os.cpu_count()
pool = Pool(processes=num_cpus) # specify number of CPUs here
np.random.seed(123456)
horder = np.arange(1, len(data['tr']) + 1)

results = pool.map(process_data, range(len(data['tr'])))

hvmaxTot,   hrmaxTot, htr = zip(*results)

plt.scatter(hvmaxTot,   hrmaxTot,  s=32,c=htr,cmap='coolwarm',linewidth=0.1)
plt.clim(0, 1.1)
clb=plt.colorbar()
clb.ax.tick_params(labelsize=18)
clb.set_label(r'$\tau=10~{\rm Gyr}/t_{c,b}$', labelpad=0, size=20)

ax.set_xlim([10, 1200])
ax.set_ylim([0.5, 2000])

ax.set_xscale('log')
ax.set_yscale('log')

ax.text(0.05, 0.9, r"SIDM+baryon with a flat HMF, hybrid", transform=ax.transAxes, fontsize=22,verticalalignment='bottom')
ax.text(0.05, 0.8, r"$\sigma_0/m=100\ {\rm cm^2/g},\ w=100\ \rm km/s$", transform=ax.transAxes, fontsize=18,verticalalignment='bottom')

fig.set_size_inches(9,8,forward='True')

plt.show()
fig.savefig('figs/fig_vmax_rmax_SIDM_baryon_flat_HMF_sigma0_100_w_100.png', bbox_inches='tight')

