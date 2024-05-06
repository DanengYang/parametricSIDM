import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import os
import math
import statistics
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from scipy.optimize import minimize, rosen, rosen_der
import matplotlib
import multiprocessing
from multiprocessing import Pool
matplotlib.use('Agg')
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

# uncomment this for a test with less points 
npoints=14
hvmax = hvmax[:npoints]
hrvmax=hrvmax[:npoints]
hcvir = hcvir[:npoints]
hmvir = hmvir[:npoints]
hrvir = hrvir[:npoints]
hc200 = hc200[:npoints]

GG=4.30073*1e-6
h=0.7

sigma0=100
w=100

def sigmaVis(v,sigma0,w):
      return (3.*sigma0*pow(w,6)*((-4*pow(v,2))/pow(w,2) + 2*(2 + pow(v,2)/pow(w,2))*np.log(1 + pow(v,2)/pow(w,2))))/pow(v,6)

def sigmaeff(veff,sigma0,w):
      fint = lambda lnv:pow(np.exp(lnv),7)*np.exp(-pow(np.exp(lnv),2)/(4*pow(veff,2)))*(2/3)*sigmaVis(np.exp(lnv),sigma0,w)*np.exp(lnv)
      val=scipy.integrate.quad(fint, np.log(0.01), np.log(15000))
      return val[0]/(512*pow(veff,8))

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
#nstep=1000
nstep=100
tf=10.0
ht=np.linspace(0,tf,nstep) # evolve 10 Gyrs

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
   veff = 1.648*np.sqrt(GG*rhoss)*rss*0.64
   veff = veff*np.sqrt((rhoss*rss*rss+rhoH*rH*rH/2)/(rhoss*rss*rss))
   sidmx=sigmaeff(veff,sigma0,w)
   trx=0
   vmax1=0
   rmax1=0
   # Assume a evolution time of 10 Gyr for all halos...
   for i in range(len(ht)):
      t=ht[i]
      dt=tf/nstep
      def eq_to_solve(zx):
        return tlb(zx) - (tf-t)
      zx, = fsolve(eq_to_solve, 1)
      Msx=mb*Mstar(Mvir, zx)/Mstar(Mvir,0) 
      rHx=rH*pow(1+zx,-0.75) # table 2 of 1404.2844, take an average from (0.2 to 1.3)
      rhoHx=Msx/(2*np.pi*pow(rHx,3))
      tc1=tcb(sidmx,rhoss,rss,rhoHx,rHx)
      dtr=dt/tc1
      if(i==0): 
         vmax1,rmax1=fvrmaxb(trx,c200,rss,rhoHx,rHx)
      if(trx<1.1): 
         trx+=dtr # this line should be earlier than the rest...
         vmaxp,rmaxp=fvrmaxb(trx+dtr*0.5,c200,rss,rhoHx,rHx)
         vmaxm,rmaxm=fvrmaxb(trx-dtr*0.5,c200,rss,rhoHx,rHx)
         vmax1+=(vmaxp-vmaxm)
         rmax1+=(rmaxp-rmaxm)
      else: break
   if(trx>1.1):
      trx=1.1
      if(dtr>0.05): print('Gravothermal evolution too fast, dtr: ',dtr)
   return horder, vmax1, rmax1, cvir, c200, rss, trx, rhoH, rH 
######################################################################################

num_cpus = os.cpu_count()
pool = Pool(processes=num_cpus) # specify number of CPUs here
#pool = Pool(processes=14) # specify number of CPUs here
np.random.seed(123456)
horder = np.arange(1, len(hvmax) + 1)

results = pool.starmap(
     calc_values,
     [(hvmax[i], hrvmax[i], hc200[i], hcvir[i], hmvir[i], horder[i]) for i in range(len(hvmax))]
)
horder1, hvmax1, hrvmax1, hcvir1, hc2001, hrss1, htr1, hrhoH1, hrH1 = zip(*results)
with open('output_SIDM_sigma0_100_w_100_baryon_n1000_integral_DMO.txt', 'w') as file:
    # Write the header to the file
    file.write('# order vmax rmax cvir c200 rss tr rhoH rH \n')
    
    # Iterate through the lists and write each line to the file
    for i in range(len(hvmax1)):
        # Create a formatted string with the data for the current row
        line = f"{horder1[i]} {hvmax1[i]} {hrvmax1[i]} {hcvir1[i]} {hc2001[i]} {hrss1[i]} {htr1[i]} {hrhoH1[i]} {hrH1[i]}\n"
        # Write the formatted string to the file
        file.write(line)

#------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.set_xlabel(r"$V_{\rm max,DM}\rm \ (km/s)$",fontsize=20)
ax.set_ylabel(r"$R_{\rm max,DM}\rm \ (kpc)$",fontsize=20)

#-------------------------------------------

plt.scatter(hvmax1,   hrvmax1,  s=32,c=htr1,cmap='coolwarm',linewidth=0.1)
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

#plt.show()

