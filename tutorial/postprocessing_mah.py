import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.interpolate
from scipy.interpolate import interp1d
from numpy import vectorize
from scipy.optimize import fsolve
import sys
import os
from parametricCoredDZ_disk_BH import *

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

rhoss= 3061204.2368647573
rss  = 27.03927876720797
tr   = 0.0659056010349193 
#sigmaeff:  0.08104422591963001
#c200:  7.516188589322926  r200cdm:  203.23231853361023
rH = 2.3230719705674034
rhoH =  897647042.1372027
Mh =  1025428546487.9087
Ms=2*np.pi*rhoH*pow(rH,3)
print('rhoH: ',rhoH,' rH: ',rH)
print('rhoss: ',rhoss,' rss: ',rss,' tr: ',tr)

#zf=-0.0064*pow(np.log10(Mh),2)+0.0237*np.log10(Mh)+1.8837
#tf=tlb(zf)
tf=10

def eq1_to_solve(sigmax):
    return tcb(sigmax, rhoss, rss, 0, 0) - tf / tr
sigmax_solution, = fsolve(eq1_to_solve, 1)
print(' SIDM: ',sigmax_solution)


def eq1_to_solve(sigmax):
    return tcb(sigmax, rhoss, rss, rhoH, rH) - tf / tr
sigmax_solution, = fsolve(eq1_to_solve, 1)
print(' SIDM: ',sigmax_solution)

nstep=1000
ht=np.linspace(0,tf,nstep)

sidmx=0.27386801
sidmx2=0.08104422591963001
trx=0
tr0=0
hmh=[]
hms=[]
htr=[]
htr0=[]
for i in range(len(ht)):
   t=ht[i]
   dt=tf/nstep
   def eq_to_solve(zx):
     return tlb(zx) - (tf-t)
   zx, = fsolve(eq_to_solve, 1)
   rHx=rH*pow(1+zx,-0.75)
   Msx=Ms*np.exp(-2*zx)
   Mhx=Mh*np.exp(-zx)
   rhoHx=Msx/(2*np.pi*pow(rHx,3))
   tc1=tcb(sidmx,rhoss,rss,rhoHx,rHx)
   dtr=dt/tc1
   trx+=dtr
   hmh.append(Mhx/Mh)
   hms.append(Msx/Ms)
   htr.append(trx)
   tr0+=dt/tcb(sidmx2,rhoss,rss,rhoH,rH)
   htr0.append(tr0)
   #if(np.mod(i,100)==0): print(t,Mhx/Mh,Msx/Ms,trx)

print(' sidm: ',sidmx,' resulting trx: ',trx)

fig = plt.figure()

lw=2
#plt.plot(ht, hmh,   color='b',linestyle='--',linewidth=lw,  label=r'$M_h/M_{h,0}$')
#plt.plot(ht, hms,   color='r',linestyle='-',linewidth=lw,  label=r'$M_s/M_{s,0}$')
plt.plot(ht, htr,   color='g',linestyle='-',linewidth=lw,  label=r'$\sigma_{\rm eff}/m=0.274~\rm cm^2/g$')
hms=np.array(hms)
#plt.plot(ht, hms/hms,   color='r',linestyle='--',linewidth=lw,  label=r'$M_s/M_{s,0}$ (no growth)')
plt.plot(ht, htr0,   color='g',linestyle='--',linewidth=lw,  label=r'No growth, $\sigma_{\rm eff}/m=0.081~\rm cm^2/g$')

plt.xlabel(r"$t\rm \ (Gyr)$",fontsize=22)
plt.ylabel(r"$\tau=t/t_c $",fontsize=22)

plt.legend(fontsize=16,loc='upper left')

ax = fig.gca()

#plt.ylim([0.0,0.1])
plt.xlim([0.,tf])

#plt.yscale('log')

fig.set_size_inches(7,7,forward='True')
plt.show()
fig.savefig('fig_MAH_M81_baryon_postprocessing.pdf', bbox_inches='tight')

