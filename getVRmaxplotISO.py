import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.patches as mpatches
import math
from math import tanh
from math import log
from math import sqrt
from scipy.interpolate import interp1d
import sys
from numpy import vectorize
from scipy.integrate import quad
#from parametricRead import tlb, rhos2, rhosm, tc, frho, rhost, rst, rct, vmaxt, rmaxt, dvmaxt, drmaxt, fvmax, frmax, vrmaxt, vvmaxt, vdrmaxt, vdvmaxt
from parametricC4 import tlb, rhos2, rhosm, tc, frho, rhost, rst, rct, vmaxt, rmaxt, dvmaxt, drmaxt, fvmax, frmax, vrmaxt, vvmaxt, vdrmaxt, vdvmaxt

cm = plt.get_cmap('Reds')

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

# m2d5e7
nstep=210./6
data1 = np.genfromtxt('cdm/data_799_cdm.txt', names=True)
data2 = np.genfromtxt('sidm/data_796_vd100.txt', names=True)
# MW analogs
data3 = np.genfromtxt('dataMAH416CDM.txt', names=True)
data4 = np.genfromtxt('dataMAH416VD100.txt', names=True)
dataM = np.genfromtxt('mathcingresultsVD100.txt', names=True)
sidmid=796
cdmid=799

GG=4.30073*1e-6
h=0.7

sigma0=147.10088
w=24.3289794155754

def sigmaVis(v,sigma0,w):
      return (3.*sigma0*pow(w,6)*((-4*pow(v,2))/pow(w,2) + 2*(2 + pow(v,2)/pow(w,2))*np.log(1 + pow(v,2)/pow(w,2))))/pow(v,6)

def sigmaeff(veff,sigma0,w):
      fint = lambda lnv:pow(np.exp(lnv),7)*np.exp(-pow(np.exp(lnv),2)/(4*pow(veff,2)))*(2/3)*sigmaVis(np.exp(lnv),sigma0,w)*np.exp(lnv)
      val=scipy.integrate.quad(fint, np.log(0.01), np.log(2000)) # up to 2000 km/s, sufficient for MW scale study!  
      return val[0]/(512*pow(veff,8))

fig = plt.figure()
plt.xlabel(r"$t_L\ \rm (Gyr)$",fontsize=22)
plt.ylabel(r"$V_{\rm max}\ \rm (km/s)$",fontsize=22)
#plt.ylabel(r"$R_{\rm max}\ \rm (kpc)$",fontsize=22)

zf=-0.0064*pow(np.log10(data1['mvir'][0]/h),2)+0.0237*np.log10(data1['mvir'][0]/h)+1.8837
tform=14-1.*tlb(zf)
tform=0.5*tform
tbins=np.linspace(0,14,10000)
te=14-tform-tbins
te[te<0]=0
plt.plot(tlb(1/data2['scale']-1), data2['vmax'], '-',linewidth=3,color='m')
plt.plot(tlb(1/data1['scale']-1), data1['vmax'], '--',linewidth=3,color='black')

fvmax = interp1d(tlb(1/data1['scale']-1), data1['vmax'], kind='cubic', fill_value="extrapolate")
# with little h in unit... only this one? 
frmax = interp1d(tlb(1/data1['scale']-1), data1['scale']*data1['rvmax'], kind='cubic', fill_value="extrapolate")
def mr(r,rhos,rs):
   return 4*3.141593*rhos*pow(rs,3)*(-1 + rs/(r + rs) - np.log(rs) + np.log(r + rs))
def rhoNFW(r,rhos,rs):
   return rhos/((r/rs)*pow(1+r/rs,2))


# all physical 
fmvir = interp1d(tlb(1/data1['scale']-1), data1['mvir']/h, kind='cubic', fill_value="extrapolate")
fmh = interp1d(tlb(1/data3['scale']-1), data3['mvir']/h, kind='cubic', fill_value="extrapolate")
frsh = interp1d(tlb(1/data3['scale']-1), data3['scale']*data3['rs']/h, kind='cubic', fill_value="extrapolate")
frhosm = interp1d(tlb(1/data3['scale']-1), rhosm(data3['mvir']/h,data3['scale']*data3['rvir']/h,data3['scale']*data3['rs']/h), kind='cubic', fill_value="extrapolate")

amin=max(data1['scale'][len(data1)-1],data3['scale'][len(data3)-1])
data1=data1[data1['scale']>amin]
data3=data3[data3['scale']>amin]

fx = interp1d(tlb(1/data1['scale']-1), data1['scale']*(data1['x']-data3['x']), kind='cubic', fill_value="extrapolate")
fy = interp1d(tlb(1/data1['scale']-1), data1['scale']*(data1['y']-data3['y']), kind='cubic', fill_value="extrapolate")
fz = interp1d(tlb(1/data1['scale']-1), data1['scale']*(data1['z']-data3['z']), kind='cubic', fill_value="extrapolate")
d1=1e3*data1['scale']*np.sqrt(pow(data1['x']-data3['x'],2)+pow(data1['y']-data3['y'],2)+pow(data1['z']-data3['z'],2))/h
fd = interp1d(tlb(1/data1['scale']-1), d1, kind='cubic', fill_value="extrapolate") # physical distance! 

rss=frmax(14-tform)/2.1626/h # physical
rhoss=rhos2(fvmax(14-tform),rss)
sigmas=sigmaeff(fvmax(14-tform)*0.64,sigma0,w)
tc0=tc(sigmas,rhoss,rss)

t0=0
t1=0
vmax1=vmaxt(0,rhoss,rss)
rmax1=rmaxt(0,rhoss,rss)
msub1=fmvir(14-tform)
ht=[]
hv=[]
trx=0

for i in range(len(te)):
   t1=te[len(te)-i-1]
   tl=tbins[len(te)-i-1] # 14-tlb
   if(t1>t0):
      dt=(t1-t0)
      tx=(t1+t0)/2
      rss0=frmax(tl)/2.1626/h # physical
      rhoss0=rhos2(fvmax(tl),rss0)
      sigmas0=sigmaeff(abs(vmax1)*0.64,sigma0,w)
      tc1=tc(sigmas0,rhoss0,rss0)
      tr1=tx/tc1
      if(tr1>1.): tr1=1.
      dtr=dt/tc1
      trx+=dtr
      vmax1+=dtr*dvmaxt(tr1,rhoss0,rss0)
      rmax1+=dtr*drmaxt(tr1,rhoss0,rss0)
      dV1=(fvmax(tl+dt/2)-fvmax(tl-dt/2))
      dR1=(frmax(tl+dt/2)-frmax(tl-dt/2))/h
      vmax1-=dV1
      rmax1-=dR1
      if(rmax1<0.1): rmax1=0.1 
      if(vmax1<2): vmax1=2
      #-- dependent information -------------------------------
      rmaxx=rmax1/(rmaxt(tr1,rhoss0,rss0)/(rss0*2.1626))
      vmaxx=vmax1/(vmaxt(tr1,rhoss0,rss0)/(1.648*rss0*np.sqrt(GG*rhoss0)))
      rsx=rmaxx/2.1626
      rhosx=rhos2(vmaxx,rsx)
      rss1=rst(tr1,rhosx,rsx)
      rc1=rct(tr1,rhosx,rsx)
      rhoss1=rhost(tr1,rhosx,rsx)
      rt1=1e10 # not applicable for isolated halos
      #--------------------------------------------------------
      ht.append(14-tx-tform)
      hv.append(vmax1)
      t0=t1
        
plt.plot(ht, hv, '-',linewidth=2,color='g')

rss=data1['rvmax'][0]/2.1626/h # physical
rhoss=rhos2(data1['vmax'][0],rss)
sigmas=sigmaeff(data1['vmax'][0]*0.64,sigma0,w)
tc0=tc(sigmas,rhoss,rss)
tform=14-1.*tlb(zf)
tbins=np.linspace(0,14,160)
te=14-tform-tbins
te[te<0]=0
# does not work for subhalos
plt.plot(tbins[te>0], vvmaxt(te[te>0]/tc0,rhoss,rss), ':',linewidth=2,color='b')

# -------------------------------- transformer, then integral model -------------------
# cdmID vd100id cdmVmaxz0 cdmRmaxz0 cdmRvirz0 vd100Vmaxz0 vd100Rmaxz0 vmax rmax vmax1 rmax1 tr0 trx rhoss1 rss1 rc1 rt1 tr1
# Units: km/s kpc
np.random.seed(cdmid) # randomness checked, there is no correlation...
tsigma=tlb(zf)*pow(10,np.random.normal(0, 0.16))
tr2=tsigma/tc0
if(tr2>1.): tr2=1.
print(cdmid,sidmid,data1['vmax'][0],data1['rvmax'][0]/h,data1['rvir'][0]/h,data2['vmax'][0],data2['rvmax'][0]/h,vmaxt(tr2,rhoss,rss),rmaxt(tr2,rhoss,rss),vmax1,rmax1,tr2,trx,rhoss1,rss1,rc1,rt1,tr1)

ax = fig.gca()

ax.text(0.1, 0.85, r"CDM-"+str(cdmid)+"\nSIDM-"+str(sidmid), transform=ax.transAxes, fontsize=22,
        verticalalignment='bottom')

fig.set_size_inches(8,8,forward='True')

#plt.ylim([10,45])
plt.xlim([0., 14])

plt.gca().invert_xaxis()
#plt.xscale('log')
#plt.yscale('log')

plt.show()
fig.savefig('fig_tL_vmax_case_cdm_'+str(cdmid)+'_'+str(sidmid)+'_C4_1000bins.png', bbox_inches='tight')
#fig.savefig('fig_tL_vmax_case_cdm_'+str(cdmid)+'_'+str(sidmid)+'_Read_10000bins.png', bbox_inches='tight')

