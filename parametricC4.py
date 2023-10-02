import numpy as np
import scipy.interpolate
import math
from math import tanh
from math import log
from math import sqrt
from scipy.interpolate import interp1d
from numpy import vectorize

GG=4.30073*1e-6
h=0.7

def rhos2(Vmax,rs):
      return pow(Vmax/1.648/rs,2)/GG

def rhosm(mvir,rvir,rs):
      return mvir/(4*3.141593*pow(rs,3)*(-1+rs/(rvir + rs) - np.log(rs) + np.log(rvir + rs)) )

def tc(sigmax,rhox,rx):
      a=0.0
      val= 150/0.75/(sigmax*2.09e-10*rhox*pow(rx,1.0-a))*pow(4*3.141593*GG*rhox,-(1.0-a)/2)
      return val

# modified cored NFW profile 
k=4
def frho(r, rhos, rs, rc):
   return rhos*rs/pow(pow(r,k)+pow(rc,k),1./k)/pow(1+r/rs,2)

def rhost(tr,rhoss,rss):
      val=2.03305816 + 0.73806287*tr+ 7.26368767*pow(tr,5) -12.72976657*pow(tr,7) + 9.91487857*pow(tr,9) -0.1448 *(1-2.03305816)*np.log(tr+0.001)
      return val*rhoss

def rst(tr,rhoss,rss):
      val=0.71779858 -0.10257242*tr + 0.24743911*pow(tr,2) -0.40794176*pow(tr,3) -0.1448 *(1-0.71779858)*np.log(tr+0.001)
      return val*rss

def rct(tr,rhoss,rss):
      val=2.55497727*np.sqrt(tr) -3.63221179*tr + 2.13141953*pow(tr,2) -1.41516784*pow(tr,3) + 0.46832269*pow(tr,4)
      return val*rss

def vmaxt(tr,rhoss,rss):
      if(tr>1.1): tr=1.1
      val=1+ 0.17774902*tr -4.39860823*pow(tr,3) + 16.65523169*pow(tr,4) -18.8674121*pow(tr,5) + 9.07680873*pow(tr,7) -2.43612321*pow(tr,9)
      return val*1.648*rss*np.sqrt(GG*rhoss)
vvmaxt = vectorize(vmaxt)

def dvmaxt(tr,rhoss,rss):
      if(tr>1.1): tr=1.1
      val=0.17774902 - 13.195824689999998*pow(tr,2) + 66.62092676*pow(tr,3) - 94.33706049999999*pow(tr,4) + 63.53766111*pow(tr,6) - 21.925108889999997*pow(tr,8)
      return val*1.648*rss*np.sqrt(GG*rhoss)
vdvmaxt = vectorize(dvmaxt)

def rmaxt(tr,rhoss,rss):
      if(tr>1.1): tr=1.1
      val=1+ 0.00762288*tr -0.71998196*pow(tr,2) + 0.33760881*pow(tr,3) -0.13753822*pow(tr,4)
      return val*rss*2.1626
vrmaxt = vectorize(rmaxt)

def drmaxt(tr,rhoss,rss):
      if(tr>1.1): tr=1.1
      val=0.00762288 - 1.43996392*tr + 1.01282643*pow(tr,2) - 0.55015288*pow(tr,3)
      return val*rss*2.1626
vdrmaxt = vectorize(drmaxt)

def fvmax(tr,rhoss,rss):
   f= lambda x:4*3.141593*frho(x,rhost(tr,rhoss,rss),rst(tr,rhoss,rss),rct(tr,rhoss,rss))*x*x # physical
   at=[]
   pos=np.linspace(0.1*rss,rss*3,160)
   #pos=10**np.linspace(np.log10(0.01),np.log10(0.32),100)
   for i in pos:
      at.append(np.sqrt(GG*(scipy.integrate.quad(f, 0.001, i)[0])/i))
   return np.max(at)

def frmax(tr,rhoss,rss):
   f= lambda x:4*3.141593*frho(x,rhost(tr,rhoss,rss),rst(tr,rhoss,rss),rct(tr,rhoss,rss))*x*x # physical
   at=[]
   pos=np.linspace(0.1*rss,rss*3,160)
   #pos=10**np.linspace(np.log10(0.01),np.log10(0.32),100)
   for i in pos:
      at.append(np.sqrt(GG*(scipy.integrate.quad(f, 0.001, i)[0])/i))
   return pos[np.argmax(at)]

# works for the cosmological parameters in YNY23
def tlb(z): # in Gyr
   return 13.647247606199668 - 11.020482589612016*np.log(1.5800327517186143/pow(1 + z,1.5) + np.sqrt(1. + 2.4965034965034962/pow(1 + z,3.)))




