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
trunc=1.0 

def rhos2(Vmax,rs):
      return pow(Vmax/1.648/rs,2)/GG

def rhosm(mvir,rvir,rs):
      return mvir/(4*3.141593*pow(rs,3)*(-1+rs/(rvir + rs) - np.log(rs) + np.log(rvir + rs)) )

def tc(sigmax,rhox,rx):
      a=0.0
      val= 150/0.75/(sigmax*2.09e-10*rhox*pow(rx,1.0-a))*pow(4*3.141593*GG*rhox,-(1.0-a)/2)
      return val


# modified cored tNFW profile 
def ftrho(r, rhos, rs, rc,rt,tr):
   f=np.tanh(r/rc)
   rhoNFW=rhos*rs/r/pow(1+r/rs,2)
   mr= 4*3.1415927*rhos*pow(rs,3)*(-1 + rs/(r + rs) - np.log(rs) + np.log(r + rs))
   #return pow(f,n)*rhoNFW + n*pow(f,n-1)*(1-f*f)/(4*3.1415927*pow(r,2)*rc)*mr 
   return (f*rhoNFW + (1-f*f)/(4*3.1415927*pow(r,2)*rc)*mr)/pow(1+pow(r/rt,2-tr),(1+3*tr))

# modified cored NFW profile 
k=4
def frho(r, rhos, rs, rc):
   f=np.tanh(r/rc)
   rhoNFW=rhos*rs/r/pow(1+r/rs,2)
   mr= 4*3.1415927*rhos*pow(rs,3)*(-1 + rs/(r + rs) - np.log(rs) + np.log(r + rs))
   #return pow(f,n)*rhoNFW + n*pow(f,n-1)*(1-f*f)/(4*3.1415927*pow(r,2)*rc)*mr 
   return f*rhoNFW + (1-f*f)/(4*3.1415927*pow(r,2)*rc)*mr

# Read profile
def rhost(tr,rhoss,rss):
      val=1.33465688 + 0.77459132*tr+ 8.04226046*pow(tr,5) -13.89112027*pow(tr,7) + 10.17999859*pow(tr,9) -0.1448 *(1-1.33465688)*np.log(tr+0.001)
      return val*rhoss

# Read profile
def rst(tr,rhoss,rss):
      val=0.87711888 -0.23724033*tr + 0.22164058*pow(tr,2) -0.38678443*pow(tr,3) -0.1448 *(1-0.87711888)*np.log(tr+0.001)
      return val*rss

# Read profile
def rct(tr,rhoss,rss):
      val= 3.32381804*np.sqrt(tr) -4.89672376*tr + 3.36707187*pow(tr,2) -2.51208772*pow(tr,3) + 0.86989356*pow(tr,4)
      return val*rss
vrct = vectorize(rct)

# Read profile
def vmaxt(tr,rhoss,rss):
      if(tr>trunc): tr=trunc
      val=1+ 0.22890031*tr -5.01833969*pow(tr,3) + 17.75041534*pow(tr,4) -19.34522681*pow(tr,5) + 8.9525718*pow(tr,7) -2.36356284*pow(tr,9)
      return val*1.648*rss*np.sqrt(GG*rhoss)
vvmaxt = vectorize(vmaxt)

# Read profile
def dvmaxt(tr,rhoss,rss):
      if(tr>trunc): tr=trunc
      val= 0.22890031 -15.05501907*pow(tr,2) + 71.00166136*pow(tr,3) - 96.72613405*pow(tr,4) + 62.668*pow(tr,6) - 21.27206556*pow(tr,8)
      return val*1.648*rss*np.sqrt(GG*rhoss)
vdvmaxt = vectorize(dvmaxt)

# Read profile
def rmaxt(tr,rhoss,rss):
      if(tr>trunc): tr=trunc
      val=1-0.6025749*tr +1.04283152*pow(tr,2) -1.48440901*pow(tr,3) + 0.52632908*pow(tr,4)
      return val*rss*2.1626
vrmaxt = vectorize(rmaxt)

# Read profile
def drmaxt(tr,rhoss,rss):
      if(tr>trunc): tr=trunc
      val=-0.602575 + 2.08566*tr - 4.45323*pow(tr,2) + 2.10532*pow(tr,3) 
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




