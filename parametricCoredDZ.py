import numpy as np
import math
from scipy.optimize import curve_fit
import scipy.interpolate
from scipy.interpolate import interp1d
from numpy import vectorize
from scipy.optimize import fsolve
import sys
from scipy.integrate import quad
from scipy.misc import derivative

GG = 4.30073e-6
#rho200c=27195. # h=0.7,Omega0=0.3
rho200c=27238.4 # the one used, Planck 

def rhoNFW(r,rhos,rs):
   return rhos/(r/rs)/pow(1+r/rs,2)

def fHern(r,rhoH,rH):
   return rhoH/((r/rH)*pow(1+r/rH,3))

def MtotNFW(r,rhoss,rss):
   if(r<=0): return 0
   return 4*np.pi*rhoss*pow(rss,3)*(-1.0 + rss/(r + rss) - np.log(rss) + np.log(r + rss))

def MrHern(r,rhoH,rH):
   return 2*np.pi*pow(r,2)*rhoH*pow(rH,3)/pow(r + rH,2)

# New fit with extended simulation snapshots shows almost identical results, hence we continue to use the one in paper 1 
def rhost(tr,rhoss,rss):
      val=2.03305816 + 0.73806287*tr+ 7.26368767*pow(tr,5) -12.72976657*pow(tr,7) + 9.91487857*pow(tr,9) -0.1448 *(1-2.03305816)*np.log(tr+0.001)
      return val*rhoss

def rst(tr,rhoss,rss):
      val=0.71779858 -0.10257242*tr + 0.24743911*pow(tr,2) -0.40794176*pow(tr,3) -0.1448 *(1-0.71779858)*np.log(tr+0.001)
      return val*rss

def rct(tr,rhoss,rss):
      val=2.55497727*np.sqrt(tr) -3.63221179*tr + 2.13141953*pow(tr,2) -1.41516784*pow(tr,3) + 0.46832269*pow(tr,4)
      return val*rss
vrct = vectorize(rct)

def frho(r, rhoss, rss, tr):
   k=4.0
   scale=0.25
   rhos = rhost(tr,rhoss,rss)
   rs = rst(tr,rhoss,rss)
   rc = rct(tr,rhoss,rss)
   return ( rhos*rs/pow(pow(r,k)+pow(rc,k),1./k)/pow(1.+r/rs,2) )

def s1_zhao_dmo(x,x0,nu,spp):
    return 1./(1.+pow(x/x0,nu))+spp*np.log10(1.+pow(x/x0,nu))

def exp1min_func(x,c1,nu,delta,mu):
    return 1.+c1*pow(x,nu)-pow(x,mu)-c1*pow(1.e-6,nu)+pow(1.e-6,mu)

def s1_mod(x,x0,nu,spp):
    val = max(0,s1_zhao_dmo(x,x0,nu,spp))
    return min(val,2)

def exp1min_mod2(x,c1,nu,delta,mu):
    val = max(0,exp1min_func(x,c1,nu,delta,mu))
    return min(val,4)

popt_s1=[1.30278262e-03,2.86267536e+00,3.20335633e-01]
popt_c2=[1.86247747e+02, 1.37125654e+00, 0.00000000e+00, 1.41574162e-01]

# Adiabatic contraction effect
# Refs: Gnedin+ arXiv/astro-ph/0406247
#       Jiang+ arXiv/astro-ph/2206.12425
# Numerical version
def MrContra(r, rhoss, rss, rhoH, rH, Rvir, Mh):  # R200 is used in the place of Rvir
    Ms = (2 * np.pi * rhoH * pow(rH, 3))
    fb = Ms / Mh
    if Ms > Mh:
        print("fb>1, exit")
        return -1
    rbar = 0.85 * pow(r / Rvir, 0.8) *Rvir
    Mi = MtotNFW(rbar,rhoss,rss) + MrHern(rbar, rhoH, rH)
    def eq_to_solve(lnrf):
        rfbar = 0.85 * pow(np.exp(lnrf) / Rvir, 0.8) * Rvir
        return np.log(r)-lnrf - np.log(1. - fb + MrHern(rfbar, rhoH, rH)/Mi) 
    rf = np.exp(fsolve(eq_to_solve, np.log(r))[0])
    return rf, (1. - fb) * MtotNFW(r,rhoss,rss)

# Use the analytic fits in Gnedin+ arXiv/astro-ph/0406247
def MrContra2(r, rhoss, rss, rhoH, rH, Rvir, Mh):  # R200 is used in the place of Rvir
    Ms = (2 * np.pi * rhoH * pow(rH, 3))
    fb = Ms / Mh
    rb=rH
    c=Rvir/rss
    gc=pow(np.log(1.+c)-c/(1.+c),-1) 
    xb=rb/Rvir
    a=2*fb*pow(1+xb,2)/(xb*xb*c*c*gc)
    A = 0.85
    w = 0.8
    x = r/Rvir
    rbar = A*pow(x,w) * Rvir
    #---------------
    Q = pow((1-fb)/(3.*a),3)+1./pow(2*a,2)
    y1 = pow(pow(Q,0.5)+1./(2*a),1./3) - pow(pow(Q,0.5)-1./(2*a),1./3)
    Qhat = pow(a,-3)*pow((1.-fb)/(1.+2*w),1.+2*w) + pow(2*a,-2)
    yw = pow(pow(Qhat,0.5)+1./(2*a),1./(1.+2*w)) - pow(pow(Qhat,0.5)-1./(2*a),1./(1.+2*w))
    y0 = y1*np.exp(-2.*a) + yw*(1. - np.exp(-2.*a))
    #---------------
    n=1 # Hernquist
    b=2*y0/(1.-y0)*(2./(n*xb)-4.*c/3)/(2.6+(1.-fb)/(a*pow(y0,2*w)))
    Mi = MtotNFW(rbar,rhoss,rss)
    txy0 = pow(1.-fb+MrHern(pow(y0,w)*rbar ,rhoH,rH)/Mi,-1)
    tx1  = pow(1.-fb+MrHern(rbar ,rhoH,rH)/Mi,-1) 
    yx = txy0*np.exp(-b*x) + tx1*(1.-np.exp(-b*x))
    rf = r*yx
    #print('contra2, yx=',yx,txy0,tx1,b)
    #print('contra2, yw=',yw,' y1:',y1,' b: ',b,' a: ',a,' c: ',c)
    return rf, (1.-fb)*MtotNFW(r,rhoss,rss)

# Global cache for storing computed values of a and c
ac_cache = {}
def rhoDZcore(r,c200,rss,tr,rhoH,rH): # Eq.11, 2004.08395 
   # Compute rhoss following arXiv:0002395
   gx=1.0/(-(c200/(1 + c200)) + np.log(1 + c200))
   rhoss=rho200c*pow(c200,3)*gx/3
   Rvir=c200*rss #r200
   Ms=(2*np.pi*rhoH*pow(rH,3))
   Mh=MtotNFW(Rvir,rhoss,rss)
   #-------------------------------------
   gamma=1.6
   gamma2=20
   reff=(rhoss*pow(rss,3) + gamma*rhoH*pow(rH,3)*0.5)/(rhoss*pow(rss,2) + gamma*rhoH*pow(rH,2)*0.5)
   rhoeff=(rhoss*rss/reff + gamma2*rhoH*pow(rH,3)/(reff*pow(reff+rH,2)) )
   ta= 1/((rhoeff)*2.09e-10)*pow(4*3.141593*GG*(rhoss*rss*rss+gamma*rhoH*rH*rH*0.5),-0.5)
   tb= 1/((rhoss)*2.09e-10)*pow(4*3.141593*GG*(rhoss*rss*rss),-0.5)
   ratio=ta/tb
   #-------------------------------------
   r1=0.01*Rvir
   s1DMO= -(pow(r1,2.0)*pow(rhoss,-1.0)*pow(rss,-1.0)*(-2.*rhoss*pow(r1,-1.)*pow(1. + r1*pow(rss,-1.),-3.) -
       rhoss*rss*pow(r1,-2.)*pow(1. + r1*pow(rss,-1.),-2.))*pow(1. + r1*pow(rss,-1.),2.))
   cDMO=Rvir/rss
   s1=s1DMO
   c2=cDMO
   #--------------------------------------------------------------
   # Get a and c numerically
   r200x=Rvir
   def MDZ(lnr,a,c):
      rx = float(r200x)/c
      r=np.exp(lnr)
      x=r/rx
      def gx(x):
          return pow(pow(x,0.5)/(1+pow(x,0.5)),2.0*(3.0-a))
      return np.log(Mh*gx(x)/gx(c)) # Eq.5 of 2004.08395
   # Check if the values are in the cache
   cache_key = (c200, rss, tr, rhoH, rH)
   if cache_key in ac_cache:
      a, c = ac_cache[cache_key]
   else:
      rticks = np.logspace(np.log10(rss*0.05),np.log10(rss*5),60)
      hrf=[]
      hm=[]
      for j in range(len(rticks)): 
         x,y = MrContra2(rticks[j],rhoss,rss,rhoH,rH,Rvir,Mh) # should really be Rvir here...
         hrf.append(max(0.01,x))
         hm.append(max(1e-3,y))
      hrf=np.array(hrf)
      hm=np.array(hm)
      popt0, pcovx0 = curve_fit(MDZ, np.log(hrf), np.log(hm))
      a=popt0[0]
      c=popt0[1]
      ac_cache[cache_key] = (a, c)
   #print('DZ parameters: ',r,a,c)
   #--------------------------------------------------------------
   #a=(1.5*s1-2.*(3.5-s1)*np.sqrt(0.01)*np.sqrt(c2))/(1.5-(3.5-s1)*np.sqrt(0.01)*np.sqrt(c2))
   #c=((s1-2.)/((3.5-s1)*np.sqrt(0.01)-1.5/np.sqrt(c2)))**2
   a=min(2.8,a) # a cannot be greater than 3..........
   c=max(1,c) # c cannot be lower than 1..............
   rx0 = float(Rvir)/c
   mu=pow(c,a-3)*pow((1+pow(c,0.5)),2*(3-a))
   rhohbar=Mh/(4*np.pi/3.*pow(Rvir,3))
   rhox0=(1-a/3.)*pow(c,3)*mu*rhohbar
   #-------------------------------------
   ftrrho = rhost(tr,rhoss,rss)/rhoss
   ftrrs = rst(tr,rhoss,rss)/rss
   ftrrc = rct(tr,rhoss,rss)/rss
   rc=rss*ftrrc # check if rx outperforms rs 
   rc=rc*pow(ratio,2.)
   rhox=rhox0*ftrrho
   rx=rx0*ftrrs
   x = r/rx
   k=4.0 *pow(ratio,0.5)

   if(abs(a-1)<1e-6):
      val=rhox/pow( pow(r/rx,k)+pow(rc/rx,k) ,1/k)/pow(1.+pow(x,1./2),2*(3.5-a))/pow(1+(rx/rx0-1)*r/Rvir,0.5)
   else:
      try:
          alphap = pow(rc/(0.4*rss), 1) * pow(rhox0 * rx0 / (rhoss * rss + 0.4 * rhoH * rH), 1 / (a - 1))
      except OverflowError:
          alphap = -np.inf  # Set to -np.inf or other if calculation overflows
      val= rhox/pow(r/rx+alphap,a-1)/pow( pow(r/rx,k)+pow(rc/rx,k) ,1/k)/pow(1.+pow(x,1./2),2*(3.5-a))/pow(1+(rx/rx0-1)*r/Rvir,0.5)
   if np.isnan(val).any():  # Changed to check the entire array
       return 0 
   else:
       return val

def fVhern(r,rhoH,rH):
   mr = 2*np.pi*pow(r,2)*rhoH*pow(rH,3)/pow(r + rH,2)
   return np.sqrt(GG*mr/r)

def fvrmaxb(tr,c200,rss,rhoH,rH):
   f= lambda lnx:4*3.141593*(rhoDZcore(np.exp(lnx),c200,rss,tr,rhoH,rH) )*pow(np.exp(lnx),3) # one np.exp(lnx) from change of variable
   at=[]
   lnpos=np.linspace(np.log(max(0.1,min(0.2*rss,0.2*rH))),np.log(max(rss*4,rH*5)),200)
   rlow=0.2*min(np.exp(lnpos))
   for i in lnpos:
      at.append(np.sqrt(GG*(scipy.integrate.quad(f, np.log(rlow), i)[0])/np.exp(i)))
   at = [x for x in at if not math.isnan(x)]
   #at = [x for x in at if x<1000]
   if(len(at)==0): return 0
   else: 
      return np.max(at),np.exp(lnpos[np.argmax(at)]) # vmax and rmax

def fvrmaxbTot(tr,c200,rss,rhoH,rH):
   f= lambda lnx:4*3.141593*(rhoDZcore(np.exp(lnx),c200,rss,tr,rhoH,rH)+fHern(np.exp(lnx),rhoH,rH))*pow(np.exp(lnx),3) # one np.exp(lnx) from change of variable
   at=[]
   lnpos=np.linspace(np.log(max(0.1,min(0.2*rss,0.2*rH))),np.log(max(rss*4,rH*5)),200)
   rlow=0.2*min(np.exp(lnpos))
   for i in lnpos:
      at.append(np.sqrt(GG*(scipy.integrate.quad(f, np.log(rlow), i)[0])/np.exp(i)))
   at = [x for x in at if not math.isnan(x)]
   if(len(at)==0): return 0
   else: return np.max(at), np.exp(lnpos[np.argmax(at)]) # vmax and rmax

def rhoC4(r,c200,rss,tr,rhoH,rH): # Eq.11, 2004.08395 
   # Compute rhoss following arXiv:00029395
   gx=1.0/(-(c200/(1 + c200)) + np.log(1 + c200))
   rhoss=rho200c*pow(c200,3)*gx/3
   #-------------------------------------
   gamma=1.6
   gamma2=20
   reff=(rhoss*pow(rss,3) + gamma*rhoH*pow(rH,3)*0.5)/(rhoss*pow(rss,2) + gamma*rhoH*pow(rH,2)*0.5)
   rhoeff=(rhoss*rss/reff + gamma2*rhoH*pow(rH,3)/(reff*pow(reff+rH,2)) )
   ta= 1/((rhoeff)*2.09e-10)*pow(4*3.141593*GG*(rhoss*rss*rss+gamma*rhoH*rH*rH*0.5),-0.5)
   tb= 1/((rhoss)*2.09e-10)*pow(4*3.141593*GG*(rhoss*rss*rss),-0.5)
   ratio=ta/tb
   #-------------------------------------
   ftrrho=2.03305816 + 0.73806287*tr+ 7.26368767*pow(tr,5) -12.72976657*pow(tr,7) + 9.91487857*pow(tr,9) -0.1448 *(1-2.03305816)*np.log(tr+0.001)
   ftrrs=0.71779858 -0.10257242*tr + 0.24743911*pow(tr,2) -0.40794176*pow(tr,3) -0.1448 *(1-0.71779858)*np.log(tr+0.001)
   ftrrc=2.55497727*np.sqrt(tr) -3.63221179*tr + 2.13141953*pow(tr,2) -1.41516784*pow(tr,3) + 0.46832269*pow(tr,4)
   rhox=rhoss*ftrrho
   rx=rss*ftrrs
   rc=rss*ftrrc*pow(ratio,2)
   k=4.0 *pow(ratio,0.5)
   return rhox/pow( pow(r/rx,k)+pow(rc/rx,k) ,1/k)/pow(1+r/rx,2)

def fvrmaxbC4(tr,c200,rss,rhoH,rH):
   f= lambda lnx:4*3.141593*(rhoC4(np.exp(lnx),c200,rss,tr,rhoH,rH))*pow(np.exp(lnx),3) # one np.exp(lnx) from change of variable
   at=[]
   lnpos=np.linspace(np.log(0.5*rss),np.log(rss*4),200)
   rlow=0.2*min(np.exp(lnpos))
   for i in lnpos:
      at.append(np.sqrt(GG*(scipy.integrate.quad(f, np.log(rlow), i)[0])/np.exp(i)))
   at = [x for x in at if not math.isnan(x)]
   if(len(at)==0): return 0
   else: return np.max(at), np.exp(lnpos[np.argmax(at)])

#def dvmaxt(tr,rhoss,rss):
#      val=0.17774902 - 13.195824689999998*pow(tr,2) + 66.62092676*pow(tr,3) - 94.33706049999999*pow(tr,4) + 63.53766111*pow(tr,6) - 21.925108889999997*pow(tr,8)
#      return val*1.648*rss*np.sqrt(GG*rhoss)
#vdvmaxt = vectorize(dvmaxt)
#
#def rmaxt(tr,rhoss,rss):
#      val=1+ 0.00762288*tr -0.71998196*pow(tr,2) + 0.33760881*pow(tr,3) -0.13753822*pow(tr,4)
#      return val*rss*2.1626
#vrmaxt = vectorize(rmaxt)


# Assuming pcov is your covariance matrix from curve_fit
def calculate_correlation_matrix(pcov):
    # Calculate the standard deviations (square root of the variances along the diagonal)
    sigma = np.sqrt(np.diag(pcov))
    # Calculate the outer product of sigma to get denominator for correlation matrix elements
    denominator = np.outer(sigma, sigma)
    # Element-wise division of the covariance matrix by the denominator to get the correlation matrix
    correlation_matrix = pcov / denominator
    # Ensure the diagonal elements are exactly 1
    np.fill_diagonal(correlation_matrix, 1)
    return correlation_matrix

def tcb(sigmax,rhox,rx,rhoH,rH):
      gamma=1.6
      gamma2=20
      reff=(rhox*pow(rx,3) + gamma*rhoH*pow(rH,3)*0.5)/(rhox*pow(rx,2) + gamma*rhoH*pow(rH,2)*0.5)
      rhoeff=(rhox*rx/reff + gamma2*rhoH*pow(rH,3)/(reff*pow(reff+rH,2)) )
      val= 150/0.9/(sigmax*(rhoeff)*2.09e-10)*pow(4*3.141593*GG*(rhox*rx*rx+gamma*rhoH*rH*rH*0.5),-0.5)
      return val

# works for the cosmological parameters in YNY23
def tlb(z): # in Gyr
   return 13.647247606199668 - 11.020482589612016*np.log(1.5800327517186143/pow(1 + z,1.5) + np.sqrt(1. + 2.4965034965034962/pow(1 + z,3.)))


