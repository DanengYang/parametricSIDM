import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
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

debug=1

data = np.genfromtxt("dataIC2574.txt", names=True)

Ms0= 5.08e8 
MBH=0
rH = 1.317
rhoH=Ms0/(2*np.pi*pow(rH,3))
#------------------------------------------------------------------------------------
# rough estimate
Mh0=1e10
R200=80
rhoss0=1.e6
rss0=20
c0=R200/rss0 # c200
tr0=0.

#------------------------------------------------------------------------------------
# Given baryons
#----------------------------------------------------------------------------------

def MrDisk(r,rH):
    Rd=(1+np.sqrt(2))/1.678*rH
    return Ms0 * (1 - np.exp(-r / Rd) * (1 + r / Rd)) 

# Modify fVcirc to handle array inputs
def fVcirc(r,c200,rss,tr,rH):
    rhoH=Ms0/(2*np.pi*pow(rH,3))
    def integrand(x,c200,rss,tr):
        return 4 * np.pi * rhoDZcore(x,c200,rss,tr,rhoH,rH,MBH) * x**2
    vec_integrand = np.vectorize(integrand)
    def integral_for_single_r(r_single):
        return np.maximum(0,scipy.integrate.quad(vec_integrand, 9e-4, r_single, args=(c200, rss, tr))[0])
    vec_integral = np.vectorize(integral_for_single_r)
    mr = vec_integral(r)+MrDisk(r,rH)+MBH
    return np.sqrt(GG * mr / r)

def fVcircDM(r,c200,rss,tr,rH):
    rhoH=Ms0/(2*np.pi*pow(rH,3))
    def integrand(x,c200,rss,tr):
        return 4 * np.pi * rhoDZcore(x,c200,rss,tr,rhoH,rH,MBH) * x**2
    vec_integrand = np.vectorize(integrand)
    def integral_for_single_r(r_single):
        return np.maximum(0,scipy.integrate.quad(vec_integrand, 9e-4, r_single, args=(c200, rss, tr))[0])
    vec_integral = np.vectorize(integral_for_single_r)
    mr = vec_integral(r)
    return np.sqrt(GG * mr / r)

def fVDisk(r,rhoH,rH):
   mr = MrDisk(r,rH)
   return np.sqrt(GG*mr/r)

def fVDiskBH(r,rhoH,rH):
   mr = MrDisk(r,rH)
   mr=mr+MBH
   return np.sqrt(GG*mr/r)

def adjust_to_bounds(x, lower_bound, upper_bound):
    return max(lower_bound, min(x, upper_bound))

# Bounds for each parameter
#bounds = ([c0*0.4,rss0*0.3,0.], [c0*2, rss0*3, 1.1])
bounds = ([1.,0.5,0.,0.2], [120, 120, 1e-10,18])
p0_adjusted = [
    adjust_to_bounds(c0, bounds[0][0], bounds[1][0]),
    adjust_to_bounds(rss0, bounds[0][1], bounds[1][1]),
    adjust_to_bounds(tr0, bounds[0][2], bounds[1][2]),
    4.
]

#RCdata=np.sqrt(np.maximum(pow(data['vrot'],2)-pow(fVDisk(data['r'],rhoH,rH),2),0)) 
RCdata=data['vrot']
#print(RCdata)

try:
    popt, pcov = curve_fit(fVcirc, data['r'], RCdata, bounds=bounds,p0=p0_adjusted) # fit to contracted dark matter only
    predicted = fVcirc(data['r'], *popt)
    residuals = RCdata - predicted
    err = np.std(residuals)
    #err=np.maximum(data['errV'],0.1) # cannot be too small
except RuntimeError as e:
    sys.exit(1)
    #pass
    #print("Error in initial attempt:", e)

c200 = popt[0]
rss = popt[1]
tr = popt[2] 
rH = popt[3]
rhoH = Ms0/(2*np.pi*pow(rH,3)) 
gx=1.0/(-(c200/(1 + c200)) + np.log(1 + c200))
rhoss=rho200c*pow(c200,3)*gx/3
perr = np.sqrt(np.diag(pcov))
correlation_matrix = calculate_correlation_matrix(pcov)
r200cdm=c200*rss
r200sidm=-1

#----------------------------------------------------------------------------------

chi_squared = np.sum((residuals / err) ** 2)
dof = len(data['r']) - len(popt)  # degrees of freedom = number of data points - number of parameters
reduced_chi_squared = chi_squared / dof

#------------------------------------------------------------------------------------

rout=data['r'][-1]
Mb=Ms0
Mdm=(fVcircDM(r200cdm,*popt)**2)*r200cdm/GG
Mdm200=MtotNFW(r200cdm, rhoss,rss)
Mb200=(fVDiskBH(r200cdm,rhoH,rH)**2)*r200cdm/GG

#------------------------------------------------------------------------------------

#zf=-0.0064*pow(np.log10(Mdm200),2)+0.0237*np.log10(Mdm200)+1.8837
#tf=tlb(zf)
tf=10

initial_guess = 1  # This should be adjusted based on your knowledge of the possible range of sigmax

def equation_to_solve(sigmax, *params):
    rhoss, rss, tr, rhoH, rH, tf = params
    return tcb(sigmax, rhoss, rss, rhoH, rH) - tf / tr
# Solve for sigmax
sigmax_solution, = fsolve(equation_to_solve, initial_guess, args=(rhoss,rss,tr, rhoH, rH, tf))
sigmax_solution10, = fsolve(equation_to_solve, initial_guess, args=(rhoss,rss,tr, rhoH, rH, 10))
tc=tcb(sigmax_solution, rhoss, rss, rhoH, rH)
tc10=tcb(sigmax_solution10, rhoss, rss, rhoH, rH)

# --------------------------------------------------------------------------------------
# Compute the rotation curves on the finer grid
hrad = 10**np.linspace(-2, np.log10(r200cdm), 200)
V_DM = fVcircDM(hrad, *popt)
V_Baryon = fVDiskBH(hrad, rhoH, rH)
V_Total = np.sqrt(V_DM**2 + V_Baryon**2)
Rmax_DM = hrad[np.argmax(V_DM)]
Vmax_DM = np.max(V_DM)
Rmax_Total = hrad[np.argmax(V_Total)]
Vmax_Total = np.max(V_Total)
# --------------------------------------------------------------------------------------
# concentration mass relation

# --------------------------------------------------------------------------------------
#echo "# name Mdm200 Mb200 r200cdm r200sidm c200cdm c200sidm Mdmout Mb rhoss err_c200 rss err_rss rhoH rH tr err_tr tf tc tc10 sigmaeff sigmaeff10 corr01 corr02 corr12 VmaxDM RmaxDM VmaxTot RmaxTot reduced_chi_squared Q s1_mod exp1min_mod2 a c Vgasout Vdiskout Vbulout"
#print("M81",Mdm200,Mb200,r200cdm,r200sidm,r200cdm/rss,r200sidm/rss,Mdm,Mb,rhoss,perr[0],popt[1],perr[1],rhoH,rH,popt[2],perr[2],tf,tc,tc10,sigmax_solution,sigmax_solution10,correlation_matrix[0][1],correlation_matrix[0][2],correlation_matrix[1][2],Vmax_DM,Rmax_DM,Vmax_Total,Rmax_Total,reduced_chi_squared,1,1,1,-1,-1,-1)
#------------------------------------------------------------------------------------

if(debug==1): 
   print(' rhoss=',rhoss)
   print(' rss  =',rss,  ' +/- ',perr[1])
   print(' tr   =',popt[2],   ' +/- ',perr[2])
   print('sigmaeff: ',sigmax_solution)
   print('c200: ',c200,' r200cdm: ',r200cdm)
   print('rH: ',rH,' rhoH: ',rhoH)
   print('Mh200: ',Mdm)

fig = plt.figure()

plt.scatter(data['r'], RCdata, s=32, label=f'Observation',color='k')
plt.plot(data['r'], fVcirc(data['r'], *popt), color='black', linestyle='-', linewidth=2, label='Total')

tr1=popt[2]
plt.plot(data['r'], fVcircDM(data['r'],*popt), 'b-',linewidth=2, label=f'Dark matter')
plt.plot(data['r'], fVDisk(data['r'], rhoH, rH), 'r-',linewidth=2, label='Baryons')
plt.plot(data['r'], np.sqrt(GG*MBH/data['r']), 'm-',linewidth=2, label='Black hole')

plt.xlabel(r"$r\rm \ (kpc)$",fontsize=22)
plt.ylabel(r"$V_{\rm circ}(r) \ \rm (km/s) $",fontsize=22)


plt.legend(fontsize=18,loc='upper left')

fs=18
ax = fig.gca()
ax.text(0.05,0.62,"IC2574+baryon", transform=ax.transAxes, fontsize=20, verticalalignment='top')
ax.text(0.5,0.96,r"CDM fit", transform=ax.transAxes, fontsize=20, verticalalignment='top')
delta=0.07
ax.text(0.5, 0.96-delta, r"$M_{h,200} = $" + f"{Mdm:.2e}" + r" $\rm M_{\odot}$", transform=ax.transAxes, fontsize=fs, verticalalignment='top')
ax.text(0.5,0.96-2*delta,r"$c_{200} = $" + f"{c200:.2f}", transform=ax.transAxes, fontsize=fs, verticalalignment='top')
ax.text(0.5,0.96-3*delta,f"$t/t_c = $ {tr1:.3f}", transform=ax.transAxes, fontsize=fs, verticalalignment='top')

plt.ylim([0,120])
plt.xlim([0.,10.5])

#plt.xscale('log')
#plt.yscale('log')

fig.set_size_inches(7,7,forward='True')
plt.show()
fig.savefig('fig_fitted_Vcirc_IC2574_CDM.pdf', bbox_inches='tight')


