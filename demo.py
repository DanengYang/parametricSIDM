import numpy as np
import matplotlib.pyplot as plt
from numpy import vectorize
from mpl_toolkits.mplot3d import Axes3D

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

# Given functions
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


# Define constants
rhoss = 1.0  # Normalized
rss = 1.0  # Normalized

# Create a grid
tr = np.linspace(0.001, 1.1, 200)  # Normalized time

r = pow(10,np.linspace(np.log10(0.01),np.log10(5),200))

R, TR = np.meshgrid(r, tr)

Density = np.zeros_like(R)

for i in range(len(tr)):
    current_tr = tr[i]
    current_rhos = rhost(current_tr, rhoss, rss)
    current_rs = rst(current_tr, rhoss, rss)
    current_rc = rct(current_tr, rhoss, rss)
    Density[i, :] = frho(R[i, :], current_rhos, current_rs, current_rc)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.log10(R), np.log10(TR), np.log10(Density), color='b',alpha=0.2, linewidth=0)

ax.set_xlabel(r'$\log_{10}(r/r_{s,0})$',fontsize=14)
ax.set_ylabel(r'$\log_{10}(t/t_{c})$',fontsize=14)
ax.set_zlabel(r"$\log_{10}(\rho/\rho_{s,0})$",fontsize=14)

plt.show()

fig.savefig('figs/demo.png', bbox_inches='tight', pad_inches=0.4)
