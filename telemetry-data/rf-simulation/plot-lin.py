#Python script to plot RF heat map based on rssi values at lat/lon coordinates
# using linear/bilinear interpolation
#
#Syntax:
#
#	python3 ./plot-lin.py file.csv
#
#	the input file will be a comma separated value (CSV) resembling the format below:
#	Note: 	the first row is the header labels
#		remrssi is remote RSSI as measured by the remote device or drone (no units)
#		rssi is the RSSI measured at the source (no units)
#		simrssi is the calculated rssi at the lat/lon from the input file (no units)
#		lat/lon are position in decimal degrees
#
# rssi,remrssi,simrssi,lat,lon
# 154,150,126,39.034535,-76.65557
# TODO: add delta and delta ratio (delta/value)

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

import sys
import os
from scipy.interpolate import griddata

np.random.seed(19680801)
npts = 4000
ngridx = 2000
ngridy = 4000

file_in_path = sys.argv[1]

# Load data from CSV
dat = np.genfromtxt(file_in_path, delimiter=',',skip_header=1)
X_dat = dat[:,3]
Y_dat = dat[:,4]
Z1_dat = dat[:,1]
Z2_dat = dat[:,2]

# Convert from pandas dataframes to numpy arrays
x, y, z1, z2 = np.array([]), np.array([]), np.array([]), np.array([])
for i in range(len(X_dat)):
        #print("x dat {} {}".format(Z_dat[i], i))
        x = np.append(x, X_dat[i])
        y = np.append(y, Y_dat[i])
        z1 = np.append(z1, Z1_dat[i])
        z2 = np.append(z2, Z2_dat[i])

#x = np.random.uniform(-2, 2, npts)
#y = np.random.uniform(-2, 2, npts)
#z = x * np.exp(-x**2 - y**2)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)

# -----------------------
# Interpolation on a grid
# -----------------------
# A contour plot of irregularly spaced data coordinates
# via interpolation on a grid.

# Create grid values first.
xi = np.linspace(x.min(), x.max(), ngridx)
yi = np.linspace(y.min(), y.max(), ngridy)

# Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
triang = tri.Triangulation(x, y)
interpolator1 = tri.LinearTriInterpolator(triang, z1)
interpolator2 = tri.LinearTriInterpolator(triang, z2)
Xi, Yi = np.meshgrid(xi, yi)
zi1 = interpolator1(Xi, Yi)
zi2 = interpolator2(Xi, Yi)

# Note that scipy.interpolate provides means to interpolate data on a grid
# as well. The following would be an alternative to the four lines above:
#from scipy.interpolate import griddata
#zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

ax1.contour(xi, yi, zi1, levels=24, linewidths=0.0, colors='k')
cntr1 = ax1.contourf(xi, yi, zi1, levels=24, cmap=plt.cm.rainbow)

fig.colorbar(cntr1, ax=ax1)
#ax1.plot(x, y, 'ko', ms=3) # comment out to remove lat/lon position points
ax1.axis('off')

ax1.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
ax1.set_title('Measured RSSI')

# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

ax2.tricontour(x, y, z2, levels=24, linewidths=0.0, colors='k')
cntr2 = ax2.tricontourf(x, y, z2, levels=24, cmap=plt.cm.rainbow)
#cntr3 = ax3.tricontourf(x, y, z2, levels=24, cmap=plt.cm.rainbow)

fig.colorbar(cntr2, ax=ax2)
ax3.plot(x, y, 'ko', ms=3) # comment out to remove lat/lon position points
ax2.axis('off')

#ax1.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
ax2.set_title('Simulated RSSI')

ax3.set_title('Flight Path')
ax3.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
ax3.axis('off')
fig.colorbar(cntr2, ax=ax3)

plt.subplots_adjust(hspace=0.5)
plt.show()
