# Python script to plot RF heat map based on rssi values at lat/lon coordinates
# using cubic/bicubic interpolation
#
# Syntax:
#
#	python3 ./plot-cub.py file.csv
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
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

file_in_path = sys.argv[1]

# Load data from CSV
dat = np.genfromtxt(file_in_path, delimiter=',', skip_header=1)
X_dat = dat[:, 3]
Y_dat = dat[:, 4]
Z1_dat = dat[:, 0]
Z2_dat = dat[:, 2]

# Convert from pandas dataframes to numpy arrays
X, Y, Z1, Z2 = np.array([]), np.array([]), np.array([]), np.array([])
for i in range(len(X_dat)):
    #print("x dat {} {}".format(Z_dat[i], i))
    X = np.append(X, X_dat[i])
    Y = np.append(Y, Y_dat[i])
    Z1 = np.append(Z1, Z1_dat[i])
    Z2 = np.append(Z2, Z2_dat[i])

# create x-y points to be used in heatmap
xi = np.linspace(X.min(), X.max(), 1000)
yi = np.linspace(Y.min(), Y.max(), 1000)

# Interpolate for plotting
zi1 = griddata((X, Y), Z1, (xi[None, :], yi[:, None]),
               method='cubic', rescale=False)
zi2 = griddata((X, Y), Z2, (xi[None, :], yi[:, None]),
               method='cubic', rescale=False)

# I control the range of my colorbar by removing data
# outside of my range of interest
zmin = 80
zmax = 300
zi1[(zi1 < zmin) | (zi1 > zmax)] = None

# Create the contour plot
#fig, (ax1, ax2) = plt.subplots(nrows=2)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)

ax1.contour(xi, yi, zi1, levels=24, linewidths=0.0, colors='k')
CS1 = ax1.contourf(xi, yi, zi1, 15, cmap=plt.cm.rainbow, vmax=zmax, vmin=zmin)

fig.colorbar(CS1, ax=ax1)
# ax1.plot(xi, yi, 'ko', ms=3) # comment out to remove lat/lon position points
ax1.axis('off')
ax1.set_title('Measured RSSI')

ax2.contour(xi, yi, zi1, levels=24, linewidths=0.0, colors='k')
CS2 = ax2.contourf(xi, yi, zi2, 15, cmap=plt.cm.rainbow, vmax=zmax, vmin=zmin)
ax2.set_title('Simulated RSSI')

ax3.set_title('Flight Path')
#ax3.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
ax3.plot(X, Y, 'ko', ms=3)
ax3.axis('off')
fig.colorbar(CS2, ax=ax3)

fig.colorbar(CS2, ax=ax2)
#ax2.plot(xi, yi, 'ko', ms=3)
ax2.axis('off')

plt.subplots_adjust(hspace=0.5)
plt.show()
