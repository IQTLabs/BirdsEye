# utils.py

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from IPython.display import clear_output

# Some transform functions
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


##################################################################
# Plotting
##################################################################

def build_plots(xp=[], belief=[], fig=None, ax=None, time_step=None):

    fig = plt.figure(figsize=(21, 7))
    plt.tight_layout()
    # Put space between plots
    plt.subplots_adjust(wspace=0.5)
    
    # Particle Plot (Polar)
    ax = fig.add_subplot(1, 3, 1, polar=True)

    grid_r, grid_theta = [],[]
    plot_r = [row[0] for row in belief]
    plot_theta = np.array([row[1] for row in belief])*np.pi/180
    plot_x_theta = xp[1]*np.pi/180
    plot_x_r = xp[0]
    
    clear_output(wait=True)
    
    ax.plot(plot_theta, plot_r, 'ro')
    ax.plot(plot_x_theta, plot_x_r, 'bo')
    ax.set_ylim(-150,150)
    ax.set_title('iteration {}'.format(time_step), fontsize=16)
   

    # Particle Plot (Polar) with Interpolation
    ax = fig.add_subplot(1, 3, 2, polar=True)

    # Create grid values first via histogram.
    nbins = 10
    counts, xbins, ybins = np.histogram2d(plot_theta, plot_r, bins=nbins)

    # Make a meshgrid for theta, r values
    tm, rm = np.meshgrid(xbins[:-1], ybins[:-1])

    # Build contour plot
    ax.contourf(tm, rm, counts)
    # True position
    ax.plot(plot_x_theta, plot_x_r, 'bo')
    ax.set_ylim(-150,150)
    ax.set_title('Interpolated Belief'.format(time_step), fontsize=16)
    
    
    # Heatmap Plot (Cartesian)
    ax = fig.add_subplot(1, 3, 3)
    
    cart  = np.array(list(map(pol2cart, belief[:,0], belief[:,1])))
    x = cart[:,0]
    y = cart[:,1]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #plt.clf()
    ax.imshow(heatmap.T, extent=extent, origin='lower')
    
    plt.show()


