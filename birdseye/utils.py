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

def build_plots(xp=[], belief=[], fig=None, ax=None, time_step=None, sensor=None, target=None, particles=None):

    fig = plt.figure(figsize=(21, 7))
    plt.tight_layout()
    # Put space between plots
    plt.subplots_adjust(wspace=0.5)
    
    # Particle Plot (Polar)
    ax = fig.add_subplot(1, 4, 1, polar=True)

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
    ax = fig.add_subplot(1, 4, 2, polar=True)

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
    ax = fig.add_subplot(1, 4, 3)
    
    cart  = np.array(list(map(pol2cart, belief[:,0], belief[:,1]*np.pi/180)))
    x = cart[:,0]
    y = cart[:,1]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #plt.clf()
    ax.imshow(heatmap.T, extent=extent, origin='lower')

    # Particle/Sensor Plot (absolute)
    if sensor is not None and target is not None and particles is not None: 
        ax = fig.add_subplot(1, 4, 4, polar=True)

        particles = np.array(particles)
        
        ax.plot(particles[:,1]*np.pi/180, particles[:,0], 'ro')
        ax.plot(sensor[1]*np.pi/180, sensor[0], 'go')
        ax.plot(target[1]*np.pi/180, target[0], 'bo')
    
        ax.set_title('Absolute position'.format(time_step), fontsize=16)

        
    
    plt.show()
    mean_bel_err, rmse = tracking_error(target, particles)
    print('mean belief error = {}, rmse = {}'.format(mean_bel_err, rmse))


def distance_to_target(target, sensor):

    pass

def tracking_error(target, particles): 

    tar_x, tar_y = pol2cart(target[0], target[1]*np.pi/180)

    particles_x, particles_y = pol2cart(particles[:,0], particles[:,1]*np.pi/180)

    # mean belief error 
    mean_x = np.mean(particles_x)
    mean_y = np.mean(particles_y)
    mean_bel_err = (mean_x - tar_x)**2 + (mean_y - tar_y)**2

    # root mean square error
    rmse = np.sqrt(np.mean((particles_x - tar_x)**2 + (particles_y - tar_y)**2))

    return mean_bel_err, rmse 





