# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:07:31 2022

@author: USER
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from read_dataset import data_from_name
from matplotlib.cm import ScalarMappable 
from  matplotlib.colors import Normalize as NormalizeCbar

def main(argv=None):
    n_sensors =25
    placement = "leverage"
    if placement.lower()=="grid":
        s = 80
    else:
        s = 40
    savefolder = "plots/" + placement

    
    #Load data
    velocity_data_name300 = "data/cavflow/cavflow30000.npy"
    #velocity_data_name155 = "data/cavflow/cavflow15500.npy"
    #sens_data_name155 = "results/sens_sensitivity_error_s"+str(n_sensors)+ \
    #                    "e100_samp"+str(n_samp) + "l800_pert"+ str(pert) + "_Re155.npz"
    sens_data_name300 = "results/network_sens_s"+str(n_sensors)+ \
                        "e100_"+str(placement) + "_Re300.npz"
    
    flow_type = 'init'
    
    
    #Load sensitivity data
    #sens_data155 = np.load(sens_data_name155)
    sens_data300 = np.load(sens_data_name300)
    
    #morris_mean_abs155 = sens_data155["morris_mean_abs"][:,1].reshape(n_sensors,2)
    #morris_std155 = sens_data155["morris_std"][:,1].reshape(n_sensors,2)
    #sensor_locations_2D155 = sens_data155["base_sensors"]
    
    jac_mean_300 = sens_data300["jac_mean"]
    jac_std_300 = sens_data300["jac_std"]
    sensor_locations_2D300 = sens_data300["base_sensors"]
    
    m = 130
    n = 130
    x_center = -np.cos(np.pi*np.arange(0,m)/(n-1))
    y_center = -np.cos(np.pi*np.arange(0,m)/(n-1))
    yv, xv = np.meshgrid(x_center, y_center)
    
    
    
    
    
    #Load velocity data
    #Xsmall155, Xsmall_test, m, n = data_from_name(velocity_data_name155)
    #Xmean155 = Xsmall155.mean(axis=0).reshape(130,130)
    #Xinit155 = Xsmall155[0,:].reshape(130,130)
    
    Xsmall300, Xsmall_test, m, n = data_from_name(velocity_data_name300)
    Xmean300 = Xsmall300.mean(axis=0).reshape(130,130)
    Xinit300 = Xsmall300[0,:].reshape(130,130)
    
    if flow_type.lower() == 'init':
       # flow1 = Xinit155
        flow2 = Xinit300
    elif flow_type.lower() == 'mean':
       # flow1 = Xmean155
        flow2 = Xmean300



    plot_ReSingle(savefolder + "total_mean_abs_Re300",
                   jac_mean_300,sensor_locations_2D300, flow2,
                  "Re=30000", r"$\mu^*$",s=s)

    
    plot_ReSingle(savefolder + "total_std_Re300",
                  jac_std_300,sensor_locations_2D300, flow2,
                  "Re=30000", r"$\sigma$", s=s)


def plot_ReSingle(savename, Morris, pos, flow, title, colorbarlabel, \
                  figsize= (4.2,3.2), s=30, c_flow = 'jet', c_bar = 'inferno'):
    n=130
    x_centers = -np.cos(np.pi*np.arange(0,n)/(n-1))
    y_centers = np.cos(np.pi*np.arange(0,n)/(n-1))
    mX, mY = np.meshgrid(x_centers, y_centers)
    fig, ax = plt.subplots(1,1, figsize=figsize)
    cmin = np.min(Morris)
    cmax = np.max(Morris)
    minmax = np.max(np.abs(flow))
    
    plt.pcolormesh(mX, mY, flow, cmap=c_flow, vmin=-minmax, vmax=minmax)
    sens = plt.scatter(pos[:,0], pos[:,1], s=s, 
                         c=Morris, cmap = c_bar, vmin=cmin, vmax= cmax,
                         edgecolors ='k', linewidths=1.2)
    
    #ax.title.set_text(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, .05, 1])
    cbar = fig.colorbar(sens)
    #cbar = fig.colorbar(sens, ax=cbar_ax)
    #cbar_ax.axis("off")
    cbar.set_label(colorbarlabel, rotation = 270, labelpad = 20, fontsize = 12)

    fig.tight_layout()
    plt.savefig(savename + ".pdf")
    plt.savefig(savename + ".png")


def plot_ReSubplot(savename, Morris1 ,Morris2, pos1, pos2, flow1, flow2, title1, \
                  title2, colorbarlabel, figsize = (9,3.8), s=30, c_flow = 'jet', \
                  c_bar = 'gray_r'):
    #Setup discretization velocity data
    n=130
    x_centers = -np.cos(np.pi*np.arange(0,n)/(n-1))
    y_centers = np.cos(np.pi*np.arange(0,n)/(n-1))
    mX, mY = np.meshgrid(x_centers, y_centers)
    fig, ax = plt.subplots(1,2, figsize=figsize, sharey = True)
    cmin = np.min([np.min(Morris1), np.min(Morris2)])
    cmax = np.max([np.max(Morris1), np.max(Morris2)])
    minmax1 = np.max(np.abs(flow1))
    minmax2 = np.max(np.abs(flow2))
    
    ax[0].pcolormesh(mX, mY, flow1, cmap=c_flow, vmin=-minmax1, vmax=minmax1)
    ax[0].scatter(pos1[:,0], pos1[:,1], s=s, 
                         c=Morris1, cmap = c_bar, vmin=cmin, vmax= cmax,
                         edgecolors ='r')
    
    ax[0].title.set_text(title1)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[1].pcolormesh(mX, mY, flow2, cmap= c_flow, vmin=-minmax2, vmax=minmax2)
    ax[1].scatter(pos1[:,0], pos2[:,1], s=30, 
                         c=Morris2, cmap = c_bar, vmin=cmin, vmax= cmax,
                         edgecolors = 'r')
    
    ax[1].title.set_text(title2)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x")

    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, .05, 1])
    cbar = fig.colorbar(ScalarMappable(NormalizeCbar(vmin = cmin, vmax = cmax),
                                       cmap = c_bar), 
                        ax=ax.ravel().tolist(), shrink=0.95)
    #cbar = fig.colorbar(sens, ax=cbar_ax)
    #cbar_ax.axis("off")
    cbar.set_label(colorbarlabel, rotation = 270, labelpad = 20, fontsize = 12)
    ax[1].title.set_text(title2)

    #plt.tight_layout()
    plt.savefig(savename + ".pdf")
    plt.savefig(savename + ".png")
    
if __name__ == "__main__":
    sys.exit(main())