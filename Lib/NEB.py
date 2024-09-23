#!/usr/bin/env python3

import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numba import jit
from scipy.interpolate import interp1d
import matplotlib.colors as colors



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#@jit(nopython=True)
def interp_bilinear(X,Y,Z,x_star,y_star,nreplica):

    '''
    Bilinear interpolation: takes a matrix Z[1...m]x[1...n], two vectors X[1...m] and Y[1...n]  and the point outside the grid [x_star, y_star] as input and returns the interpolated point z_star.  (For now implemented only for square matrices mxm )
    
    '''

    z_star = np.zeros(nreplica)
    idx_new = np.zeros(nreplica) 
    idy_new = np.zeros(nreplica)

    # Find the indexes of the new position
    for i in range(1,nreplica-1):
                                                                  
        idx_list = np.array(np.where( X > x_star[i] ) )
        idy_list = np.array(np.where( Y > y_star[i] ) )
        idx_new[i] = idx_list[0,0]
        idy_new[i] = idy_list[0,0]


    idx_new = [ int(x) for x in idx_new ]
    idy_new = [ int(x) for x in idy_new ]
    
    for i in range(1,nreplica-1):

        z1 = Z[idy_new[i]-1][idx_new[i]-1]
        z2 = Z[idy_new[i]][idx_new[i]-1]      
        z3 = Z[idy_new[i]-1][idx_new[i]]
        z4 = Z[idy_new[i]-1][idx_new[i]]
        
        t = ( x_star[i] - X[idx_new[i]-1] )/(X[idx_new[i]] - X[idx_new[i]-1])
        u = ( y_star[i] - Y[idy_new[i]-1] )/(Y[idy_new[i]] - Y[idy_new[i]-1])

        z_star[i] = (1-t)*(1-u)*z1 + t*(1-u)*z2 + t*u*z3 + (1-t)*u*z4
        
        
    return z_star


def find_gradient(X,Y,Z,x_star,y_star,nreplica,dx,dy):

    x1 = np.zeros(nreplica)
    x2 = np.zeros(nreplica)
    y1 = np.zeros(nreplica)
    y2 = np.zeros(nreplica)   

    for i in  range(1,nreplica-1):
     
        x1[i] = x_star[i] + dx
        x2[i] = x_star[i] - dx
        y1[i] = y_star[i] + dy
        y2[i] = y_star[i] - dy
        
    z_x1 = interp_bilinear(X,Y,Z,x1,y_star,nreplica)
    z_x2 = interp_bilinear(X,Y,Z,x2,y_star,nreplica)

    z_y1 = interp_bilinear(X,Y,Z,x_star,y1,nreplica)
    z_y2 = interp_bilinear(X,Y,Z,x_star,y2,nreplica)

    dzdx = ( z_x1 - z_x2 )/( 2*dx )
    dzdy = ( z_y1 - z_y2 )/( 2*dy )

    grad = np.array([dzdx, dzdy])

    return grad

def find_minima(X, Y, Z, grad):

    '''
    Search stationary points and compute numerical Hessian to find minima (TODO)
    
    '''
    minima = []
    x_min = []
    y_min = []
    threshold = 0.3

    for i in range(1,200-1):

        for j in range(1,200-1):


            if grad[i,j] < threshold:
        
               minima.append( Z[i,j])
               x_min.append(X[i])
               y_min.append( Y[j]) 


    print("number of stationary points found for threshold "+str(threshold)+": "+str(len(minima))+"")


    return minima, x_min, y_min



def initialize_pathway(X,Y,Z,tspan, nreplica,indice_1,indice_2,spacing):


    idx = np.linspace(indice_1[0],indice_2[0],spacing)
    idx = np.array(idx,dtype=int).reshape(spacing)
    idy = np.linspace(indice_1[1],indice_2[1],spacing)
    idy = np.array(idy,dtype=int).reshape(spacing)

    min_a = np.array([X[indice_1[0]],Y[indice_1[1]]])
    min_b = np.array([X[indice_2[0]],Y[indice_2[1]]])

    pos_store = np.zeros((2,nreplica,tspan))
    pos_store[:,:,0] = np.array([X[idx],Y[idy]]) 

    x_star = np.zeros(nreplica)
    y_star = np.zeros(nreplica)
    x_star[0] = X[idx[0]]
    y_star[0] = Y[idy[0]]
    x_star[-1] = X[idx[-1]]
    y_star[-1] = Y[idx[-1]]
    z_star_hist = np.zeros((nreplica,tspan))
    z_star_hist[:,0] = Z[idy,idx]

    return pos_store, z_star_hist


def spring_constant_function(curvature, k_min, k_max):
    """
    Calculates the spring constant based on the curvature.

    Args:
        curvature: The curvature value at an image point.
        k_min: The minimum spring constant.
        k_max: The maximum spring constant.

    Returns:
        k_el: The calculated spring constant.
    """

    # Linear mapping
    k_el = k_max + (k_min - k_max) *(curvature)

    return k_el


def calculate_curvature(X, Y, Z, x_star, y_star,dx,dy):
    """
    Calculates the curvature at each image point using finite differences.

    Args:
        X: 1D array of x-coordinates of the grid points.
        Y: 1D array of y-coordinates of the grid points.
        Z: 2D array representing the potential energy surface.
        x_star: 1D array of x-coordinates of the image points.
        y_star: 1D array of y-coordinates of the image points.

    Returns:
        curvature: 1D array of curvature values at each image point.
    """

    nreplica = len(x_star)
    curvature = np.zeros(nreplica)

    # Calculate gradients at each image point
    grad = find_gradient(X, Y, Z, x_star, y_star, nreplica, dx, dy)  

    # Estimate curvature using finite differences
    for i in range(1, nreplica - 1):
        # Forward and backward differences for first derivatives
        dx_forward = x_star[i + 1] - x_star[i]
        dx_backward = x_star[i] - x_star[i - 1]
        dy_forward = y_star[i + 1] - y_star[i]
        dy_backward = y_star[i] - y_star[i - 1]

        # Central differences for second derivatives
        d2x = (dx_forward - dx_backward) / 2  
        d2y = (dy_forward - dy_backward) / 2

        # Curvature formula (assuming arc-length parameterization)
        curvature[i] = np.abs(
            (d2y * grad[0, i] - d2x * grad[1, i]) 
            / (grad[0, i]**2 + grad[1, i]**2)**(3/2)
        )

    # Handle endpoints (curvature set to zero for now, you can refine this)
    curvature[0] = 0
    curvature[-1] = 0

    return curvature

#@jit(nopython=True)
def get_force(X,Y,gradient,nreplica,k_el,curvature):


    '''
    Compute forces based on the elastic band method (G. Henkelman, H. Jónsson, DOI:10.1063/1.1323224)
                                                                                                               '''                                                                                                                              
    total_force = np.zeros((2,nreplica))
    k_max = 800
    k_min = 100
    
    for i in range(1,nreplica-1):
    
        #find unit vectors tangent to the path
        dr1 = np.array([X[i], Y[i]]) - np.array([X[i-1], Y[i-1]])
        dr2 = np.array([X[i+1], Y[i+1]]) - np.array([X[i], Y[i]])
        norm_dr1 = np.sqrt(np.sum(np.power(dr1, 2)))
        norm_dr2 = np.sqrt(np.sum(np.power(dr2, 2)))       
        tangent = dr1/norm_dr1 + dr2/norm_dr2
        norm_tangent =  np.sqrt(np.sum(np.power(tangent, 2)))
        versor_tangent = tangent/norm_tangent
        
        # Adaptive spring constant
        k_el = spring_constant_function(curvature[i],k_min,k_max)
    
        # compute elastic force parallel to the path (spring force)
        square_versor = np.dot(versor_tangent,versor_tangent)
        force_parallel = k_el*np.dot((dr2-dr1),square_versor)
           
        # compute force perpendicular to the path (true force)
        force_perpendicular =  gradient[:,i] -  np.dot(gradient[:,i],versor_tangent)
        
        # compute total force
        total_force[:,i] = force_parallel - force_perpendicular
        

    return  total_force



def plot_movie(X,Y,Z,pos_store,z_star_hist,length):

    [ x, y ] = np.meshgrid(X,Y)

    # customize colormap
    arr = np.linspace(0, 50, 100).reshape((10, 10))    
    cmap = plt.get_cmap('RdYlBu_r')
    new_cmap = truncate_colormap(cmap, 0.1, 0.9)

    # Postprocessing 
    fig,ax = plt.subplots(1,2)

    for i in range(0,length):
    
        ax[0].cla()
        ax[1].cla()            
        line=interp1d(x=pos_store[0,:,i],y=z_star_hist[:,i], kind=2)
        x2 = np.linspace(start=pos_store[0,0,i], stop=pos_store[0,-1,i], num=1000)
        y2 = line(x2)        
        ax[0].plot(pos_store[0,:,i],pos_store[1,:,i], color='black', marker='o', label='line with marker')
        cp=ax[0].contourf(x,y,Z,levels=range(0,50,1),cmap=new_cmap,antialiased=False,alpha=0.8)    
        ax[0].set_ylim([np.min(Y),np.max(Y)])
        ax[0].set_xlim([np.min(X),np.max(X)])
        ax[0].set_ylabel('CV2',fontsize=11)
        ax[0].set_xlabel('CV1',fontsize=11)
        ax[0].set_title(' X-Y Path ')
        ax[1].scatter(pos_store[0,:,i],z_star_hist[:,i], marker='o', color='red')
        ax[1].plot(x2,y2,color='b')
        ax[1].set_ylim([0,30])
        ax[1].set_xlim([0,12])
        ax[1].set_xlabel('d(C-C) [Bohr]')
        ax[1].set_title('Minimum Free Energy Path [kcal/mol]')
    
        plt.pause(0.0001)                                             
    
    

    plt.show()

    return


def climbing_image(X,Y,Z,tspan,dt,mass,nreplica,k_el, indice_1, indice_2,spacing, climbing_replica, pos_old, z_old,flag):

    """
    Performs Climbing Image Nudged Elastic Band (CI-NEB) optimization on a 2D potential energy surface.

    Input Args:

        X: 1D array of x-coordinates defining the grid for the potential energy surface.
        Y: 1D array of y-coordinates defining the grid for the potential energy surface.
        Z: 2D array representing the potential energy values on the grid (Z[i, j] is the energy at (X[i], Y[j])).
        tspan: Total number of time steps for the optimization.
        dt: Time step for each iteration.
        mass: Fictitious mass associated with each replica (image) in the NEB chain.
        nreplica: Number of replicas (images) used to discretize the reaction path.
        k_el: Base spring constant for the elastic band connecting the images.
        indice_1: Tuple (i, j) representing the indices of the starting minimum in the `Z` array.
        indice_2: Tuple (i, j) representing the indices of the ending minimum in the `Z` array.
        spacing: Desired number of images between the starting and ending minima.
        climbing_replica: Index (1-based) of the replica designated as the "climbing image."
        pos_old: 2D array of initial positions for all replicas (shape: (2, nreplica)).
        z_old: 1D array of initial potential energy values for all replicas.
        flag: Integer flag controlling the optimization mode:
            0: Steepest descent relaxation.
            1: Climbing image NEB.
            2: Standard NEB (elastic band).

    Output Args:

        pos_store: 3D array storing the positions of all replicas at each time step (shape: (2, nreplica, tspan)).
        z_star_hist: 2D array storing the potential energy values of all replicas at each time step (shape: (nreplica, tspan)).
        k: The final iteration number reached in the optimization.
    """

    [ x, y ] = np.meshgrid(X,Y)

    dx = np.abs( X[0] - X[1] )
    dy = np.abs( Y[0] - Y[1] )

    # Initialize vector with random velocities
    v0 = np.random.rand(2,nreplica)*(-0.5)

    # Hinged points are fixed
    v0[0,0] = 0
    v0[0,-1] = 0
    v0[1,0] = 0
    v0[1,-1] = 0
    v = v0
    mass = 1

    pos_store = np.zeros((2,nreplica,tspan))
    pos_store[:,:,0] = np.reshape(np.array(pos_old[:,:]),(2,nreplica)) 
    z_star_hist = np.zeros((nreplica,tspan))
    z_star_hist[:,0] =  np.reshape(np.array(z_old[:]),nreplica) 

    alfa = 0.0001 # alfa = 0.001
    threshold = 5e-3
    x_star = np.reshape(np.array(pos_old[0,:]),nreplica)
    y_star = np.reshape(np.array(pos_old[1,:]),nreplica)

    xx = x_star
    yy = y_star

    # Initialize force
    grad = find_gradient(X,Y,Z,x_star,y_star,nreplica,dx,dy)

    # Switch direction for climbing replica 
    try: 
        if flag == 1:
    
           grad[:,:][0][climbing_replica-1] = - grad[:,:][0][climbing_replica-1]
           grad[:,:][1][climbing_replica-1] = - grad[:,:][1][climbing_replica-1]
        
        elif flag == 0 or flag == 2: # Steepest descent
            pass
        else:
            raise ValueError("Invalid flag value. Must be 0,1 or 2.")
    except ValueError as e:
        print(f"Error: {e}")
        return

    curvature = calculate_curvature(X,Y,Z,x_star,y_star,dx,dy)    
    total_force = get_force(xx,yy,grad,nreplica,k_el,curvature)

    k = 0
    # time loop
    for j in range(1,tspan):
        
        if flag == 2:

           v_half = v+dt/2*total_force/mass
           pos = np.array([xx, yy]) + dt*v_half

        else:

           pos = np.array([xx, yy]) + alfa*total_force
           
        pos_store[:,:,j] = pos

        x_star = pos[0][:]    
        y_star = pos[1][:] 
        
        z_star = interp_bilinear(X,Y,Z,x_star,y_star,nreplica)
        z_star_hist[:,j] = z_star  
        z_star_hist[0,j] = z_star_hist[0,0]
        z_star_hist[-1,j] = z_star_hist[-1,0]
   
        # Update Gradient
        grad = find_gradient(X,Y,Z,x_star,y_star,nreplica,dx,dy)

        try:
            if flag == 1:
        
               grad[:,:][0][climbing_replica-1] = - grad[:,:][0][climbing_replica-1]
               grad[:,:][1][climbing_replica-1] = - grad[:,:][1][climbing_replica-1]
    
            elif flag == 0 or flag == 2: # Steepest descent
                pass
            else:
                raise ValueError("Invalid flag value. Must be 0,1 or 2.")
        except ValueError as e:
            print(f"Error: {e}")
            return 

            
        curvature = calculate_curvature(X,Y,Z,x_star,y_star,dx,dy)
        total_force = get_force(xx,yy,grad,nreplica,k_el,curvature)
        
        try:
            if flag == 2:
    
               # Compute Max Force
               max_force = 0
               for i in range(1,nreplica-1):
       
                   max_force = max_force + np.sqrt(np.sum(np.power(total_force[:,i], 2)))
                          
               if max_force < threshold:
         
                  print("Equilibrium positions found for Max Force threshold: "+str(threshold)+"")
       
                  break
       
               # Update Variables
               v_new = v_half + dt/2*total_force/mass
               v = v_new
                
            elif flag == 0 or flag == 1:
                    
               # Check Convergence
               norm_pos = np.sqrt(np.sum(np.power((pos_store[:,:,j]-pos_store[:,:,j-1]), 2)))
       
               if norm_pos < threshold:
                                                                                                   
                  print("Equilibrium positions found for threshold: "+str(threshold)+"")
                                                                                                   
                  break
            else:
                raise ValueError("Invalid flag value. Must be 0, 1, or 2.")
        except ValueError as e:
            print(f"Error: {e}")
            return
   
        xx = x_star
        yy = y_star
        
        k = k + 1

    fig,ax = plt.subplots(1,2, figsize=(8, 4))   

    ax[0].cla()
    ax[1].cla()            
    line=interp1d(x=np.linspace(1,nreplica,nreplica),y=z_star_hist[:,k], kind=2)
    x2 = np.linspace(start=1, stop=nreplica, num=1000)
    y2 = line(x2)        
    ax[0].plot(pos_store[0,:,k],pos_store[1,:,k], color='black', marker='o', label='line with marker')
    cp=ax[0].contourf(x,y,Z,levels=range(0,150,5),cmap='viridis',antialiased=False,alpha=0.9)    
    ax[0].set_ylim([np.min(Y), np.max(Y)])
    ax[0].set_xlim([np.min(X), np.max(X)])
    ax[0].set_ylabel('CV2',fontsize=11)
    ax[0].set_xlabel('CV1',fontsize=11)
    ax[0].set_title(' X-Y Path ')
    ax[1].scatter(np.linspace(1,nreplica,nreplica),z_star_hist[:,k], marker='o', color='red')
    ax[1].plot(x2,y2,color='b')
    ax[1].set_ylim([0,np.max(z_star_hist[:,k])+10])
    ax[1].set_xlim([1,nreplica])
    ax[1].set_xlabel('Reaction Coodinate', fontsize=11)
    ax[1].set_title('Minimum Free Energy Path [kJ/mol]')
    
    #plt.show()
    
    return  pos_store, z_star_hist, k
