
import numpy as np
# from pde_find import *
from numba import jit,njit

def FiniteDiff(u, dx):
    
    n = u.size
    ux = np.zeros(n)

    # for i in range(1, n - 1):
    ux[1:n-1] = (u[2:n] - u[0:n-2]) / (2 * dx)

    ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
    ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
    return ux


def FiniteDiff2(u, dx):

    n = u.size
    ux = np.zeros(n)

    ux[1:n-1] = (u[2:n] - 2 * u[1:n-1] + u[0:n-2]) / dx ** 2

    ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
    ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
    return ux



# @jit(nopython=True)  
def Diff(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))
    
    if len(dxt.shape) == 2:
        dxt = dxt[:,0]
    if name == 'x':
        dxt = dxt[2]-dxt[1]
        # for i in range(m):
        #     uxt[:, i] = FiniteDiff(u[:, i], dxt)
        uxt[1:n-1,:] = (u[2:n,:] - u[0:n-2,:]) / (2 * dxt)

        uxt[0,:] = (-3.0 / 2 * u[0,:] + 2 * u[1,:] - u[2,:] / 2) / dxt
        uxt[n - 1,:] = (3.0 / 2 * u[n - 1,:] - 2 * u[n - 2,:] + u[n - 3,:] / 2) / dxt
    # elif name == 't':
    #     for i in range(n):
    #         uxt[i, :] = FiniteDiff(u[i, :], dxt)

    else:
        assert False
        NotImplementedError()

    return uxt

# @jit(nopython=True)  
def Diff2(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """
    if len(dxt.shape) == 2:
        dxt = dxt[:,0]
    n, m = u.shape
    uxt = np.zeros((n, m))
    dxt = dxt[2]-dxt[1]
    if name == 'x':

        uxt[1:n-1,:] =(u[2:n,:] - 2 * u[1:n-1,:] + u[0:n-2,:]) / dxt ** 2

        uxt[0,:] = (2 * u[0,:] - 5 * u[1,:] + 4 * u[2,:] - u[3,:]) / dxt ** 2
        uxt[n - 1,:] = (2 * u[n - 1,:] - 5 * u[n - 2,:] + 4 * u[n - 3,:] - u[n - 4,:]) / dxt ** 2
    else:
        assert False
        NotImplementedError()

    return uxt

# @jit(nopython=True)  
def Diff3(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        uxt=Diff2(u,dxt,dim,name)
        uxt = Diff(uxt,dxt, dim, name)
        # dxt = dxt[2]-dxt[1]

        # for i in range(m):
        #     uxt[:, i] = FiniteDiff2(u[:, i], dxt)
        #     uxt[:,i] = FiniteDiff(uxt[:,i],dxt )

    else:
        assert False
        NotImplementedError()

    return uxt

# @jit(nopython=True)  
def Diff4(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    
    if name == 'x':
        uxt=Diff2(u,dxt,dim,name)
        uxt = Diff2(uxt,dxt, dim, name)
    else:
        assert False
        NotImplementedError()
 

    return uxt



def Diff_2(u, dxt, name=1):
    """
    Here dx is a scalar, name is a str indicating what it is
    """
    if u.shape == dxt.shape:
        return u/dxt
    t,n,m = u.shape
    uxt = np.zeros((t, n, m))
    if len(dxt.shape) == 2:
        dxt = dxt[:,0]
    dxt = dxt.ravel()
    # import pdb;pdb.set_trace()
    if name == 1:
        dxt = dxt[2]-dxt[1]
        uxt[:,1:n-1,:] = (u[:,2:n,:]-u[:,:n-2,:])/2/dxt
        
        uxt[:,0,:] = (u[:,1,:]-u[:,-1,:])/2/dxt
        uxt[:,-1,:] = (u[:,0,:]-u[:,-2,:])/2/dxt
    elif name == 2:
        dxt = dxt[2]-dxt[1]
        uxt[:,:,1:m-1] = (u[:,:,2:m]-u[:,:,:m-2])/2/dxt
        
        uxt[:,:,0] = (u[:,:,1]-u[:,:,-1])/2/dxt
        uxt[:,:,-1] = (u[:,:,0]-u[:,:,-2])/2/dxt
        # uxt[:,:,0] = (-3.0 / 2 * u[:,:,0] + 2 * u[:,:,1] - u[:,:,2] / 2) / dxt
        # uxt[:,:,n - 1] = (3.0 / 2 * u[:,:,n - 1] - 2 * u[:,:,n - 2] + u[:,:,n - 3] / 2) / dxt
    else:
        assert False, 'not supported'     

    return uxt

# @jit(nopython=True)
def Diff2_2(u, dxt, name=1): 
    """
    Here dx is a scalar, name is a str indicating what it is
    """
  
    
    if u.shape == dxt.shape:
        return u/dxt
    t,n,m = u.shape
    uxt = np.zeros((t, n, m))
    dxt = dxt.ravel()
    # try: 
    if name == 1:
        dxt = dxt[2]-dxt[1]
        uxt[:,1:n-1,:]= (u[:,2:n,:] - 2 * u[:,1:n-1,:] + u[:,0:n-2,:]) / dxt ** 2
        uxt[:,0,:] = (u[:,1,:]+u[:,-1,:]-2*u[:,0,:])/dxt ** 2
        uxt[:,-1,:] = (u[:,0,:]+u[:,-2,:]-2*u[:,-1,:])/dxt ** 2
        # uxt[:,0,:] = (2 * u[:,0,:] - 5 * u[:,1,:] + 4 * u[:,2,:] - u[:,3,:]) / dxt ** 2
        # uxt[:,n - 1,:] = (2 * u[:,n - 1,:] - 5 * u[:,n - 2,:] + 4 * u[:,n - 3,:] - u[:,n - 4,:]) / dxt ** 2
    elif name == 2:
        dxt = dxt[2]-dxt[1]
        uxt[:,:,1:m-1]= (u[:,:,2:m] - 2 * u[:,:,1:m-1] + u[:,:,0:m-2]) / dxt ** 2
        uxt[:,:,0] = (u[:,:,1]+u[:,:,-1]-2*u[:,:,0])/dxt ** 2
        uxt[:,:,-1] = (u[:,:,0]+u[:,:,-2]-2*u[:,:,-1])/dxt ** 2  
        # uxt[:,:,0] = (2 * u[:,:,0] - 5 * u[:,:,1] + 4 * u[:,:,2] - u[:,:,3]) / dxt ** 2
        # uxt[:,:,n - 1] = (2 * u[:,:,n - 1] - 5 * u[:,:,n - 2] + 4 * u[:,:,n - 3] - u[:,:,n - 4]) / dxt ** 2
        
    else:
        raise NotImplementedError(f"Diff2_2: name={name} not supported")
# except:
    #     import pdb;pdb.set_trace()

    return uxt

@jit(nopython=True)
def Laplace(u,x):
    x1,x2 = x
    uxt = Diff2_2(u,x1, name = 1)
    uxt += Diff2_2(u,x2, name = 2)
    return uxt


def Laplace_3(u, x):
    """3D Laplacian: nabla^2 u = d2u/dx2 + d2u/dy2 + d2u/dz2"""
    x1, x2, x3 = x
    uxt = Diff2_3(u, x1, name=1)
    uxt += Diff2_3(u, x2, name=2)
    uxt += Diff2_3(u, x3, name=3)
    return uxt


def Diff_3(u, dxt, name=1):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    # import pdb;pdb.set_trace()
    if u.shape == dxt.shape:
        return u/dxt
    t,n,m,p = u.shape
    uxt = np.zeros((t, n, m, p))
    if len(dxt.shape) == 2:
        dxt = dxt[:,0]
    dxt = dxt.ravel()
    # import pdb;pdb.set_trace()
    dxt = dxt[2]-dxt[1]
    if name == 1:
        uxt[:,1:n-1,:,:] = (u[:,2:n,:,:]-u[:,:n-2,:,:])/2/dxt
        
        # uxt[:,0,:,:] = (u[:,1,:,:]-u[:,-1,:,:])/2/dxt
        # uxt[:,-1,:,:] = (u[:,0,:,:]-u[:,-2,:,:])/2/dxt
        uxt[:,0,:,:] = (-3.0 / 2 * u[:,0,:,:] + 2 * u[:,1,:,:] - u[:,2,:,:] / 2) / dxt
        uxt[:,n - 1,:,:] = (3.0 / 2 * u[:,n - 1,:,:] - 2 * u[:,n - 2,:,:] + u[:,n - 3,:,:] / 2) / dxt
    elif name == 2:
        
        uxt[:,:,1:m-1,:] = (u[:,:,2:m,:]-u[:,:,:m-2,:])/2/dxt
        
        # uxt[:,:,0,:] = (u[:,:,1,:]-u[:,:,-1,:])/2/dxt
        # uxt[:,:,-1,:] = (u[:,:,0,:]-u[:,:,-2,:])/2/dxt
        uxt[:,:,0,:] = (-3.0 / 2 * u[:,:,0,:] + 2 * u[:,:,1,:] - u[:,:,2,:] / 2) / dxt
        uxt[:,:,m- 1,:] = (3.0 / 2 * u[:,:,m - 1,:] - 2 * u[:,:,m - 2,:] + u[:,:,m - 3,:] / 2) / dxt
    elif name == 3:
        uxt[:,:,:,1:p-1] = (u[:,:,:,2:p]-u[:,:,:,:p-2])/2/dxt
        uxt[:,:,:,0] = (-3.0 / 2 * u[:,:,:,0] + 2 * u[:,:,:,1] - u[:,:,:,2] / 2) / dxt
        uxt[:,:,:,p - 1] = (3.0 / 2 * u[:,:,:,p - 1] - 2 * u[:,:,:,p - 2] + u[:,:,:,p - 3] / 2) / dxt
    else:
        assert False, 'not supported'     

    return uxt

# @jit(nopython=True)
def Diff2_3(u, dxt, name=1): 
    """
    Here dx is a scalar, name is a str indicating what it is
    """
  
    
    if u.shape == dxt.shape:
        return u/dxt
    t,n,m,p = u.shape
    uxt = np.zeros((t, n, m,p))
    dxt = dxt.ravel()
    # try: 
    dxt = dxt[2]-dxt[1]
    if name == 1:
        
        uxt[:,1:n-1,:,:]= (u[:,2:n,:,:] - 2 * u[:,1:n-1,:,:] + u[:,0:n-2,:,:]) / dxt ** 2
        # uxt[:,0,:,:] = (u[:,1,:,:]+u[:,-1,:,:]-2*u[:,0,:,:])/dxt ** 2
        # uxt[:,-1,:,:] = (u[:,0,:,:]+u[:,-2,:,:]-2*u[:,-1,:,:])/dxt ** 2
        uxt[:,0,:,:] = (2 * u[:,0,:,:] - 5 * u[:,1,:,:] + 4 * u[:,2,:,:] - u[:,3,:,:]) / dxt ** 2
        uxt[:,n - 1,:,:] = (2 * u[:,n - 1,:,:] - 5 * u[:,n - 2,:,:] + 4 * u[:,n - 3,:,:] - u[:,n - 4,:,:]) / dxt ** 2
    elif name == 2:

        uxt[:,:,1:m-1,:]= (u[:,:,2:m,:] - 2 * u[:,:,1:m-1,:] + u[:,:,0:m-2,:]) / dxt ** 2
        # uxt[:,:,0,:] = (u[:,:,1,:]+u[:,:,-1,:]-2*u[:,:,0,:,:])/dxt ** 2
        # uxt[:,:,-1,:] = (u[:,:,0,:]+u[:,:,-2,:]-2*u[:,:,-1,:])/dxt ** 2  
        uxt[:,:,0,:] = (2 * u[:,:,0,:] - 5 * u[:,:,1,:] + 4 * u[:,:,2,:] - u[:,:,3,:]) / dxt ** 2
        uxt[:,:,m - 1,:] = (2 * u[:,:,m - 1,:] - 5 * u[:,:,m - 2,:] + 4 * u[:,:,m - 3,:] - u[:,:,m - 4,:]) / dxt ** 2
    elif name == 3:
        # Fixed: original DISCOVER had same bug (m instead of p)
        uxt[:,:,:,1:p-1]= (u[:,:,:,2:p] - 2 * u[:,:,:,1:p-1] + u[:,:,:,0:p-2]) / dxt ** 2
        # uxt[:,:,:,0] = (u[:,:,:,1]+u[:,:,:,-1]-2*u[:,:,:,0])/dxt ** 2
        # uxt[:,:,-1] = (u[:,:,0]+u[:,:,-2]-2*u[:,:,-1])/dxt ** 2  
        # Fixed: original DISCOVER used 3D index [:,:,0] instead of 4D [:,:,:,0]
        uxt[:,:,:,0] = (2 * u[:,:,:,0] - 5 * u[:,:,:,1] + 4 * u[:,:,:,2] - u[:,:,:,3]) / dxt ** 2
        uxt[:,:,:,p - 1] = (2 * u[:,:,:,p - 1] - 5 * u[:,:,:,p - 2] + 4 * u[:,:,:,p - 3] - u[:,:,:,p- 4]) / dxt ** 2
    else:
        raise NotImplementedError(f"Diff2_3: name={name} not supported")
    return uxt  # Fixed: original DISCOVER missing return in this function