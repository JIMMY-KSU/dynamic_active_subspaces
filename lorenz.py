from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import active_subspaces as ac
import dynamic_AS as dy
import quadrature as qdt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(Y, times, r, s, b):
    """Gronwall system of ODEs for states and partials of the lorenz system
    Parameters
    ----------
    Y: ndarray
        12-by-1 array of states and partials of states wrt parameters
    times: ndarray
        time array 
    r, s, b: float
        parameter values for which the system should be evaluated
        
    Returns
    -------
    dYdt: ndarray
        12-by-1 array of ODEs
    """
    x, y, z, dxdr, dydr, dzdr, dxds, dyds, dzds, dxdb, dydb, dzdb=Y
    dYdt=np.array([
    #states
    s * (y - x),\
    x * (r - z) -y,\
    x * y - b * z,\
    
    #partials wrt rho
    (-s * dxdr) + (s * dydr),\
    (r * dxdr) - dydr - (x * dzdr) + x,\
    (y * dxdr) + (x * dydr) - (b * dzdr),\
    
    #partials wrt sigma
    (-s * dxds) + (s * dyds) + (y - x),\
    (r * dxds) - dyds - (x * dzds),\
    (y * dxds) + (x * dyds) - (b * dzds),\
    
    #partials wrt beta
    (-s * dxdb) + (s * dydb),\
    (r * dxdb) - dydb - (x * dzdb),\
    (y * dxdb) + (x * dydb) - (b * dzdb) - z
    ])
    return dYdt
    
def l_reg(Y, times, r, s, b):
    """System of ODEs for states the lorenz system
    Parameters
    ----------
    Y: ndarray
        3-by-1 array of states 
    times: ndarray
        time array 
    r, s, b: float
        parameter values for which the system should be evaluated
        
    Returns
    -------
    dYdt: ndarray
        3-by-1 array of ODEs
    """
    x, y, z =Y
    
    return s * (y - x), x * (r - z) - y, x * y - b * z 

    
def lorenz_fun(x, times, y0, sys, q):
    """evaluates the QOI for given parameters x
    Parameters
    ----------
    x: ndarray
        M-by-m array of input parameters at which points the function should be evaluated
    times: ndarray
        time array
    y0: ndarray
        1-by-3 array of initial conditions for sys
    sys: ndarray
        1-by-3 ODE system (NOT GRONWALL--unnecessary expense)
    q: int
        index corresponding to QOI. 1, 2, 3= x, y, z respectively
        
    Returns
    -------
    f: ndarray
        len(times)-by-M array of solutions for QOI over time.
    """
    f = np.zeros([len(times), len(x)])
    for i in range(len(x)):
        r, s, b = x[i, :]
        sol = odeint(sys, y0, times, args = (r, s, b))
        f[:, i] = sol[:, q]
    return f 
     
r, s, b = 28, 10, (8/3)
nominal = np.array([r, s, b]) 
Yl = .9 * nominal; Yu = 1.1 * nominal
Yl = Yl.reshape((1, 3)); Yu = Yu.reshape((1, 3))   
y0 = [1.0, 1.0, 1.0]
yg0 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
times = np.arange(0.0, 8.0, 0.02)
m = 3
q = 2 #state
M = 729
c = 'fdm'
h = 10 ** -8
filename = 'fdm_long_lorenz'
eig, w_1, XX, f, w_2, w_3, fmin, fmax, y = dy.dynamo(times, y0, l_reg, m, Yl, Yu, lorenz_fun, y0, l_reg, q, c, h)

for i in range(len(times)):
    dy.dynanimate(i, eig, m, w_1, XX, M, f, w_2, w_3, times, fmin, fmax, filename) 

"""
states = odeint(lorenz, yg0, times,args=(r,s,b))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:,0], states[:,1], states[:,2])
plt.show()
"""