from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import active_subspaces as ac
import quadrature as qdt

def log_gronwall_gradient(sys, yg0, times, x, k, G, y, Yu, Yl):
    """Approximate the log-transform gradient 
    Parameters
    ----------
    sys : ndarray
        Gronwall system of ODEs specific to the dynamical system
    yg0 : ndarray
        initial conditions for the Gronwall system
    times : ndarray
        time array used in odeint
    x : ndarray
        1-by-len(x) array of parameters
    k : int
        index in range M corresponding to the kth input parameters
    G : ndarray 
        len(times)-by-M-by-len(x) array. each M-by-len(x) matrix corresponds to the gradient
        approximation at time t.
    y : ndarray
        M-by-len(x) array of log transformed input parameters
    Yl: ndarray
        1-by-m array of lower bounds on parameter space
    Yu: ndarray
        1-by-m array of upper bounds on parameter space
    Returns
    -------
    G : ndarray 
        len(times)-by-M-by-len(x) array. each M-by-len(x) matrix corresponds to the gradient
        approximation at time t. after one iteration the kth column of each M-by-len(x) matrix 
        will contain the gradient approximation.
        
    Notes
    -------
        the gradient is specific to the dynamical system. the odeint() args() input must be 
        changed dependent on the system and input parameters. the line g=np.array( [ ] ) must 
        also be changed if the QOI is changed and if the dimension changes. 
    """
    l = len(x)
    r, s, b = x
    sola = odeint(sys, yg0, times, args = (r, s, b))
    
    for t in range(1, len(times)):
        #NOTE! THE FOLLOWING 'g' IS SPECIFIC TO sys, TO THE QOI, AND THE PARAMETER SPACE DIMENSION.
        #IT MUST BE CHANGED ACCORDINGLY.
        g = np.array( [sola[t, 5], sola[t, 8], sola[t, 11]] )
        g = g * (np.log10(Yu)-np.log10(Yl))/2
        tg = g * (10 ** y[k, :] * np.log(10))
        nor = np.linalg.norm(tg)
        
        if (nor == 0):
            tg = tg
            print 'Norm is zero.' 
            print t
        else:
            tg = tg / nor
            
        tg = tg.reshape(l, )
        G[t][k, :] = tg.transpose()
    return G 
    
def log_fourth_order_FDM(sys, y0, times, x, h, G, k, q, Yu, Yl, y):
    """ computes a fourth order FDM log-transformed gradient of the given dynamical system
    Parameters
    ----------
    sys : ndarray
        system of ODEs specific to the dynamical system
    y0 : ndarray
        initial conditions for the ODE system
    times : ndarray
        time array used in odeint
    x : ndarray
        1-by-len(x) array of parameters
    k : int
        index in range M corresponding to the kth input parameters
    h : int
        stepsize for FDM
    G : ndarray 
        len(times)-by-M-by-len(x) array. each M-by-len(x) matrix corresponds to the gradient
        approximation at time t.
    q : int
        corresponds to the index of the QOI in the solution
    y : ndarray
        M-by-len(x) array of log transformed input parameters
    Yl : ndarray
        1-by-m array of lower bounds on parameter space
    Yu : ndarray
        1-by-m array of upper bounds on parameter space
        
    Returns
    -------
    G : ndarray 
        len(times)-by-M-by-len(x) array. each M-by-len(x) matrix corresponds to the gradient
        approximation at time t. after one iteration the kth column of each M-by-len(x) matrix 
        will contain the gradient approximation.
        
    Notes
    -------
        the gradient is specific to the dynamical system. the odeint() args() input must be changed
        dependent on the system and input parameters. the line where we vstack() the partials of the 
        state with respect to the parameters is also specific to a 3-dimensional parameter space. if
        this is not the case, it must be changed.
    
    """
    r, s, b = x
    l = len(x)
    #f'(x)=(-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h))/12h
    
    plustwor = odeint(sys, y0, times, args = (r + (2 * h), s, b))
    minustwor = odeint(sys, y0, times, args = (r - (2 * h) ,s ,b))
    plusoner = odeint(sys, y0, times, args = (r + (h), s, b))
    minusoner = odeint(sys, y0, times, args = (r - (h), s, b))
    
    plustwos = odeint(sys, y0, times, args = (r, s + (2 * h), b))
    minustwos = odeint(sys, y0, times, args = (r, s - (2 * h), b))
    plusones = odeint(sys, y0, times, args = (r, s + (h), b))
    minusones = odeint(sys, y0, times, args = (r, s - (h), b))
    
    plustwob = odeint(sys, y0, times, args = (r, s, b + (2 * h)))
    minustwob = odeint(sys, y0, times, args = (r, s, b - (2 * h)))
    plusoneb = odeint(sys, y0, times, args = (r, s, b + h))
    minusoneb = odeint(sys, y0, times, args = (r, s, b - h))

        
    dESdr = (-plustwor[:, q] + 8 * plusoner[:, q] - 8 * minusoner[:, q] + minustwor[:, q]) / (12 * h)
    dESds = (-plustwos[:, q] + 8 * plusones[:, q] - 8 * minusones[:, q] + minustwos[:, q]) / (12 * h)
    dESdb = (-plustwob[:, q] + 8 * plusoneb[:, q] - 8 * minusoneb[:, q] + minustwob[:, q]) / (12 * h)
    
    for t in range(1, len(times)):
        #NOTE! THE FOLLOWING 'g' IS SPECIFIC TO sys, TO THE QOI, AND THE PARAMETER SPACE DIMENSION.
        #IT MUST BE CHANGED ACCORDINGLY.
        g = np.vstack((dESdr[t], dESds[t], dESdb[t]))
        g = g.reshape(l, )
        g = g * (np.log10(Yu) - np.log10(Yl)) / 2
        tg = g * (10 ** y[k, :] * np.log(10))
        nor = np.linalg.norm(tg)
        
        if (nor == 0):
            tg = tg
            print 'Norm is zero.' 
            print t
        else:
            tg = tg / nor
            
        tg = tg.reshape(l, )
        G[t][k, :] = tg.transpose()
    return G 

    
def finite_diff_grad(sys, y0, times, x, h, G, k, q, Yu, Yl, y):
    
    """ computes a second order FDM log-transformed gradient of the given dynamical system
    Parameters
    ----------
    sys : ndarray
        system of ODEs specific to the dynamical system
    y0 : ndarray
        initial conditions for the ODE system
    times : ndarray
        time array used in odeint
    x : ndarray
        1-by-len(x) array of parameters
    k : int
        index in range M corresponding to the kth input parameters
    h : int
        stepsize for FDM
    G : ndarray 
        len(times)-by-M-by-len(x) array. each M-by-len(x) matrix corresponds to the gradient
        approximation at time t.
    q : int
        corresponds to the index of the QOI in the solution
    y : ndarray
        M-by-len(x) array of log transformed input parameters
    Yl : ndarray
        1-by-m array of lower bounds on parameter space
    Yu : ndarray
        1-by-m array of upper bounds on parameter space
        
    Returns
    -------
    G : ndarray 
        len(times)-by-M-by-len(x) array. each M-by-len(x) matrix corresponds to the gradient
        approximation at time t. after one iteration the kth column of each M-by-len(x) matrix 
        will contain the gradient approximation.
        
    Notes
    -------
        the gradient is specific to the dynamical system. the odeint() args() input must be changed
        dependent on the system and input parameters. the line where we vstack() the partials of the 
        state with respect to the parameters is also specific to a 3-dimensional parameter space. if
        this is not the case, it must be changed.
    
    """
    r, s, b = x
    l = len(x)
    
    firstr = odeint(sys, y0, times, args = (r + h, s, b))
    secondr = odeint(sys, y0, times, args = (r - h, s, b))
    firsts = odeint(sys, y0, times, args = (r, s + h, b))
    seconds = odeint(sys, y0, times, args = (r, s - h, b))
    firstb = odeint(sys, y0, times, args = (r, s, b + h))
    secondb = odeint(sys, y0, times, args = (r, s, b - h))
    
    ES1r = firstr[:, q]; ES2r = secondr[:, q]; ES1s = firsts[:, q]; ES2s = seconds[:, q];
    ES1b = firstb[:, q]; ES2b = secondb[:, q]
    dESdr = (ES1r - ES2r) / (2 * h); dESds = (ES1s - ES2s) / (2 * h); dESdb = (ES1b - ES2b) / (2 * h);
    
    for t in range(1, len(times)):
        #NOTE! THE FOLLOWING 'g' IS SPECIFIC TO sys, TO THE QOI, AND THE PARAMETER SPACE DIMENSION.
        #IT MUST BE CHANGED ACCORDINGLY.
        g = np.vstack((dESdr[t], dESds[t], dESdb[t]))
        g = g.reshape(l, )
        g = g * (np.log10(Yu) - np.log10(Yl)) / 2
        tg = g * (10 ** y[k, :] * np.log(10))
        nor = np.linalg.norm(tg)
        
        if (nor == 0):
            tg = tg
            print 'Norm is zero.' 
            print t
        else:
            tg = tg / nor
            
        tg = tg.reshape(l, )
        G[t][k, :] = tg.transpose()
    return G
    
def approx_C(G, M, m):
    """ Approximate matrix C at time t with random sampling
    Parameters
    ----------
    M : int
        number of MC samples used in the random sampling.
    m: int
        input parameter dimension
    G : ndarray 
        M-by-m matrix corresponding to the gradient approximation at time t.
        
    Returns
    -------
    C : ndarray 
        m-by-m matrix approximation of C at time t
    """
    S = np.zeros([m, m])
    for i in range(M):
        g = G[i, :].reshape(m, 1) 
        temp = np.outer(g, g) 
        S = S + temp
    C = S / M
    return C

def sub_dist(W1, W2): 
    """ Compute distance between subspaces defined by eq 3.49
    Parameters
    ----------
    W1 : ndarray
        subspace 1
    W2: ndarray
        subspace 2
        
    Returns
    -------
    d : int
        distance between W1 and W2
    """
    d = np.linalg.norm(np.transpose(W1) * W2)
    return d
    
def bootstrap_replicate_C(G, M, j, m):
    """ Compute the bootstrap replicate C_i
    Parameters
    ----------
    G : ndarray
        M-by-m matrix corresponding to the gradient approximation at time t.
    M : int
        number of MC samples used in the random sampling.
    j : ndarray
        M-by-1 array of random indices
    m : int
        input parameter dimension
    Returns
    -------
    d : int
        distance between W1 and W2
    """
    S = np.zeros([m, m])
    for i in range(M):
        j_k = j[i]
        g = G[j_k, :].reshape(m, 1) 
        temp = np.outer(g, g)
        S = S + temp
    C = S / M
    return C

def sorted_eigh(C):
    """Compute eigenpairs and sort.
    
    Parameters
    ----------
    C : ndarray
        matrix whose eigenpairs you want
        
    Returns
    -------
    e : ndarray
        vector of sorted eigenvalues
    W : ndarray
        orthogonal matrix of corresponding eigenvectors
    
    Notes
    -----
    Eigenvectors are unique up to a sign. We make the choice to normalize the
    eigenvectors so that the first component of each eigenvector is positive.
    This normalization is very helpful for the bootstrapping. 
    """
    e, W = np.linalg.eigh(C)
    e = abs(e)
    ind = np.argsort(e)
    e = e[ind[::-1]]
    W = W[:, ind[::-1]]
    for i in range(np.shape(W)[0]):
        if (W[0, i] < 0):
            W[:, i] = W[:, i] * (-1.)
    return e.reshape((1, e.size)), W

def log_transform(yg0, times, sys, m, Yl, Yu, h, y0, q, c):
    """ computes C, eigenvalues, and eigenvectors at all time, log-transformed space y,
        and gauss-legendre weights XX
    Parameters
    ----------
    yg0 : ndarray
        initial conditions for the ODE system sys
    times : ndarray
        time array used in odeint
    sys : ndarray
        Gronwall system of ODEs specific to the dynamical system
        OR
        Regular system of ODEs specific to the dynamical system
        Depends on whether or not you are using Gronwall gradients or FDM gradients
    m: int
        input parameter dimension
    Yl : ndarray
        1-by-m array of lower bounds on parameter space
    Yu : ndarray
        1-by-m array of upper bounds on parameter space
    h : float
        stepsize for finite difference methods
    y0 : ndarray
        initial conditions for the regular ODE system 
    q : int
        corresponds to the index of the QOI in the solution
    c : string
        options are 'gron' or 'fdm'. defines which gradient approximation technique to use.
        
    Returns
    -------
    y: ndarray
        M-by-m array of log-transformed and shifted input parameters
    v_t: ndarray
        len(times)-by-m matrix of eigenvalues. each row is the eigenvalues of C at time t
    W_t: ndarray
        len(times)-by-m-by-m matrix. W_t[i] is an m-by-m matrix corresponding to the eigenvectors
        of C at time i.
    XX: ndarray
        M-by-m matrix of gauss_legendre nodes. Between [-1,1].
    
    Notes
    -------
        
    """
    M = 729
    C_t = np.zeros((len(times), m, m))
    G = np.zeros((len(times), M, m))
    y = np.zeros([M, m])
    v_t = np.zeros([len(times), m])
    W_t = np.zeros((len(times), m, m))
    
    XX, W = qdt.gauss_legendre([9, 9, 9])
    x = ((XX + 1) / 2) * (Yu - Yl) + Yl
    for k in range(M):
        for l in range(m):
            y[k, l] = np.log10(x[k, l])    
        a = 10 ** y[k, :]
        if(c == 'gron'):
            G = log_gronwall_gradient(sys, yg0, times, a, k, G, y, Yu, Yl)
        elif(c == 'fdm'):
            G = log_fourth_order_FDM(sys, y0, times, a, h, G, k, q, Yu, Yl, y)
        else:
            raise Exception('Please specify a gradient approximation method')
    for t in range(len(times)):
        C_t[t] = np.dot(G[t].transpose(), G[t] * W)
        v_t[t], W_t[t] = sorted_eigh(C_t[t])
    return y, v_t, W_t, XX

def dynanimate(i, eig, m, w_1, XX, M, f, w_2, w_3, times, fmin, fmax, filename):
    """plots and saves dynamic active subspace subplots
    Parameters
    ----------
    i : int
        time index
    eig : ndarray
        len(times)-by-m array of eigenvalues. 
    m : int
        input parameter dimension
    w_1 : ndarray
        m-by-1 array, first eigenvector of C
    w_2 : ndarray
        m-by-1 array, second eigenvector of C
    w_3 : ndarray
        m-by-1 array, third eigenvector of C
    XX : ndarray
        M-by-m matrix of gauss-legendre nodes. Between [-1,1].
    M : int
        total number of gauss-legendre nodes
    f : ndarray
        len(times)-by-M array of function evaluations for QOI at time i
    times : ndarray
        time array
    fmin : float
        minimum of the function
    fmax : float
        maximum of the function
    filename : string
        images to be saved under this name
        
    Returns
    -------
    saves plots at time i
    
    """
    t = len(times)
    fT = times[t - 1]
    eigv = eig[i, :].reshape(m, )
    
    #x-axis for SSP
    w = w_1[i, :]
    w = w.reshape(m, 1)
    x_ax = XX.dot(w)
    x_ax = x_ax.reshape(M, )
    
    #y-axis for 1D SSP
    y_ax = f[i, :]
    y_ax = y_ax.reshape(M, )
    
    #y-axis for 2D SSP
    av_2 = w_2[i, :]
    av_2 = av_2.reshape(m, )
    y_ax_2 = XX.dot(av_2)
    y_ax_2 = y_ax_2.reshape(M, )
    
    #Weights of first active variable
    av_1 = w_1[i, :]
    av_1 = av_1.reshape(m, )

    #Weights of second active variable
    av_2 = w_2[i, :]
    av_2 = av_2.reshape(m, )
    
    #Weights of inactive variable
    av_3 = w_3[i, :]
    av_3 = av_3.reshape(m, )
    
    #Colours for scatter plots
    col = f[i, :]
    col = col.reshape(M, )
    
    #Current time and colours for plotting individual weights
    tt = [times[i], times[i], times[i]]
    C = ['b', 'g', 'r']
    
    #Past and current time, eigenvalues, and variable weights
    partial_time = times[0:i]
    partial_eig_1 = eig[0:i, 0]
    partial_eig_2 = eig[0:i, 1]
    partial_eig_3 = eig[0:i, 2]
    part_w1_p = w_1[0:i, 0]
    part_w1_n = w_1[0:i, 1]
    part_w1_c = w_1[0:i, 2]
    part_w2_p = w_2[0:i, 0]
    part_w2_n = w_2[0:i, 1]
    part_w2_c = w_2[0:i, 2]
    part_w3_p = w_3[0:i, 0]
    part_w3_n = w_3[0:i, 1]
    part_w3_c = w_3[0:i, 2]
    
    #Defines figure size suitable for subplots
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 14
    fig_size[1] = 9
    fa, axarr = plt.subplots(2, 4)
    
    #Plots Eigenvalues in time. Future time is dashed, past time is filled in, and present time is big dots
    axarr[0, 0].set_xlim(0, fT)
    axarr[0, 0].set_ylim(10 ** -20, 100)
    axarr[0, 0].scatter(tt, eigv, c = C, s = 90)
    axarr[0, 0].semilogy(times, eig[:, 0], 'b', linestyle = ':')
    axarr[0, 0].semilogy(times, eig[:, 1], 'g', linestyle = ':')
    axarr[0, 0].semilogy(times, eig[:, 2], 'r', linestyle = ':')
    axarr[0, 0].semilogy(partial_time, partial_eig_1, 'b', linewidth = '3', label = r'$\lambda_1$')
    axarr[0, 0].semilogy(partial_time, partial_eig_2, 'g', linewidth = '3', label = r'$\lambda_2$')
    axarr[0, 0].semilogy(partial_time, partial_eig_3, 'r', linewidth = '3', label = r'$\lambda_3$')
    axarr[0, 0].set_title('Eigenvalues')
    axarr[0, 0].legend(loc = 'best', fontsize = 10)
    
    #Plots AV1 weights in time. Future time is dashed, past time is filled in, and present time is big dots
    axarr[0, 1].set_xlim(0, fT)
    axarr[0, 1].set_ylim(-1.8, 1.8)
    axarr[0, 1].scatter(tt, av_1, c = C, s = 90)
    axarr[0, 1].plot(times, w_1[:, 0], 'b', linestyle=':')
    axarr[0, 1].plot(times, w_1[:, 1], 'g', linestyle=':')
    axarr[0, 1].plot(times, w_1[:, 2], 'r', linestyle=':')
    axarr[0, 1].plot(partial_time, part_w1_p, 'b', linewidth='3', label=r'$\sigma$')
    axarr[0, 1].plot(partial_time, part_w1_n, 'g', linewidth='3', label=r'$\rho$')
    axarr[0, 1].plot(partial_time, part_w1_c, 'r', linewidth='3', label=r'$\beta$') 
    axarr[0, 1].set_title('AV1 Parameter Weights')
    axarr[0, 1].legend(loc='best',fontsize=10)
    
    #Plots AV2 weights in time. Future time is dashed, past time is filled in, and present time is big dots
    axarr[0, 2].set_xlim(0, fT)
    axarr[0, 2].set_ylim(-1.8, 1.8)
    axarr[0, 2].scatter(tt, av_2, c=C, s=90) 
    axarr[0, 2].plot(times, w_2[:, 0], 'b', linestyle=':')
    axarr[0, 2].plot(times, w_2[:,1], 'g', linestyle=':')
    axarr[0, 2].plot(times, w_2[:,2], 'r', linestyle=':')
    axarr[0, 2].plot(partial_time, part_w2_p, 'b', linewidth='3', label=r'$\sigma$')
    axarr[0, 2].plot(partial_time, part_w2_n, 'g', linewidth='3', label=r'$\rho$')
    axarr[0, 2].plot(partial_time, part_w2_c, 'r', linewidth='3', label=r'$\beta$') 
    axarr[0, 2].set_title('AV2 Parameter Weights')
    axarr[0, 2].legend(loc='best', fontsize=10)
    
    #Plots inactive weights in time. Future time is dashed, past time is filled in, and present time is big dots
    axarr[0, 3].set_xlim(0, fT)
    axarr[0, 3].set_ylim(-1.8, 1.8)
    axarr[0, 3].scatter(tt, av_3, c=C, s=90)
    axarr[0, 3].plot(times, w_3[:,0], 'b', linestyle=':')
    axarr[0, 3].plot(times, w_3[:,1], 'g', linestyle=':')
    axarr[0, 3].plot(times, w_3[:,2], 'r', linestyle=':')
    axarr[0, 3].plot(partial_time, part_w3_p, 'b', linewidth='3', label=r'$\sigma$')
    axarr[0, 3].plot(partial_time, part_w3_n, 'g', linewidth='3', label=r'$\rho$')
    axarr[0, 3].plot(partial_time, part_w3_c, 'r', linewidth='3', label=r'$\beta$')
    axarr[0, 3].set_title('Inactive Parameter Weights')
    axarr[0, 3].legend(loc='best', fontsize=10)

    #Plots 1D SSP at present time
    axarr[1, 0].set_xlim(-1.8, 1.8)
    axarr[1, 0].set_ylim(-.01, (fmax+1))
    axarr[1, 0].set_title('Sufficient Summary Plot')
    axarr[1, 0].scatter(x_ax, y_ax, c=col, cmap="viridis", vmin=fmin, vmax=fmax)
    
    #Plots 2D SSP at present time
    axarr[1, 1].set_xlim(-1.8, 1.8)
    axarr[1, 1].set_ylim(-1.8, 1.8)
    axarr[1, 1].set_title('Sufficient Summary Plot')    
    kk=axarr[1, 1].scatter(x_ax, y_ax_2, c=col, cmap="viridis", vmin=fmin, vmax=fmax)
    axarr[1, 1].legend(loc='best', fontsize=6)
    
    #Horsetail plot of possible solutions, uture time is dashed, past time is filled in, and present time is black line
    axarr[1, 2].set_xlim(0, fT)
    axarr[1, 2].plot(times, f[:, 1:100], linestyle=':')
    axarr[1, 2].plot(partial_time, f[0:i, 1:100], linewidth='2')
    axarr[1, 2].axvline(x=times[i], color='k', lw='1.5')
    axarr[1, 2].set_title('Solutions')
    
    #Adds colorbar legend and title
    fa.colorbar(kk, ax=axarr[1, 1])
    fa.suptitle('z, time = %.1f' % times[i], fontsize=20)
    #plt.show()
    fa.savefig(filename+'%.1f.png' % i)
    plt.close() 
    
def dynamo(times, yg0, sys, m, Yl, Yu, QOI, y0, reg_sys, q, c, h):
    """computes the eigenvalues and vectors and normalizes sign of eigenvectors
    Parameters
    ----------
    times : ndarray
        time array used in odeint
    yg0 : ndarray
        initial conditions for the ODE system
    sys : ndarray
        Gronwall system of ODEs specific to the dynamical system
    m: int
        input parameter dimension
    Yl: ndarray
        1-by-m array of lower bounds on parameter space
    Yu: ndarray
        1-by-m array of upper bounds on parameter space
    QOI: function
        function that evaluates the QOI for the given system
    y0: ndarray
        1-by-3 array of initial conditions for reg_sys
    reg_sys: ndarray
        1-by-3 ODE system (NOT GRONWALL--unnecessary expense)
    q: int
        index corresponding to state of dynamical system that is QOI.
    c : string
        options are 'gron' or 'fdm'. defines which gradient approximation technique to use.
    h : float
        stepsize for FDM    
        
    Returns
    -------
    eig: ndarray
        len(times)-by-m array of eigenvalues. 
    w_1: ndarray
        m-by-1 array, first eigenvector of C
    w_2: ndarray
        m-by-1 array, second eigenvector of C
    w_3: ndarray
        m-by-1 array, third eigenvector of C
    XX: ndarray
        M-by-m matrix of gauss-legendre nodes. Between [-1,1].
    f: ndarray
        len(times)-by-M array of solutions for QOI over time.
    fmin: int
        minimum of the QOI 
    fmax: int
        maximum of the QOI
    """
    eig = np.zeros([len(times), m])
    w_1 = np.zeros([len(times), m])
    w_2 = np.zeros([len(times), m])
    w_3 = np.zeros([len(times), m])
    fmin = 100000
    fmax = 0
    M = 729
    y, v_t, W_t, XX = log_transform(yg0, times, sys, m, Yl, Yu, h, y0, q, c)
    eig = v_t 
     
    for i in range(len(times)):
        w_1_temp = W_t[i][:, 0]
        w_2_temp = W_t[i][:, 1]
        w_3_temp = W_t[i][:, 2]
        w_1[i, :] = w_1_temp.reshape((1, 3)) 
        w_2[i, :] = w_2_temp.reshape((1, 3))  
        w_3[i, :] = w_3_temp.reshape((1, 3))  
        
    f = QOI(10 ** y, times, y0, reg_sys, q)
    
    #Normalizes sign of the eigenvectors w_1, w_2, and w_3
    for i in range(len(times) - 1):
        if (np.linalg.norm(w_1[i, :] - w_1[i + 1, :]) > np.linalg.norm(w_1[i, :] + w_1[i + 1, :])):
            w_1[i + 1, :] = -1. * w_1[i + 1, :]
    for i in range(len(times) - 1):
        if (np.linalg.norm(w_2[i, :] - w_2[i + 1, :]) > np.linalg.norm(w_2[i, :] + w_2[i + 1, :])):
            w_2[i + 1, :] = -1. * w_2[i + 1, :]      
    for i in range(len(times) - 1):
        if (np.linalg.norm(w_3[i, :] - w_3[i + 1, :]) > np.linalg.norm(w_3[i, :] + w_3[i + 1, :])):
            w_3[i + 1, :] = -1. * w_3[i + 1, :] 
            
    #Max and Min of f        
    for i in range(len(times)):
        if (fmin > min(f[i, :])):
            fmin = min(f[i, :])
        if (fmax < max(f[i, :])):
            fmax = max(f[i, :])

    return eig, w_1, XX, f, w_2, w_3, fmin, fmax, y