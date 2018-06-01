"""
2D Periodic Doi-Onsager Model using Bingham Closure
Timestepping is done using a first-order Forward-Euler/Backward-Euler split

dstein@flatironinstitute.org
rfarhadifar@flatironinstitute.org
"""

import numpy as np
import scipy as sp
import scipy.special
import scipy.interpolate
import os
import sys
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

################################################################################
### Parameters

nxpow = 6             # nx = 2**nxpow
nypow = 6             # ny = 2**nypow
xmin = -np.pi
xmax =  np.pi
ymin = -np.pi
ymax =  np.pi
tmax = 20              # end time for simulation
dt_modifier = 2        # dt = 2**6 * 0.1 / n / dt_modifier
d_R = 0.05             # rotational diffusion coefficient
d_T = 0.05             # translational diffusion coefficient
kesi = 1.0             # mean-field torque strength
alpha = -5.0           # extensile stretching coefficient
beta = 1.0             # shape factor
phi = 1.0              # concentration
background_force_magnitude = 0.0
initial_perturbation_size = 1e-3
stein = True
plot = True            # whether to generate plots or not
plot_how_often = 0.2   # how often to plot
use_tex = True         # set this to False for (much) faster plotting
timestepper = 'RK4_BE' # which timestepper to use (either 'FE_BE' or 'RK4_BE')

stein_path = '/Users/dstein/Dropbox (Simons Foundation)/Doi-Onsager Model Update/Output/'
jessica_path = '/Users/JessFlores/flatiron-2018/output'
path = stein_path if stein else jessica_path
os.chdir(path)

if use_tex:
    mpl.rc('text', usetex=True)

################################################################################
### Parameters that probably shouldn't be changed (or just depend on other parameters)

nx = 2**nxpow
ny = 2**nypow
dt0 = 2**6 * 0.1 / (nx*ny)**(1/2.) / dt_modifier

################################################################################
### Setup

print('\nSetting things up')

# grid on which physical equations live
xran = xmax - xmin
yran = ymax - ymin
xv, xh = np.linspace( xmin, xmax, nx, endpoint = False, retstep = True )
yv, yh = np.linspace( ymin, ymax, ny, endpoint = False, retstep = True )
x, y = np.meshgrid(xv, yv, indexing='ij')
if np.max(np.abs(np.array((xh-yh)))) > 1e-15:
    raise ValueError('Please define domain bounds and ns such that the grid spacing is isotropic')
h = xh

n_timesteps = int(np.ceil(tmax / dt0))
dt = tmax / n_timesteps

# fourier grid
kxv = np.fft.fftfreq ( nx, h/xran )
kyv = np.fft.fftfreq ( ny, h/yran )
kx, ky = np.meshgrid(kxv, kyv, indexing='ij')
# operators
ksq = kx**2 + ky**2
lap = -ksq
rlap = lap.copy()
rlap[0,0] = 1.0
zlapi = 1.0 / rlap
zlapi[0,0] = 0.0
# heat operator for diffusion
diffuser = 1.0 - d_T*lap*dt
idiffuser = 1.0 / diffuser
# pseudospectral filter
max_kx = np.abs(kxv).max()
max_ky = np.abs(kyv).max()
decayer = np.exp(-36*(np.abs(kx)/max_kx)**36)*np.exp(-36*(np.abs(ky)/max_ky)**36)
# eliminate nyquist frequency for single derivative operators
kxv[int(nx/2)] = 0.0
kyv[int(ny/2)] = 0.0
kx, ky = np.meshgrid(kxv, kyv, indexing='ij')
ikx = 1j * kx
iky = 1j * ky
iks = np.zeros((2,nx,ny), dtype=complex)
iks[0][:] = ikx
iks[1][:] = iky

################################################################################
### Periodic Stokes Solver

def stokes(fh):
    div_fh = np.einsum('i...,i...->...',iks,fh)
    ph = zlapi*div_fh
    uh = zlapi*(iks*ph - fh)
    return uh, ph

################################################################################
### Bingham Closure

# seems like about 40 points is sufficient for machine accuracy of all sums
thetas, dtheta = np.linspace(0,2*np.pi,40,endpoint=False,retstep=True)
ct = np.cos(thetas)
st = np.sin(thetas)
ctst = ct*st
st2 = st*st
c2t = np.cos(2*thetas)
ct2 = ct**2
ct2c2t = ct2*c2t
ct4 = ct**4
ct3st = ct**3*st
myone = np.ones_like(thetas)

NN = 10000
mu = np.linspace(0.5,1.0,NN,endpoint=True)
mus = mu[:-1]
big_one = np.ones_like(mus)
newton_tol = 1e-12
jacobian_eps = 1e-6

def bingham_integral1(l1,f):
    Z = 2*np.pi*sp.special.iv(0,l1)
    integrand = np.exp(l1[:,None]*c2t[None,:])*f[None,:]
    return dtheta*np.sum(integrand, axis=-1)/Z

def bingham_integral2(l1,f):
    Z = 2*np.pi*sp.special.iv(0,l1)
    integrand = np.exp(l1[:,:,None]*c2t[None,None,:])*f[None,None,:]
    return dtheta*np.sum(integrand, axis=-1)/Z

def full_bingham_integral(l1, B00, B01, B11, f):
    Z = 2*np.pi*sp.special.iv(0,l1)
    integrand = np.exp(B00[:,:,None]*ct2[None,None,:]+2*B01[:,:,None]*ctst[None,None,:]+B11[:,:,None]*st2[None,None,:])*f[None,None,:]
    return dtheta*np.sum(integrand, axis=-1)/Z

def get_jacobian(l1):
    I0 = sp.special.iv(0,l1)
    Im1 = sp.special.iv(-1,l1)
    Ip1 = sp.special.iv(1,l1)
    dI0 = (Im1+Ip1)/2.0
    Ja = -dI0/I0*bingham_integral1(l1,ct2)
    Jb = bingham_integral1(l1,ct2c2t)
    return Ja + Jb

def mu_to_lambda(mu):
    l1 = big_one*0.5
    err = bingham_integral1(l1,ct2) - mu
    err_max = np.abs(err).max()
    while err_max > newton_tol:
        jac = get_jacobian(l1)
        l1 -= err/jac
        err = bingham_integral1(l1,ct2) - mu
        err_max = np.abs(err).max()
        print('   Residual is: {:0.3e}'.format(err_max))
    return l1

print('Making reference map for Bingham Closure')
l1 = mu_to_lambda(mus)
# now integrate these to get S0000 and S0001 in the special coordinate system
S0000 = bingham_integral1(l1, ct4)
S0000 = np.concatenate((S0000, (1.0,)))
# get an interpolater for S0000
interper = sp.interpolate.interp1d(mu, S0000, kind='cubic')

def rotate(l1, R):
    p = np.zeros_like(R)
    p[:,:,0,0] = l1
    p[:,:,1,1] = -l1
    return np.einsum('...ik,...kl,...jl->...ij',R,p,R)

def bingham_closure(D, E):
    """
    Direct Estimation of Bingham Closure (through rotation)
    """
    Dd = np.transpose(D, (2,3,0,1))
    EV = np.linalg.eigh(Dd)
    Eval = EV[0][:,:,::-1]
    Evec = EV[1][:,:,:,::-1]
    mu = Eval[:,:,0]
    mu[mu<0.5] = 0.5
    mu[mu>1.0] = 1.0
    tS0000 = interper(mu)
    tS0011 = Eval[:,:,0] - tS0000
    tS1111 = Eval[:,:,1] - tS0011
    # transform to real coordinates
    l00, l01, l10, l11 = Evec[:,:,0,0], Evec[:,:,0,1], Evec[:,:,1,0], Evec[:,:,1,1]
    S0000 = l00**4*tS0000 + 6*l01**2*l00**2*tS0011 + l01**4*tS1111
    S0001 = l00**3*l10*tS0000 + (3*l00*l01**2*l10+3*l00**2*l01*l11)*tS0011 + l01**3*l11*tS1111
    # get the others
    S0011 = D[0,0] - S0000
    S1111 = D[1,1] - S0011
    S0111 = D[0,1] - S0001
    # perform contractions
    SD = np.zeros_like(D)
    SD[0,0,:,:] = S0000*D[0,0] + 2*S0001*D[0,1] + S0011*D[1,1]
    SD[0,1,:,:] = S0001*D[0,0] + 2*S0011*D[0,1] + S0111*D[1,1]
    SD[1,1,:,:] = S0011*D[0,0] + 2*S0111*D[0,1] + S1111*D[1,1]
    SD[1,0,:,:] = SD[0,1]
    SE = np.zeros_like(E)
    SE[0,0,:,:] = S0000*E[0,0] + 2*S0001*E[0,1] + S0011*E[1,1]
    SE[0,1,:,:] = S0001*E[0,0] + 2*S0011*E[0,1] + S0111*E[1,1]
    SE[1,1,:,:] = S0011*E[0,0] + 2*S0111*E[0,1] + S1111*E[1,1]
    SE[1,0,:,:] = SE[0,1]
    return SD, SE

################################################################################
### Update D

def stress_update(Dh, Uh):
    """
    Computes f(D) for the right hand side of the equation D_t = f(D) + g(D)
    f(D) = 4*kesi*(D*D-S[D]:D) - 2d d_R(D-(phi/d)I) - u dot grad D + grad u^T D + D grad u - 2E:S[D]
    Here:
        (grad u)_{ij} = partial_i u_j
        E = (grad u + grad u^T)/2
    """

    # get D in real space
    D = np.fft.ifft2(Dh).real
    # get decayed spectra for U
    dUh = Uh*decayer
    # get derivatives of U
    gUh = np.einsum('i...,j...->ij...',iks,dUh)
    # put these things into real space
    dU = np.fft.ifft2(dUh).real
    gU = np.fft.ifft2(gUh).real
    # get decayed spectra for D
    dDh = Dh*decayer
    # compute gradient of D
    gDh = np.einsum('i...,jk...->ijk...',iks,dDh)
    # put these things into real space
    dD = np.fft.ifft2(dDh).real
    gD = np.fft.ifft2(gDh).real
    # compute D gU
    DgU = np.einsum('ij...,jk...->ik...',dD,gU)
    # compute gU^T D
    gUtD = np.einsum('ji...,jk...->ik...',gU,dD)
    # compute u dot gD
    U_dot_gD = np.einsum('i...,ijk...->jk...',dU,gD)
    # compute DD
    DD = np.einsum('ij...,jk...->ik...',dD,dD)
    # compute E
    E = (gU + np.transpose(gU,axes=(1,0,2,3)))/2.0
    # compute S[D]:D, E:S[D]
    SD, SE = bingham_closure(D, E)
    # compute the 'update'
    d = 2.0
    update = 4*kesi*(DD-SD) - 2*d*d_R*(D-(phi/d)*big_eye) - U_dot_gD + gUtD + DgU - 2*SE
    # send the update to Fourier Space
    updateh = np.fft.fft2(update)
    return updateh

################################################################################
### update Sigma

def Sigma_update(Dh, Uh):
    """
    Computes Sigma = alpha D + beta S[D]:E - 2*kesi*beta*(DD - S[D]:D)
    Here:
        (grad u)_{ij} = partial_i u_j
        E = (grad u + grad u^T)/2
    """
    # get decayed spectra for U
    dUh = Uh*decayer
    # get derivatives of U
    gUh = np.einsum('i...,j...->ij...',iks,dUh)
    # put these things into real space
    gU = np.fft.ifft2(gUh).real
    # get decayed spectra for D
    dDh = Dh*decayer
    # put these things into real space
    dD = np.fft.ifft2(dDh).real
    # compute DD
    DD = np.einsum('ij...,jk...->ik...',dD,dD)
    # compute E
    E = (gU + np.transpose(gU,axes=(1,0,2,3)))/2.0
    # compute S[D]:D, S[D]:E
    SD, SE = bingham_closure(D, E)
    # compute the stress (minus the alpha*D part!)
    sigma = beta*SE - 2*kesi*beta*(DD - SD)
    # send to fourier space
    sigmah = np.fft.fft2(sigma)
    # add in alpha*D part
    sigmah += alpha*Dh
    return sigmah

################################################################################
### Save the components of D as image

def smallify(X, by):
    return X[::by,::by]
rad_to_deg = lambda x: x*360/(2*np.pi)

def Graphics(D, NUM):
    D00 = D[0,0]
    D01 = D[0,1]
    D11 = D[1,1]

    D_full = np.zeros((nx,ny,2,2), dtype=float)
    D_full[:,:,0,0] = D00
    D_full[:,:,0,1] = D01
    D_full[:,:,1,0] = D01
    D_full[:,:,1,1] = D11
    D_eig = np.linalg.eigh(D_full)
    p1 = D_eig[1][:,:,:,-1]
    p2 = D_eig[1][:,:,:,-2]
    p1m = D_eig[0][:,:,-1]
    p2m = D_eig[0][:,:,-2]
    num_ellipses = 32
    smallify_factor = int(nx/num_ellipses)
    sx = smallify(x, smallify_factor) + h/2.0
    sy = smallify(y, smallify_factor) + h/2.0
    snx = int(nx / smallify_factor)
    sny = int(ny / smallify_factor)
    spx = smallify(p1[:,:,0], smallify_factor)
    spy = smallify(p1[:,:,1], smallify_factor)
    sp1m = smallify(p1m[:,:], smallify_factor)
    sp2m = smallify(p2m[:,:], smallify_factor)
    ang = np.arctan2(spy[:,:],spx[:,:])
    fig, ax = plt.subplots(1,1)
    sizeit = h*smallify_factor/1.5
    XY = np.column_stack([sx.ravel(), sy.ravel()])
    ells = mpl.collections.EllipseCollection(offsets=XY, widths=sp1m*sizeit, heights=sp2m*sizeit, angles=rad_to_deg(ang), units='xy', transOffset=ax.transData)
    ells.set_array((sx + sy).ravel())
    ells.set_facecolor('black')
    ells.set_facecolors('black')
    ells.set_edgecolors('black')
    ells.set_edgecolor('black')
    ells.set_color('black')
    ax.add_collection(ells)
    ax.set(xlim=(-np.pi,np.pi), ylim=(-np.pi,np.pi))
    ax.set(xticks=(-np.pi, 0, np.pi), xticklabels=(r'$-\pi$', r'$0$', r'$\pi$'))
    ax.set(yticks=(-np.pi, 0, np.pi), yticklabels=(r'$-\pi$', r'$0$', r'$\pi$'))
    ax.set_aspect('equal')
    plt.tight_layout()
    filename = 'Fast_Bingham_' + str(NUM).zfill(4)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

################################################################################
### Initialize Variables

print('   Initializing Variables')

# driving forces
f_driving = np.zeros((2,nx,ny))
f_driving[0][:] = background_force_magnitude*np.sin(x)*np.cos(y)
f_driving[1][:] = -background_force_magnitude*np.cos(x)*np.sin(y)
# in fourier space
fh_driving = np.fft.fft2(f_driving)

# solve for an initial velocity field
Uh, ph = stokes(fh_driving)

# get the identity tensor
big_eye = np.zeros((2,2,nx,ny))
big_eye[0,0] += 1.0
big_eye[1,1] += 1.0

complex_rand = lambda sh: np.random.rand(*sh) + 1j*np.random.rand(*sh)

# initialize D (to the identity tensor)
SD = big_eye.copy()/np.sqrt(2.0)
SDh = np.fft.fft2(SD)
SDh[0,0] += complex_rand([nx, ny])*initial_perturbation_size*decayer
SD = np.fft.ifft2(SDh).real
D = np.einsum('ij...,jk...->ik...',SD,SD)

# take the FFT of D
Dh = np.fft.fft2(D)

# set time to 0
t = 0

# get CPU time at start of computation
start_time = time.time()

################################################################################
### Timestepping Loop

print('\nRunning simulation')

NUM = 1
next_plot_time = 0.0

for i in range(n_timesteps):
    print('   Time: {:0.4f}'.format(t), 'of', tmax, '\r', end='')
    sys.stdout.flush()

    if timestepper == 'FE_BE':
        # explicit portion of update
        updateh = stress_update(Dh, Uh)
        Dh += dt*updateh
        # apply viscosity using Backward-Euler
        Dh *= idiffuser
    if timestepper == 'RK4_BE':
        # explicit portion of update
        temp_Dh = Dh.copy()
        k1 = dt*stress_update(temp_Dh, Uh)
        temp_Dh = Dh + 0.5*k1
        k2 = dt*stress_update(temp_Dh, Uh)
        temp_Dh = Dh + 0.5*k2
        k3 = dt*stress_update(temp_Dh, Uh)
        temp_Dh = Dh + k3
        k4 = dt*stress_update(temp_Dh, Uh)
        const = 1.0/6.0
        Dh += const*(k1 + 2*k2 + 2*k3 + k4)
        
        # apply viscosity using Backward-Euler
        Dh *= idiffuser

    # now let's modify D11 so that its exactly 1 - D00
    D00 = np.fft.ifft2(Dh[0,0]).real
    D11 = 1.0 - D00
    Dh[1,1] = np.fft.fft2(D11)
    # symmetrize
    Dh = (Dh + Dh.transpose(1,0,2,3))/2.0

    # compute Sigma
    Sigmah = Sigma_update(Dh, Uh)
    divSigmah = np.einsum('i...,ij...->j...',iks,Sigmah)

    # update the velocity field
    fh = divSigmah
    if t < 1:
        fh += fh_driving
    Uh, ph = stokes(fh)

    # graphic outputs
    if plot:
        if t >= next_plot_time:
            D = np.fft.ifft2(Dh).real
            Graphics(D, NUM)
            NUM += 1
            next_plot_time += plot_how_often

    # update the time variables
    t += dt
print('   Time: {:0.4f}'.format(t), 'of', tmax, '\r', end='')

D = np.fft.fft2(Dh)

end_time = time.time()
print ('\n\nSimulation took {:0.3f}'.format(time.time() - start_time), 'seconds')

if plot:
    os.system('ffmpeg -y -framerate 10 -i Fast_Bingham_%04d.png -vf scale=-2:720 -b:v 9600k -pix_fmt yuv420p bingham.mp4')

