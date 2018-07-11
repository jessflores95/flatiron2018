"""
3D Periodic Doi-Onsager Model using Bingham Closure
In this version, the mapping from Eigenvalues of D to S_0000, S_1111, and S_2222
In the rotated coordinate system is done via the Chaubal/Leal functions
Timestepping is done using a first-order Forward-Euler/Backward-Euler split

dstein@flatironinstitute.org
rfarhadifar@flatironinstitute.org
"""

import numpy as np
import scipy as scp
import scipy.optimize
import os
import sys
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema
plt.ion()

################################################################################
### Parameters

nxpow = 6           # nx = 2**nxpow
nypow = 6           # ny = 2**nypow
nzpow = 6           # nz = 2**nzpow
xmin = -np.pi
xmax =  np.pi
ymin = -np.pi
ymax =  np.pi
zmin = -np.pi
zmax =  np.pi
tmax = 50           # end time for simulation
dt_modifier = 5     # dt = 2**6 * 0.1 / n / dt_modifier
d_R = 0.1           # rotational diffusion coefficient
d_T = 0.1           # translational diffusion coefficient
kesi = 0.5          # mean-field torque strength
alpha = -5.0        # extensile stretching coefficient
beta = 1.0          # shape factor
phi = 1.0           # concentration
closure = 'bingham'     # choose which closure to use ('doi', 'K1', 'bingham')
S2_cutoff = 1e-12   # cutoff for isotropic case (only for K1 closure)
background_force_magnitude = 0.0
initial_perturbation_size = 1.0e-3

################################################################################
### Parameters that probably shouldn't be changed (or just depend on other parameters)

nx = 2**nxpow
ny = 2**nypow
nz = 2**nzpow
dt0 = 2**6*0.1 / (nx*ny*nz)**(1/3.) / dt_modifier

initial_perturbation_time = 5*dt0

################################################################################
### Setup

print('\nSetting things up')

# grid on which physical equations live
xran = xmax - xmin
yran = ymax - ymin
zran = zmax - zmin
xv, xh = np.linspace( xmin, xmax, nx, endpoint = False, retstep = True )
yv, yh = np.linspace( ymin, ymax, ny, endpoint = False, retstep = True )
zv, zh = np.linspace( zmin, zmax, nz, endpoint = False, retstep = True )
x, y, z = np.meshgrid(xv, yv, zv, indexing='ij')
#if np.max(np.abs(np.array((xh-yh,yh-zh,xh-zh)))) > 1e-15:
   # raise ValueError('Please define domain bounds and ns such that the grid spacing is isotropic')
h = xh

n_timesteps = int(np.ceil(tmax / dt0))
dt = tmax / n_timesteps

# fourier grid
kxv = np.fft.fftfreq( nx, h/xran )
kyv = np.fft.fftfreq( ny, h/yran )
kzv = np.fft.fftfreq( nz, h/zran )
kx, ky, kz = np.meshgrid(kxv, kyv, kzv, indexing='ij')
# operators
ksq = kx**2 + ky**2 + kz**2
lap = -ksq
rlap = lap.copy()
rlap[0,0,0] = 1.0
zlapi = 1.0 / rlap
zlapi[0,0,0] = 0.0
# heat operator for diffusion
diffuser = 1.0 - d_T*lap*dt
idiffuser = 1.0 / diffuser
# pseudospectral filter
max_kx = np.abs(kxv).max()
max_ky = np.abs(kyv).max()
max_kz = np.abs(kzv).max()
decayer = np.exp(-36*(np.abs(kx)/max_kx)**36)*np.exp(-36*(np.abs(ky)/max_ky)**36)*np.exp(-36*(np.abs(kz)/max_kz)**36)
# eliminate nyquist frequency for single derivative operators
kxv[int(nx/2)] = 0.0
kyv[int(ny/2)] = 0.0
kzv[int(nz/2)] = 0.0
kx, ky, kz = np.meshgrid(kxv, kyv, kzv, indexing='ij')
ikx = 1j * kx
iky = 1j * ky
ikz = 1j * kz
iks = np.zeros((3,nx,ny,nz), dtype=complex)
iks[0][:] = ikx
iks[1][:] = iky
iks[2][:] = ikz

################################################################################
### Define convenience function for 3D FFT

fft3  = lambda x: np.fft.fftn (x, axes=(-3,-2,-1))
ifft3 = lambda x: np.fft.ifftn(x, axes=(-3,-2,-1))

################################################################################
### Periodic Stokes Solver

def stokes(fh):
    div_fh = np.einsum('i...,i...->...',iks,fh)
    ph = zlapi*div_fh
    uh = zlapi*(iks*ph - fh)
    return uh, ph

################################################################################
### Bingham Closure Function

def bingham_closure(D, E):
    """
    Estimation of Bingham Closure (through rotation) and Chaubal/Leal function
    """
    # get eigendecomposition
    Dd = np.transpose(D, (2,3,4,0,1))
    EV = np.linalg.eigh(Dd)
    Eval = EV[0][:,:,:,::-1]
    Evec = EV[1][:,:,:,:,::-1]
    mu0 = Eval[:,:,:,0]
    mu1 = Eval[:,:,:,1]
    mu2 = Eval[:,:,:,2]
    # get tS0000, tS1111, and tS2222 via Chaubal/Leal formula
    tS0000 =  0.412251 + 0.896044*mu0 - 2.026540*mu0**2 + 1.710790*mu0**3 - 3.44613*mu1 + 6.13032*mu1**2 - 3.75580*mu1**3 + 3.243780*mu0*mu1 + 0.381274*mu0**2*mu1 - 3.223310*mu0*mu1**2
    tS1111 =  0.150497 - 0.293780*mu0 + 0.092542*mu0**2 + 0.044353*mu0**3 - 0.28173*mu1 + 1.92763*mu1**2 - 1.46408*mu1**3 + 0.473044*mu0*mu1 + 0.037776*mu0**2*mu1 + 0.406202*mu0*mu1**2
    tS2222 = -0.610534 + 5.010940*mu0 - 8.165180*mu0**2 + 3.764260*mu0**3 + 4.39396*mu1 - 6.77247*mu1**2 + 2.96854*mu1**3 - 15.10930*mu0*mu1 + 10.68660*mu0**2*mu1 + 9.920540*mu0*mu1**2
    # get the other nonzero terms of tS
    tS1122 = (mu1 + mu2 - mu0 + tS0000 - tS1111 - tS2222)/2.0
    tS0022 = mu2 - tS2222 - tS1122
    tS0011 = mu0 - tS0000 - tS0022
    # compute the required terms of S by rotation
    l00, l01, l02 = Evec[:,:,:,0,0], Evec[:,:,:,0,1], Evec[:,:,:,0,2]
    l10, l11, l12 = Evec[:,:,:,1,0], Evec[:,:,:,1,1], Evec[:,:,:,1,2]
    l20, l21, l22 = Evec[:,:,:,2,0], Evec[:,:,:,2,1], Evec[:,:,:,2,2]
    S0000 = l00**4*tS0000+6*l00**2*l01**2*tS0011+6*l00**2*l02**2*tS0022+l01**4*tS1111+6*l01**2*l02**2*tS1122+l02**4*tS2222
    S1111 = l10**4*tS0000+6*l10**2*l11**2*tS0011+6*l10**2*l12**2*tS0022+l11**4*tS1111+6*l11**2*l12**2*tS1122+l12**4*tS2222
    S2222 = l20**4*tS0000+6*l20**2*l21**2*tS0011+6*l20**2*l22**2*tS0022+l21**4*tS1111+6*l21**2*l22**2*tS1122+l22**4*tS2222
    S0001 = l00**3*l10*tS0000+(3*l00*l01**2*l10 + 3*l00**2*l01*l11)*tS0011+(3*l00*l02**2*l10 + 3*l00**2*l02*l12)*tS0022+l01**3*l11*tS1111+(3*l01*l02**2*l11 + 3*l01**2*l02*l12)*tS1122+l02**3*l12*tS2222
    S0002 = l00**3*l20*tS0000+(3*l00*l01**2*l20 + 3*l00**2*l01*l21)*tS0011+(3*l00*l02**2*l20 + 3*l00**2*l02*l22)*tS0022+l01**3*l21*tS1111+(3*l01*l02**2*l21 + 3*l01**2*l02*l22)*tS1122+l02**3*l22*tS2222
    S0012 = l00**2*l10*l20*tS0000+(l01**2*l10*l20 + 2*l00*l01*l11*l20 + 2*l00*l01*l10*l21 + l00**2*l11*l21)*tS0011+(l02**2*l10*l20 + 2*l00*l02*l12*l20 + 2*l00*l02*l10*l22 + l00**2*l12*l22)*tS0022+l01**2*l11*l21*tS1111+(l02**2*l11*l21 + 2*l01*l02*l12*l21 + 2*l01*l02*l11*l22 + l01**2*l12*l22)*tS1122+l02**2*l12*l22*tS2222
    S0111 = l00*l10**3*tS0000+(3*l01*l10**2*l11 + 3*l00*l10*l11**2)*tS0011+(3*l02*l10**2*l12 + 3*l00*l10*l12**2)*tS0022+l01*l11**3*tS1111+(3*l02*l11**2*l12 + 3*l01*l11*l12**2)*tS1122+l02*l12**3*tS2222
    S0112 = l00*l10**2*l20*tS0000+(2*l01*l10*l11*l20 + l01*l10**2*l21 + l00*l11**2*l20 + 2*l00*l10*l11*l21)*tS0011+(2*l02*l10*l12*l20 + l02*l10**2*l22 + l00*l12**2*l20 + 2*l00*l10*l12*l22)*tS0022+l01*l11**2*l21*tS1111+(2*l02*l11*l12*l21 + l02*l11**2*l22 + l01*l12**2*l21 + 2*l01*l11*l12*l22)*tS1122+l02*l12**2*l22*tS2222
    S1112 = l10**3*l20*tS0000+(3*l10*l11**2*l20 + 3*l10**2*l11*l21)*tS0011+(3*l10*l12**2*l20 + 3*l10**2*l12*l22)*tS0022+l11**3*l21*tS1111+(3*l11*l12**2*l21 + 3*l11**2*l12*l22)*tS1122+l12**3*l22*tS2222
    # and the remaining terms by identity
    S1122 = (D[1,1] + D[2,2] - D[0,0] + S0000 - S1111 - S2222)/2.0
    S0022 =  D[2,2] - S2222 - S1122
    S0011 =  D[0,0] - S0000 - S0022
    S0122 =  D[0,1] - S0001 - S0111
    S0222 =  D[0,2] - S0002 - S0112
    S1222 =  D[1,2] - S0012 - S1112
    # now perform the contractions
    SD = np.zeros_like(D)
    SD[0,0][:] = S0000*D[0,0] + S0011*D[1,1] + S0022*D[2,2] + 2*S0001*D[0,1] + 2*S0002*D[0,2] + 2*S0012*D[1,2]
    SD[0,1][:] = S0001*D[0,0] + S0111*D[1,1] + S0122*D[2,2] + 2*S0011*D[0,1] + 2*S0012*D[0,2] + 2*S0112*D[1,2]
    SD[0,2][:] = S0002*D[0,0] + S0112*D[1,1] + S0222*D[2,2] + 2*S0012*D[0,1] + 2*S0022*D[0,2] + 2*S0122*D[1,2]
    SD[1,1][:] = S0011*D[0,0] + S1111*D[1,1] + S1122*D[2,2] + 2*S0111*D[0,1] + 2*S0112*D[0,2] + 2*S1112*D[1,2]
    SD[1,2][:] = S0012*D[0,0] + S1112*D[1,1] + S1222*D[2,2] + 2*S0112*D[0,1] + 2*S0122*D[0,2] + 2*S1122*D[1,2]
    SD[2,2][:] = S0022*D[0,0] + S1122*D[1,1] + S2222*D[2,2] + 2*S0122*D[0,1] + 2*S0222*D[0,2] + 2*S1222*D[1,2]
    SD[1,0][:] = SD[0,1]
    SD[2,0][:] = SD[0,2]
    SD[2,1][:] = SD[1,2]
    SE = np.zeros_like(E)
    SE[0,0][:] = S0000*E[0,0] + S0011*E[1,1] + S0022*E[2,2] + 2*S0001*E[0,1] + 2*S0002*E[0,2] + 2*S0012*E[1,2]
    SE[0,1][:] = S0001*E[0,0] + S0111*E[1,1] + S0122*E[2,2] + 2*S0011*E[0,1] + 2*S0012*E[0,2] + 2*S0112*E[1,2]
    SE[0,2][:] = S0002*E[0,0] + S0112*E[1,1] + S0222*E[2,2] + 2*S0012*E[0,1] + 2*S0022*E[0,2] + 2*S0122*E[1,2]
    SE[1,1][:] = S0011*E[0,0] + S1111*E[1,1] + S1122*E[2,2] + 2*S0111*E[0,1] + 2*S0112*E[0,2] + 2*S1112*E[1,2]
    SE[1,2][:] = S0012*E[0,0] + S1112*E[1,1] + S1222*E[2,2] + 2*S0112*E[0,1] + 2*S0122*E[0,2] + 2*S1122*E[1,2]
    SE[2,2][:] = S0022*E[0,0] + S1122*E[1,1] + S2222*E[2,2] + 2*S0122*E[0,1] + 2*S0222*E[0,2] + 2*S1222*E[1,2]
    SE[1,0][:] = SE[0,1]
    SE[2,0][:] = SE[0,2]
    SE[2,1][:] = SE[1,2]
    return SD, SE

################################################################################
### D Updater

def stress_update(Dh, Uh):
    """
    Computes f(D) for the right hand side of the equation D_t = f(D) + g(D)
    f(D) = 4*kesi*(D*D-S[D]:D) - 2d d_R(D-(phi/d)I) - u dot grad D + grad u^T D + D grad u - 2E:S[D]
    Here:
        (grad u)_{ij} = partial_i u_j
        E = (grad u + grad u^T)/2
    """
    st = time.time()
    # get D in real space
    D = ifft3(Dh).real
    # get decayed spectra for U
    dUh = Uh*decayer
    # get derivatives of U
    gUh = np.einsum('i...,j...->ij...',iks,dUh)
    # put these things into real space
    dU = ifft3(dUh).real
    gU = ifft3(gUh).real
    # get decayed spectra for D
    dDh = Dh*decayer
    # compute gradient of D
    gDh = np.einsum('i...,jk...->ijk...',iks,dDh)
    # put these things into real space
    dD = ifft3(dDh).real
    gD = ifft3(gDh).real
    # compute D gU
    DgU = np.einsum('ij...,jk...->ik...',dD,gU)
    # compute gU^T D
    gUtD = np.einsum('ji...,jk...->ik...',gU,dD)
    # compute u dot gD
    U_dot_gD = np.einsum('i...,ijk...->jk...',dU,gD)
    # compute DD
    DD = np.einsum('ij...,jk...->ik...',dD,dD)
    # compute E
    E = (gU + np.transpose(gU,axes=(1,0,2,3,4)))/2.0
    # compute S[D]:D, E:S[D]
    d = 3.0
    et = time.time()
    pre_time = et - st
    st = time.time()
    if closure == 'doi':
        D2 = np.einsum('ij...,kl...->ijkl...',dD,dD)
        SDD = np.einsum('ijkl...,ij...->kl...',D2,DD)
        ESD = np.einsum('ij...,ijkl...->kl...',E,D2)
        update = 4*kesi*(DD-SDD) - 2*d*d_R*(D-(phi/d)*big_eye) - U_dot_gD + gUtD + DgU - 2*ESD
    elif closure == 'bingham':
        SD, SE = bingham_closure(dD, E)
        update = 4*kesi*(DD-SD) - 2*d*d_R*(D-(phi/d)*big_eye) - U_dot_gD + gUtD + DgU - 2*SE
    elif closure == 'K1':
        S22 = (3.0*np.einsum('ij...,ij...->...',dD,dD)-1.0)/2.0
        S22[S22 < 0] = 0.0
        S2 = np.sqrt(S22)
        ai = np.abs(S2) > S2_cutoff
        ii = np.logical_not(ai)
        # get ESD for isotropic values
        ESD = np.zeros((3,3,nx,ny,nz))
        ESD[:,:,ii] = 0.0/15*big_eye[:,:,ii]
        # get ESD for anisotropic values
        nu = 3.0/5
        zeta = (1.0-(1.0-S2[ai])**nu)/S2[ai]
        weight1 = zeta/3.0
        weight2 = -zeta/9.0*(1.0-(2/7.0)*S2[ai]) + 1/7.0
        weight3 = 2.0*zeta*(1/27.0-4/189.0*S2[ai]-1/35.0*S2[ai]**2) + 2.0/35
        M1 = np.einsum('ij...,ij...,kl...->kl...',E,dD,dD) + 2.0*np.einsum('ij...,jk...,kl...->il...',dD,E,dD)
        M2 = np.einsum('ij...,ij...,kl...->kl...',E,dD,big_eye) + 4.0*np.einsum('ij...,jk...->ik...',dD,E)
        M3 = E
        ESD[:,:,ai] = weight1*M1[:,:,ai] + weight2*M2[:,:,ai] + weight3*M3[:,:,ai]
        SDD = (7.0+5.0*S2-12.0*S2**2)/35.0*(dD-big_eye/3.0)
        # compute the 'update'
        update = 4*kesi*SDD - 2*d*d_R*(dD-(phi/d)*big_eye) - U_dot_gD + gUtD + DgU - 2*ESD
    else:
        raise ValueError('Choose a valid closure scheme!')
    et = time.time()
    closure_time = et - st
    # send the update to Fourier Space
    st = time.time()
    updateh = fft3(update)
    et = time.time()
    post_time = et - st
    #print('Everything else: {:0.3e}'.format(pre_time + post_time))
    #print('Closure time     {:0.3e}:'.format(closure_time))
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
    gU = ifft3(gUh).real
    # get decayed spectra for D
    dDh = Dh*decayer
    # put these things into real space
    dD = ifft3(dDh).real
    # compute DD
    DD = np.einsum('ij...,jk...->ik...',dD,dD)
    # compute E
    E = (gU + np.transpose(gU,axes=(1,0,2,3,4)))/2.0
    # compute S[D]:D, S[D]:E
    if closure == 'doi':
        D2 = np.einsum('ij...,kl...->ijkl...',dD,dD)
        SDD = np.einsum('ijkl...,ij...->kl...',D2,DD)
        SDE = np.einsum('ijkl...,ij...->kl...',D2,E)
        # compute the stress (minus the alpha*D part!)
        sigma = beta*SDE - 2*kesi*beta*(DD - SDD)
    elif closure == 'bingham':
        SD, SE = bingham_closure(dD, E)
        # compute the stress (minus the alpha*D part!)
        sigma = beta*SE - 2*kesi*beta*(DD - SD)
    elif closure == 'K1':
        S22 = (3.0*np.einsum('ij...,ij...->...',dD,dD)-1.0)/2.0
        S22[S22 < 0] = 0.0
        S2 = np.sqrt(S22)
        ai = np.abs(S2) > S2_cutoff
        ii = np.logical_not(ai)
        # get ESD for isotropic values
        ESD = np.zeros((3,3,nx,ny,nz))
        ESD[:,:,ii] = 0.0/15*big_eye[:,:,ii]
        # get ESD for anisotropic values
        nu = 3.0/5
        zeta = (1.0-(1.0-S2[ai])**nu)/S2[ai]
        weight1 = zeta/3.0
        weight2 = -zeta/9.0*(1.0-(2/7.0)*S2[ai]) + 1/7.0
        weight3 = 2.0*zeta*(1/27.0-4/189.0*S2[ai]-1/35.0*S2[ai]**2) + 2.0/35
        M1 = np.einsum('ij...,ij...,kl...->kl...',E,dD,dD) + 2.0*np.einsum('ij...,jk...,kl...->il...',dD,E,dD)
        M2 = np.einsum('ij...,ij...,kl...->kl...',E,dD,big_eye) + 4.0*np.einsum('ij...,jk...->ik...',dD,E)
        M3 = E
        ESD[:,:,ai] = weight1*M1[:,:,ai] + weight2*M2[:,:,ai] + weight3*M3[:,:,ai]
        SDE = ESD
        SDD = (7.0+5.0*S2-12.0*S2**2)/35.0*(dD-big_eye/3.0)
        # compute the stress (minus the alpha*D part!)
        sigma = beta*SDE - 2*kesi*beta*SDD
    else:
        raise ValueError('Choose a valid closure scheme!')
    # send to fourier space
    sigmah = fft3(sigma)
    # add in alpha*D part
    sigmah += alpha*Dh
    return sigmah

################################################################################
### Initialize Variables


print('   Initializing Variables')

# driving forces
f_driving = np.zeros((3,nx,ny,nz), float)
f_driving[0][:] = background_force_magnitude*np.sin(x)*np.cos(y)
f_driving[1][:] = -background_force_magnitude*np.cos(x)*np.sin(y)
f_driving[2][:] = np.zeros((nx,ny,nz),dtype=float)
# in fourier space
fh_driving = fft3(f_driving)

# random f
##fh_random = np.zeros((3,nx,ny,nz), dtype=complex)
##my_rand = lambda sh: (np.random.rand(*sh).reshape(sh)-0.5)*2
##complex_rand = lambda sh: my_rand(sh) + 1j*my_rand(sh)
##def full_rand(nx, ny, nz, initial_perturbation_size):
##    #rand = complex_rand([nx,ny,nz])*initial_perturbation_size*nx*ny*nz/(2*np.pi)
##    rand = np.exp(1j*kz*z)*intial_perturbation_size*nz/(2*np.pi)
##    rand[0][:] = 0.0
##    rand[1][:] = 0.0
##    #rand[5:-5,:,:] = 0.0
##    #rand[:,5:-5,:] = 0.0
##    #rand[:,:,5:-5] = 0.0
##    return rand

# solve for an initial velocity field
Uh, ph = stokes(fh_driving)

# get the identity tensor
big_eye = np.zeros((3,3,nx,ny,nz))
big_eye[0,0] += 1.0
big_eye[1,1] += 1.0
big_eye[2,2] += 1.0

# initialize D (to the scaled identity tensor)
D = big_eye.copy()/3.0
D_pure = D.copy()
## add perturbation to D, make sure to respect trace constraint
D_pert = np.zeros((3,nx,ny,nz),dtype=float)
D_pert = (np.exp(1j*kz*z)+np.exp(-1j*kz*z))*intial_perturbation_size*nz/(2*np.pi)
D_pert[:,:,:,:,0] = 0.0 #only z affected
D += D_pert
D[1,1,:,:,:] = 1.0 - D[0,0] #trace constraint

# take the FFT of D
Dh = fft3(D)

def simulation(Dh,D):
    # set time to 0
    t = 0

    # get CPU time at start of computation
    start_time = time.time()

    ################################################################################
    ### Timestepping Loop

    print('\nRunning simulation')

    Dh00_total = np.zeros(n_timesteps)
    Dh01_total = np.zeros(n_timesteps)
    Dh02_total = np.zeros(n_timesteps)
    Dh10_total = np.zeros(n_timesteps)
    Dh11_total = np.zeros(n_timesteps)
    Dh12_total = np.zeros(n_timesteps)
    Dh20_total = np.zeros(n_timesteps)
    Dh21_total = np.zeros(n_timesteps)
    Dh22_total = np.zeros(n_timesteps)

    Dh00_exp = np.zeros(n_timesteps)
    Dh01_exp = np.zeros(n_timesteps)
    Dh02_exp = np.zeros(n_timesteps)
    Dh10_exp = np.zeros(n_timesteps)
    Dh11_exp = np.zeros(n_timesteps)
    Dh12_exp = np.zeros(n_timesteps)
    Dh20_exp = np.zeros(n_timesteps)
    Dh21_exp = np.zeros(n_timesteps)
    Dh22_exp = np.zeros(n_timesteps)


    gamma = (1/5)*(alpha-2*kesi*beta/5)/(1+beta/15)
    omega_k = -(ksq*d_T-4*kesi/5 +6*d_R)

    Dh_off = np.zeros(Dh.shape,dtype=complex)
    Dh_diag = np.zeros(Dh.shape,dtype=complex)

    NUM = 1
    ns = np.arange(0,n_timesteps,1)
    t_total = ns*dt
    for i_ind, i in enumerate(ns):
        print('   Time: {:0.4f}'.format(t), 'of', tmax, '\r', end='')
        sys.stdout.flush()
        D_tilde = D-D_pure
        Dh_tilde = fft3(D_tilde)

        Dh00_total[i_ind] = np.sqrt(((np.abs(Dh_tilde[0,0]))**2).sum()*xh*yh*zh)
        Dh01_total[i_ind] = np.sqrt(((np.abs(Dh_tilde[0,1]))**2).sum()*xh*yh*zh)
        Dh02_total[i_ind] = np.sqrt(((np.abs(Dh_tilde[0,2]))**2).sum()*xh*yh*zh)
        Dh10_total[i_ind] = np.sqrt(((np.abs(Dh_tilde[1,0]))**2).sum()*xh*yh*zh)
        Dh11_total[i_ind] = np.sqrt(((np.abs(Dh_tilde[1,1]))**2).sum()*xh*yh*zh)
        Dh12_total[i_ind] = np.sqrt(((np.abs(Dh_tilde[1,2]))**2).sum()*xh*yh*zh)
        Dh20_total[i_ind] = np.sqrt(((np.abs(Dh_tilde[2,0]))**2).sum()*xh*yh*zh)
        Dh21_total[i_ind] = np.sqrt(((np.abs(Dh_tilde[2,1]))**2).sum()*xh*yh*zh)
        Dh22_total[i_ind] = np.sqrt(((np.abs(Dh_tilde[2,2]))**2).sum()*xh*yh*zh)


        Dh_off = -(gamma+omega_k)*Dh_tilde
        Dh_diag = -omega_k*Dh_tilde

        Dh00_exp[i_ind] = np.sqrt(((np.abs(Dh_diag[0,0]))**2).sum()*xh*yh*zh)
        Dh01_exp[i_ind] = np.sqrt(((np.abs(Dh_diag[0,1]))**2).sum()*xh*yh*zh)
        Dh02_exp[i_ind] = np.sqrt(((np.abs(Dh_off[0,2]))**2).sum()*xh*yh*zh)
        Dh10_exp[i_ind] = np.sqrt(((np.abs(Dh_diag[1,0]))**2).sum()*xh*yh*zh)
        Dh11_exp[i_ind] = np.sqrt(((np.abs(Dh_diag[1,1]))**2).sum()*xh*yh*zh)
        Dh12_exp[i_ind] = np.sqrt(((np.abs(Dh_off[1,2]))**2).sum()*xh*yh*zh)
        Dh20_exp[i_ind] = np.sqrt(((np.abs(Dh_off[2,0]))**2).sum()*xh*yh*zh)
        Dh21_exp[i_ind] = np.sqrt(((np.abs(Dh_off[2,1]))**2).sum()*xh*yh*zh)
        Dh22_exp[i_ind] = np.sqrt(((np.abs(Dh_diag[2,2]))**2).sum()*xh*yh*zh)



        # get the updates
        updateh = stress_update(Dh, Uh)

        # update D using Forward-Euler
        Dh += dt*updateh

        # apply viscosity using Backward-Euler
        Dh *= idiffuser

        # now let's modify D to be sure it is symmetric and satisfies eigenvalue constraints
        D = ifft3(Dh).real
        D = (D + D.transpose(1,0,2,3,4))/2.0
        Dd = np.transpose(D, (2,3,4,0,1))
        EV = np.linalg.eigh(Dd)
        Eval = EV[0]
        Evec = EV[1]
        # fix eigenvalues
        SEval = np.sum(Eval, -1)
        Eval = Eval / SEval[:,:,:,None]
        DEval = np.zeros((nx,ny,nz,3,3), dtype=float)
        for j in range(3):
            DEval[:,:,:,j,j] = Eval[:,:,:,j]
        Dd = np.einsum('...ij,...jk,...lk->...il',Evec,DEval,Evec)
        D = Dd.transpose((3,4,0,1,2))
        Dh = fft3(D)

        # compute Sigma
        Sigmah = Sigma_update(Dh, Uh)
        divSigmah = np.einsum('i...,ij...->j...',iks,Sigmah)

        # update the velocity field
        fh = fh_driving + divSigmah
##        if t < initial_perturbation_time:
##            fh_random[0][:] = full_rand(nx, ny, nz, initial_perturbation_size)
##            fh_random[1][:] = full_rand(nx, ny, nz, initial_perturbation_size)
##            fh_random[2][:] = full_rand(nx, ny, nz, initial_perturbation_size)
##            fh += fh_random
        Uh, ph = stokes(fh)

        # update the time variables
        t += dt
    print('   Time: {:0.4f}'.format(t), 'of', tmax, '\r', end='')

    end_time = time.time()
    print ('\n\nSimulation took {:0.3f}'.format(time.time() - start_time), 'seconds')

    print('END OF SIMULATION')
    return Dh00_total, Dh01_total,Dh02_total,Dh10_total,Dh11_total,Dh12_total,Dh20_total,Dh21_total,Dh22_total

Dh00_total, Dh01_total,Dh02_total,Dh10_total,Dh11_total,Dh12_total,Dh20_total,Dh21_total,Dh22_total = simulation(Dh)

file_Dh00_total = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh00_total.txt'
file_Dh01_total = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh01_total.txt'
file_Dh02_total = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh02_total.txt'
file_Dh10_total = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh10_total.txt'
file_Dh11_total = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh11_total.txt'
file_Dh12_total = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh12_total.txt'
file_Dh20_total = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh20_total.txt'
file_Dh21_total = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh21_total.txt'
file_Dh22_total = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh22_total.txt'

file_Dh00_exp = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh00_exp.txt'
file_Dh01_exp = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh01_exp.txt'
file_Dh02_exp = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh02_exp.txt'
file_Dh10_exp = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh10_exp.txt'
file_Dh11_exp = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh11_exp.txt'
file_Dh12_exp = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh12_exp.txt'
file_Dh20_exp = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh20_exp.txt'
file_Dh21_exp = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh21_exp.txt'
file_Dh22_exp = '/mnt/home/jflores/Doi-OnsagerModel/plane_wave_pert_z/D_pert/Dh22_exp.txt'

np.savetxt(file_Dh00_total,Dh00_total)
np.savetxt(file_Dh01_total,Dh01_total)
np.savetxt(file_Dh02_total,Dh02_total)
np.savetxt(file_Dh10_total,Dh10_total)
np.savetxt(file_Dh11_total,Dh11_total)
np.savetxt(file_Dh12_total,Dh12_total)
np.savetxt(file_Dh20_total,Dh20_total)
np.savetxt(file_Dh21_total,Dh21_total)
np.savetxt(file_Dh22_total,Dh22_total)

np.savetxt(file_Dh00_exp,Dh00_exp)
np.savetxt(file_Dh01_exp,Dh01_exp)
np.savetxt(file_Dh02_exp,Dh02_exp)
np.savetxt(file_Dh10_exp,Dh10_exp)
np.savetxt(file_Dh11_exp,Dh11_exp)
np.savetxt(file_Dh12_exp,Dh12_exp)
np.savetxt(file_Dh20_exp,Dh20_exp)
np.savetxt(file_Dh21_exp,Dh21_exp)
np.savetxt(file_Dh22_exp,Dh22_exp)


##
##
######################
#######################
##
##
####open files
##
##file_Dh00_total = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh00_total.txt'
##file_Dh01_total = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh01_total.txt'
##file_Dh02_total = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh02_total.txt'
##file_Dh10_total = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh10_total.txt'
##file_Dh11_total = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh11_total.txt'
##file_Dh12_total = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh12_total.txt'
##file_Dh20_total = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh20_total.txt'
##file_Dh21_total = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh21_total.txt'
##file_Dh22_total = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh22_total.txt'
##
##file_Dh00_exp = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh00_exp.txt'
##file_Dh01_exp = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh01_exp.txt'
##file_Dh02_exp = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh02_exp.txt'
##file_Dh10_exp = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh10_exp.txt'
##file_Dh11_exp = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh11_exp.txt'
##file_Dh12_exp = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh12_exp.txt'
##file_Dh20_exp = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh20_exp.txt'
##file_Dh21_exp = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh21_exp.txt'
##file_Dh22_exp = '/Users/JessFlores/Desktop/flatiron-2018/3d_data/plane_wave_pert_z/Dh22_exp.txt'
##
##Dh00_total_data = [np.loadtxt(file_Dh00_total)]
##Dh01_total_data = [np.loadtxt(file_Dh01_total)]
##Dh02_total_data = [np.loadtxt(file_Dh02_total)]
##Dh10_total_data = [np.loadtxt(file_Dh10_total)]
##Dh11_total_data = [np.loadtxt(file_Dh11_total)]
##Dh12_total_data = [np.loadtxt(file_Dh12_total)]
##Dh20_total_data = [np.loadtxt(file_Dh20_total)]
##Dh21_total_data = [np.loadtxt(file_Dh21_total)]
##Dh22_total_data = [np.loadtxt(file_Dh22_total)]
##
##Dh00_exp_data = [np.loadtxt(file_Dh00_exp)]
##Dh01_exp_data = [np.loadtxt(file_Dh01_exp)]
##Dh02_exp_data = [np.loadtxt(file_Dh02_exp)]
##Dh10_exp_data = [np.loadtxt(file_Dh10_exp)]
##Dh11_exp_data = [np.loadtxt(file_Dh11_exp)]
##Dh12_exp_data = [np.loadtxt(file_Dh12_exp)]
##Dh20_exp_data = [np.loadtxt(file_Dh20_exp)]
##Dh21_exp_data = [np.loadtxt(file_Dh21_exp)]
##Dh22_exp_data = [np.loadtxt(file_Dh22_exp)]
##
##Dh00_total = np.array(Dh00_total_data[0])
##Dh01_total = np.array(Dh01_total_data[0])
##Dh02_total = np.array(Dh02_total_data[0])
##Dh10_total = np.array(Dh10_total_data[0])
##Dh11_total = np.array(Dh11_total_data[0])
##Dh12_total = np.array(Dh12_total_data[0])
##Dh20_total = np.array(Dh20_total_data[0])
##Dh21_total = np.array(Dh21_total_data[0])
##Dh22_total = np.array(Dh22_total_data[0])
##
##Dh00_exp = np.array(Dh00_exp_data[0])
##Dh01_exp = np.array(Dh01_exp_data[0])
##Dh02_exp = np.array(Dh02_exp_data[0])
##Dh10_exp = np.array(Dh10_exp_data[0])
##Dh11_exp = np.array(Dh11_exp_data[0])
##Dh12_exp = np.array(Dh12_exp_data[0])
##Dh20_exp = np.array(Dh20_exp_data[0])
##Dh21_exp = np.array(Dh21_exp_data[0])
##Dh22_exp = np.array(Dh22_exp_data[0])
##
##def f(x,A,B):
##    return A*x+B
##
#### take derivatives
##
##Dh00_big = np. pad(Dh00_total,1,mode='wrap')
##Dh00_t = (Dh00_big[2:]-Dh00_big[:-2])/(2*dt)
##Dh00_exp_big = np.pad(Dh00_exp,1,mode='wrap')
##Dh00_exp_t = (Dh00_exp_big[2:]-Dh00_exp_big[:-2])/(2*dt)
##
##Dh00_t_lin = []
##Dh00_exp_t_lin = []
##Dh00_total_lin = []
##for i in range(len(Dh00_t)):
##    if Dh00_t[i] > 0.2 and Dh00_t[i] < 250 and Dh00_total[i] < 300:
##        Dh00_t_lin.append(Dh00_t[i])
##        Dh00_exp_t_lin.append(Dh00_exp_t[i])
##        Dh00_total_lin.append(Dh00_total[i])
##Dh00_total_lin = np.array(Dh00_total_lin)   
##A00,B00 = curve_fit(f,Dh00_total_lin,Dh00_t_lin)[0]
##A00_exp,B00_exp = curve_fit(f,Dh00_total_lin,Dh00_exp_t_lin)[0]
##y_00 = np.zeros(Dh00_total_lin.shape)
##y_00 = A00*Dh00_total_lin+B00
##y_00_exp = np.zeros(Dh00_total_lin.shape)
##y_00_exp = A00_exp*Dh00_total_lin+B00_exp
##
##
##Dh01_big = np. pad(Dh01_total,1,mode='wrap')
##Dh01_t = (Dh01_big[2:]-Dh01_big[:-2])/(2*dt)
##Dh01_exp_big = np.pad(Dh01_exp,1,mode='wrap')
##Dh01_exp_t = (Dh01_exp_big[2:]-Dh01_exp_big[:-2])/(2*dt)
##
##Dh01_t_lin = []
##Dh01_exp_t_lin = []
##Dh01_total_lin = []
##for i in range(len(Dh01_t)):
##    if Dh01_t[i] > 0.2 and Dh01_t[i] < 200 and Dh01_total[i] < 400:
##        Dh01_t_lin.append(Dh01_t[i])
##        Dh01_exp_t_lin.append(Dh01_exp_t[i])
##        Dh01_total_lin.append(Dh01_total[i])
##Dh01_total_lin = np.array(Dh01_total_lin)   
##A01,B01 = curve_fit(f,Dh01_total_lin,Dh01_t_lin)[0]
##A01_exp,B01_exp = curve_fit(f,Dh01_total_lin,Dh01_exp_t_lin)[0]
##y_01 = np.zeros(Dh01_total_lin.shape)
##y_01 = A01*Dh01_total_lin+B01
##y_01_exp = np.zeros(Dh01_total_lin.shape)
##y_01_exp = A01_exp*Dh01_total_lin+B01_exp
##
##
##Dh02_big = np. pad(Dh02_total,1,mode='wrap')
##Dh02_t = (Dh02_big[2:]-Dh02_big[:-2])/(2*dt)
##Dh02_exp_big = np.pad(Dh02_exp,1,mode='wrap')
##Dh02_exp_t = (Dh02_exp_big[2:]-Dh02_exp_big[:-2])/(2*dt)
##
##Dh02_t_lin = []
##Dh02_exp_t_lin = []
##Dh02_total_lin = []
##for i in range(len(Dh02_t)):
##    if Dh02_t[i] > 0.2 and Dh02_t[i] < 200 and Dh02_total[i] < 350:
##        Dh02_t_lin.append(Dh02_t[i])
##        Dh02_exp_t_lin.append(Dh02_exp_t[i])
##        Dh02_total_lin.append(Dh02_total[i])
##Dh02_total_lin = np.array(Dh02_total_lin)   
##A02,B02 = curve_fit(f,Dh02_total_lin,Dh02_t_lin)[0]
##A02_exp,B02_exp = curve_fit(f,Dh02_total_lin,Dh02_exp_t_lin)[0]
##y_02 = np.zeros(Dh02_total_lin.shape)
##y_02 = A02*Dh02_total_lin+B02
##y_02_exp = np.zeros(Dh02_total_lin.shape)
##y_02_exp = A02_exp*Dh02_total_lin+B02_exp
##
##
##Dh10_big = np. pad(Dh10_total,1,mode='wrap')
##Dh10_t = (Dh10_big[2:]-Dh10_big[:-2])/(2*dt)
##Dh10_exp_big = np.pad(Dh10_exp,1,mode='wrap')
##Dh10_exp_t = (Dh10_exp_big[2:]-Dh10_exp_big[:-2])/(2*dt)
##Dh10_t_lin = []
##Dh10_exp_t_lin = []
##Dh10_total_lin = []
##for i in range(len(Dh10_t)):
##    if Dh10_t[i] > 0.2 and Dh10_t[i] < 200 and Dh10_total[i] < 400:
##        Dh10_t_lin.append(Dh10_t[i])
##        Dh10_exp_t_lin.append(Dh10_exp_t[i])
##        Dh10_total_lin.append(Dh10_total[i])
##Dh10_total_lin = np.array(Dh10_total_lin)   
##A10,B10 = curve_fit(f,Dh10_total_lin,Dh10_t_lin)[0]
##A10_exp,B10_exp = curve_fit(f,Dh10_total_lin,Dh10_exp_t_lin)[0]
##y_10 = np.zeros(Dh10_total_lin.shape)
##y_10 = A10*Dh10_total_lin+B10
##y_10_exp = np.zeros(Dh10_total_lin.shape)
##y_10_exp = A10_exp*Dh10_total_lin+B10_exp
##
##Dh11_big = np. pad(Dh11_total,1,mode='wrap')
##Dh11_t = (Dh11_big[2:]-Dh11_big[:-2])/(2*dt)
##Dh11_exp_big = np.pad(Dh11_exp,1,mode='wrap')
##Dh11_exp_t = (Dh11_exp_big[2:]-Dh11_exp_big[:-2])/(2*dt)
##Dh11_t_lin = []
##Dh11_exp_t_lin = []
##Dh11_total_lin = []
##for i in range(len(Dh11_t)):
##    if Dh11_t[i] > 0.2 and Dh11_t[i] < 150 and Dh11_total[i] < 200:
##        Dh11_t_lin.append(Dh11_t[i])
##        Dh11_exp_t_lin.append(Dh11_exp_t[i])
##        Dh11_total_lin.append(Dh11_total[i])
##Dh11_total_lin = np.array(Dh11_total_lin)   
##A11,B11 = curve_fit(f,Dh11_total_lin,Dh11_t_lin)[0]
##A11_exp,B11_exp = curve_fit(f,Dh11_total_lin,Dh11_exp_t_lin)[0]
##y_11 = np.zeros(Dh11_total_lin.shape)
##y_11 = A11*Dh11_total_lin+B11 
##y_11_exp = np.zeros(Dh11_total_lin.shape)
##y_11_exp = A11_exp*Dh11_total_lin+B11_exp
##
##Dh12_big = np. pad(Dh12_total,1,mode='wrap')
##Dh12_t = (Dh12_big[2:]-Dh12_big[:-2])/(2*dt)
##Dh12_exp_big = np.pad(Dh12_exp,1,mode='wrap')
##Dh12_exp_t = (Dh12_exp_big[2:]-Dh12_exp_big[:-2])/(2*dt)
##Dh12_t_lin = []
##Dh12_exp_t_lin = []
##Dh12_total_lin = []
##for i in range(len(Dh12_t)):
##    if Dh12_t[i] > 0.2 and Dh12_t[i] < 300 and Dh12_total[i] < 400:
##        Dh12_t_lin.append(Dh12_t[i])
##        Dh12_exp_t_lin.append(Dh12_exp_t[i])
##        Dh12_total_lin.append(Dh12_total[i])
##Dh12_total_lin = np.array(Dh12_total_lin)   
##A12,B12 = curve_fit(f,Dh12_total_lin,Dh12_t_lin)[0]
##A12_exp,B12_exp = curve_fit(f,Dh12_total_lin,Dh12_exp_t_lin)[0]
##y_12 = np.zeros(Dh12_total_lin.shape)
##y_12 = A12*Dh12_total_lin+B12
##y_12_exp = np.zeros(Dh12_total_lin.shape)
##y_12_exp = A12_exp*Dh12_total_lin+B12_exp
##
##
##Dh20_big = np. pad(Dh20_total,1,mode='wrap')
##Dh20_t = (Dh20_big[2:]-Dh20_big[:-2])/(2*dt)
##Dh20_exp_big = np.pad(Dh20_exp,1,mode='wrap')
##Dh20_exp_t = (Dh20_exp_big[2:]-Dh20_exp_big[:-2])/(2*dt)
##Dh20_t_lin = []
##Dh20_exp_t_lin = []
##Dh20_total_lin = []
##for i in range(len(Dh20_t)):
##    if Dh20_t[i] > 0.2 and Dh20_t[i] < 200 and Dh20_total[i] < 400:
##        Dh20_t_lin.append(Dh20_t[i])
##        Dh20_exp_t_lin.append(Dh20_exp_t[i])
##        Dh20_total_lin.append(Dh20_total[i])
##Dh20_total_lin = np.array(Dh20_total_lin)   
##A20,B20 = curve_fit(f,Dh20_total_lin,Dh20_t_lin)[0]
##A20_exp,B20_exp = curve_fit(f,Dh20_total_lin,Dh20_exp_t_lin)[0]
##y_20 = np.zeros(Dh20_total_lin.shape)
##y_20 = A20*Dh20_total_lin+B20
##y_20_exp = np.zeros(Dh20_total_lin.shape)
##y_20_exp = A20_exp*Dh20_total_lin+B20_exp
##
##Dh21_big = np. pad(Dh21_total,1,mode='wrap')
##Dh21_t = (Dh21_big[2:]-Dh21_big[:-2])/(2*dt)
##Dh21_exp_big = np.pad(Dh21_exp,1,mode='wrap')
##Dh21_exp_t = (Dh21_exp_big[2:]-Dh21_exp_big[:-2])/(2*dt)
##Dh21_t_lin = []
##Dh21_exp_t_lin = []
##Dh21_total_lin = []
##for i in range(len(Dh21_t)):
##    if Dh21_t[i] > 0.2 and Dh21_t[i] < 300 and Dh21_total[i] < 300:
##        Dh21_t_lin.append(Dh21_t[i])
##        Dh21_exp_t_lin.append(Dh21_exp_t[i])
##        Dh21_total_lin.append(Dh21_total[i])
##Dh21_total_lin = np.array(Dh21_total_lin)   
##A21,B21 = curve_fit(f,Dh21_total_lin,Dh21_t_lin)[0]
##A21_exp,B21_exp = curve_fit(f,Dh21_total_lin,Dh21_exp_t_lin)[0]
##y_21 = np.zeros(Dh21_total_lin.shape)
##y_21 = A21*Dh21_total_lin+B21 
##y_21_exp = np.zeros(Dh21_total_lin.shape)
##y_21_exp = A21_exp*Dh21_total_lin+B21_exp
##
##Dh22_big = np. pad(Dh22_total,1,mode='wrap')
##Dh22_t = (Dh22_big[2:]-Dh22_big[:-2])/(2*dt)
##Dh22_exp_big = np.pad(Dh22_exp,1,mode='wrap')
##Dh22_exp_t = (Dh22_exp_big[2:]-Dh22_exp_big[:-2])/(2*dt)
##Dh22_t_lin = []
##Dh22_exp_t_lin = []
##Dh22_total_lin = []
##for i in range(len(Dh22_t)):
##    if Dh22_t[i] > 0.2 and Dh22_t[i] < 140 and Dh22_total[i] < 160:
##        Dh22_t_lin.append(Dh22_t[i])
##        Dh22_exp_t_lin.append(Dh22_exp_t[i])
##        Dh22_total_lin.append(Dh22_total[i])
##Dh22_total_lin = np.array(Dh22_total_lin)   
##A22,B22 = curve_fit(f,Dh22_total_lin,Dh22_t_lin)[0]
##A22_exp,B22_exp = curve_fit(f,Dh22_total_lin,Dh22_exp_t_lin)[0]
##y_22 = np.zeros(Dh22_total_lin.shape)
##y_22 = A22*Dh22_total_lin+B22
##y_22_exp = np.zeros(Dh22_total_lin.shape)
##y_22_exp = A22_exp*Dh22_total_lin+B22_exp
##
##gs = gridspec.GridSpec(3,3)
##plt.figure()
##plt.suptitle('$\\xi$ = '+str("{0:.2f}".format(kesi))+  ', $\\beta$ = '+str("{0:.2f}".format(beta)) +', $\\alpha$ = '+str("{0:.2f}".format(alpha)) +', $d_R$ = '+str("{0:.2f}".format(d_R))+', $d_T$ = '+str("{0:.2f}".format(d_T))+', Plane wave pert = $e^{ik_zz}*initial\\_perturbation\\_size*nz/(2\\pi)$',fontsize=8  )
##
##ax = plt.subplot(gs[0,0])
##plt.plot(Dh00_total_lin,y_00,label='$(\\hat{D}^{\\prime}_{11})_t=$'+str("{0:.2f}").format(A00)+'$\\hat{D}^{\\prime}_{11} $' )
##plt.plot(Dh00_total_lin,Dh00_t_lin,':')
##plt.plot(Dh00_total_lin,y_00_exp,label='$(\\hat{D}^{\\prime}_{11})_t=-\\omega_k\\hat{D}^{\\prime}_{11}=$'+str("{0:.2f}").format(A00_exp)+'$\\hat{D}^{\\prime}_{11} $' )
##plt.plot(Dh00_total_lin,Dh00_exp_t_lin,':')
##plt.xlabel('$\\hat{D}^{\\prime}_{11} $')
##plt.ylabel('$(\\hat{D}^{\\prime}_{11})_t$')
##plt.legend(loc='upper left',prop={'size':6})
##
##ax = plt.subplot(gs[0,1])
##plt.plot(Dh01_total_lin,y_01,label='$(\\hat{D}^{\\prime}_{12})_t=$'+str("{0:.2f}").format(A01)+'$\\hat{D}^{\\prime}_{12} $' )
##plt.plot(Dh01_total_lin,Dh01_t_lin,':')
##plt.plot(Dh01_total_lin,y_01_exp,label='$(\\hat{D}^{\\prime}_{12})_t=-\\omega_k\\hat{D}^{\\prime}_{12}=$'+str("{0:.2f}").format(A01_exp)+'$\\hat{D}^{\\prime}_{12} $' )
##plt.plot(Dh01_total_lin,Dh01_exp_t_lin,':')
##plt.xlabel('$\\hat{D}^{\\prime}_{12} $')
##plt.ylabel('$(\\hat{D}^{\\prime}_{12})_t$')
##plt.legend(loc='upper left',prop={'size':6})
##
##ax = plt.subplot(gs[0,2])
##plt.plot(Dh02_total_lin,y_02,label='$(\\hat{D}^{\\prime}_{13})_t=$'+str("{0:.2f}").format(A02)+'$\\hat{D}^{\\prime}_{13} $' )
##plt.plot(Dh02_total_lin,Dh02_t_lin,':')
##plt.plot(Dh02_total_lin,y_02_exp,label='$(\\hat{D}^{\\prime}_{13})_t=-(\\gamma+\\omega_k)\\hat{D}^{\\prime}_{13}=$'+str("{0:.2f}").format(A02_exp)+'$\\hat{D}^{\\prime}_{13} $' )
##plt.plot(Dh02_total_lin,Dh02_exp_t_lin,':')
##plt.xlabel('$\\hat{D}^{\\prime}_{13} $')
##plt.ylabel('$(\\hat{D}^{\\prime}_{13})_t$')
##plt.legend(loc='upper left',prop={'size':6})
##
##ax = plt.subplot(gs[1,0])
##plt.plot(Dh10_total_lin,y_10,label='$(\\hat{D}^{\\prime}_{21})_t=$'+str("{0:.2f}").format(A10)+'$\\hat{D}^{\\prime}_{21} $' )
##plt.plot(Dh10_total_lin,Dh10_t_lin,':')
##plt.plot(Dh10_total_lin,y_10_exp,label='$(\\hat{D}^{\\prime}_{21})_t=-\\omega_k\\hat{D}^{\\prime}_{21}=$'+str("{0:.2f}").format(A10_exp)+'$\\hat{D}^{\\prime}_{21} $' )
##plt.plot(Dh10_total_lin,Dh10_exp_t_lin,':')
##plt.xlabel('$\\hat{D}^{\\prime}_{21} $')
##plt.ylabel('$(\\hat{D}^{\\prime}_{21})_t$')
##plt.legend(loc='upper left',prop={'size':6})
##
##ax = plt.subplot(gs[1,1])
##plt.plot(Dh11_total_lin,y_11,label='$(\\hat{D}^{\\prime}_{22})_t=$'+str("{0:.2f}").format(A11)+'$\\hat{D}^{\\prime}_{22} $' )
##plt.plot(Dh11_total_lin,Dh11_t_lin,':')
##plt.plot(Dh11_total_lin,y_11_exp,label='$(\\hat{D}^{\\prime}_{22})_t=-\\omega_k\\hat{D}^{\\prime}_{22}=$'+str("{0:.2f}").format(A11_exp)+'$\\hat{D}^{\\prime}_{22} $' )
##plt.plot(Dh11_total_lin,Dh11_exp_t_lin,':')
##plt.xlabel('$\\hat{D}^{\\prime}_{22} $')
##plt.ylabel('$(\\hat{D}^{\\prime}_{22})_t$')
##plt.legend(loc='upper left',prop={'size':6})
##
##ax = plt.subplot(gs[1,2])
##plt.plot(Dh12_total_lin,y_12,label='$(\\hat{D}^{\\prime}_{23})_t=$'+str("{0:.2f}").format(A12)+'$\\hat{D}^{\\prime}_{23} $' )
##plt.plot(Dh12_total_lin,Dh12_t_lin,':')
##plt.plot(Dh12_total_lin,y_12_exp,label='$(\\hat{D}^{\\prime}_{23})_t=-(\\gamma+\\omega_k)\\hat{D}^{\\prime}_{23}=$'+str("{0:.2f}").format(A12_exp)+'$\\hat{D}^{\\prime}_23} $' )
##plt.plot(Dh12_total_lin,Dh12_exp_t_lin,':')
##plt.xlabel('$\\hat{D}^{\\prime}_{23} $')
##plt.ylabel('$(\\hat{D}^{\\prime}_{23})_t$')
##plt.legend(loc='upper left',prop={'size':6})
##
##ax = plt.subplot(gs[2,0])
##plt.plot(Dh20_total_lin,y_20,label='$(\\hat{D}^{\\prime}_{31})_t=$'+str("{0:.2f}").format(A20)+'$\\hat{D}^{\\prime}_{31} $' )
##plt.plot(Dh20_total_lin,Dh20_t_lin,':')
##plt.plot(Dh20_total_lin,y_20_exp,label='$(\\hat{D}^{\\prime}_{31})_t=-(\\gamma+\\omega_k)\\hat{D}^{\\prime}_{31}=$'+str("{0:.2f}").format(A20_exp)+'$\\hat{D}^{\\prime}_{31} $' )
##plt.plot(Dh20_total_lin,Dh20_exp_t_lin,':')
##plt.xlabel('$\\hat{D}^{\\prime}_{31} $')
##plt.ylabel('$(\\hat{D}^{\\prime}_{31})_t$')
##plt.legend(loc='upper left',prop={'size':6})
##
##ax = plt.subplot(gs[2,1])
##plt.plot(Dh21_total_lin,y_21,label='$(\\hat{D}^{\\prime}_{32})_t=$'+str("{0:.2f}").format(A21)+'$\\hat{D}^{\\prime}_{32} $' )
##plt.plot(Dh21_total_lin,Dh21_t_lin,':')
##plt.plot(Dh21_total_lin,y_21_exp,label='$(\\hat{D}^{\\prime}_{32})_t=-(\\gamma+\\omega_k)\\hat{D}^{\\prime}_{32}=$'+str("{0:.2f}").format(A21_exp)+'$\\hat{D}^{\\prime}_{32} $' )
##plt.plot(Dh21_total_lin,Dh21_exp_t_lin,':')
##plt.xlabel('$\\hat{D}^{\\prime}_{32} $')
##plt.ylabel('$(\\hat{D}^{\\prime}_{32})_t$')
##plt.legend(loc='upper left',prop={'size':6})
##
##ax = plt.subplot(gs[2,2])
##plt.plot(Dh22_total_lin,y_22,label='$(\\hat{D}^{\\prime}_{33})_t=$'+str("{0:.2f}").format(A22)+'$\\hat{D}^{\\prime}_{33} $' )
##plt.plot(Dh22_total_lin,Dh22_t_lin,':')
##plt.plot(Dh22_total_lin,y_22_exp,label='$(\\hat{D}^{\\prime}_{33})_t=-\\omega_k\\hat{D}^{\\prime}_{33}=$'+str("{0:.2f}").format(A22_exp)+'$\\hat{D}^{\\prime}_{33} $' )
##plt.plot(Dh22_total_lin,Dh22_exp_t_lin,':')
##plt.xlabel('$\\hat{D}^{\\prime}_{33} $')
##plt.ylabel('$(\\hat{D}^{\\prime}_{33})_t$')
##plt.legend(loc='upper left',prop={'size':6})
##
##plt.tight_layout()
##filename = '/mnt/home/jflores/3D_stability.png'
##plt.savefig(filename)
##plt.close()
##
