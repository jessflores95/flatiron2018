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


################################################################################
### Parameters

k_loop=np.linspace(0,5,5)
m00_loop = np.zeros(k_loop.shape,dtype=float)
m01_loop = np.zeros(k_loop.shape,dtype=float)
m02_loop = np.zeros(k_loop.shape,dtype=float)
m10_loop = np.zeros(k_loop.shape,dtype=float)
m11_loop = np.zeros(k_loop.shape,dtype=float)
m12_loop = np.zeros(k_loop.shape,dtype=float)
m20_loop = np.zeros(k_loop.shape,dtype=float)
m21_loop = np.zeros(k_loop.shape,dtype=float)
m22_loop = np.zeros(k_loop.shape,dtype=float)

m00_exp = np.zeros(k_loop.shape,dtype=float)
m01_exp = np.zeros(k_loop.shape,dtype=float)
m02_exp = np.zeros(k_loop.shape,dtype=float)
m10_exp = np.zeros(k_loop.shape,dtype=float)
m11_exp = np.zeros(k_loop.shape,dtype=float)
m12_exp = np.zeros(k_loop.shape,dtype=float)
m20_exp = np.zeros(k_loop.shape,dtype=float)
m21_exp = np.zeros(k_loop.shape,dtype=float)
m22_exp = np.zeros(k_loop.shape,dtype=float)

def f(x,A,B):
    return A*x+B

def L2_diff(Dh_t,xh,yh,zh):
    return np.sqrt( (np.abs(Dh_t)**2).sum() *xh*yh*zh )

for q_ind,q in enumerate(k_loop):
    k = k_loop[q_ind]
    nxpow = 6           # nx = 2**nxpow
    nypow = 6           # ny = 2**nypow
    nzpow = 6           # nz = 2**nzpow
    xmin = -np.pi
    xmax =  np.pi
    ymin = -np.pi
    ymax =  np.pi
    zmin = -np.pi
    zmax =  np.pi
    #tmax = 50           # end time for simulation
    dt_modifier = 5     # dt = 2**6 * 0.1 / n / dt_modifier
    d_R = 0.05           # rotational diffusion coefficient
    d_T = 0.05           # translational diffusion coefficient
    kesi = 0.5          # mean-field torque strength
    alpha = -1.0        # extensile stretching coefficient
    beta = 0.874          # shape factor
    phi = 1.0           # concentration
    closure = 'bingham'     # choose which closure to use ('doi', 'K1', 'bingham')
    S2_cutoff = 1e-12   # cutoff for isotropic case (only for K1 closure)
    background_force_magnitude = 0.0
    initial_perturbation_size = 1.0e-3
    gamma = (1.0/5.0)*( (alpha-(2.0/5.0)*kesi*beta)/ (1+(beta/15.0)) )
    omega = d_T*(k**2) + 6*d_R - (4.0/5.0)*kesi
    tmax = np.abs(math.floor(-np.log(initial_perturbation_size)/(gamma+omega)))

    ################################################################################
    ### Parameters that probably shouldn't be changed (or just depend on other parameters)

    nx = 2**nxpow
    ny = 2**nypow
    nz = 2**nzpow
    dt0 = 2**6*0.1 / (nx*ny*nz)**(1/3.) / dt_modifier

    initial_perturbation_time = 5*dt0

    ################################################################################
    ### Setupnu

    print('\nSetting things up')

    # grid on which physical equations live
    xran = xmax - xmin
    yran = ymax - ymin
    zran = zmax - zmin
    xv, xh = np.linspace( xmin, xmax, nx, endpoint = False, retstep = True )
    yv, yh = np.linspace( ymin, ymax, ny, endpoint = False, retstep = True )
    zv, zh = np.linspace( zmin, zmax, nz, endpoint = False, retstep = True )
    x, y, z = np.meshgrid(xv, yv, zv, indexing='ij')
    if np.max(np.abs(np.array((xh-yh,yh-zh,xh-zh)))) > 1e-15:
        raise ValueError('Please define domain bounds and ns such that the grid spacing is isotropic')
    h = xh

    n_timesteps = int(np.ceil(tmax / dt0))
    t_total=mp.linspace(0,tmax,n_timesteps)
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
    fh_random = np.zeros((3,nx,ny,nz), dtype=complex)
    my_rand = lambda sh: (np.random.rand(*sh).reshape(sh)-0.5)*2
    complex_rand = lambda sh: my_rand(sh) + 1j*my_rand(sh)
    def full_rand(nx, ny, nz, initial_perturbation_size):
        rand = complex_rand([nx,ny,nz])*initial_perturbation_size*nx*ny*nz/(2*np.pi)
        rand[5:-5,:,:] = 0.0
        rand[:,5:-5,:] = 0.0
        rand[:,:,5:-5] = 0.0
        return rand

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


    ## add plane wave perturbation to D, perturbation is both symmetric and has trace zero
    D_pert = np.zeros(D.shape,dtype=float)
    D_pert[0,0] = -(1.0/3.0)*np.cos(k*z)*initial_perturbation_size
    D_pert[0,1] = np.cos(k*z)*initial_perturbation_size
    D_pert[0,2] = np.cos(k*z)*initial_perturbation_size
    D_pert[1,0] = np.cos(k*z)*initial_perturbation_size
    D_pert[1,1] = -(1.0/3.0)*np.cos(k*z)*initial_perturbation_size
    D_pert[1,2] = np.cos(k*z)*initial_perturbation_size
    D_pert[2,0] = np.cos(k*z)*initial_perturbation_size
    D_pert[2,1] = np.cos(k*z)*initial_perturbation_size
    D_pert[2,2] = (2.0/3.0)*np.cos(k*z)*initial_perturbation_size
    D += D_pert #D remains symmetric and conserves trace 1 with addition of perturbation

    # take the FFT of D
    Dh = fft3(D)

    err_Dh00 = np.zeros(n_timesteps,dtype=float)
    err_Dh01 = np.zeros(n_timesteps,dtype=float)
    err_Dh02 = np.zeros(n_timesteps,dtype=float)
    err_Dh10 = np.zeros(n_timesteps,dtype=float)
    err_Dh11 = np.zeros(n_timesteps,dtype=float)
    err_Dh12 = np.zeros(n_timesteps,dtype=float)
    err_Dh20 = np.zeros(n_timesteps,dtype=float)
    err_Dh21 = np.zeros(n_timesteps,dtype=float)
    err_Dh22 = np.zeros(n_timesteps,dtype=float)

    ns = np.arange(0,n_timesteps,1) 
    # set time to 0
    t = 0

    # get CPU time at start of computation
    start_time = time.time()

    ################################################################################
    ### Timestepping Loop

    print('\nRunning simulation')

    NUM = 1
    for i _ind, i in enumerate(ns):
        print('   Time: {:0.4f}'.format(t), 'of', tmax, '\r', end='')
        sys.stdout.flush()
        D_tilde = D - D_pure

        err_Dh00 = L2_diff(D_tilde[0,0],xh,yh,zh)
        err_Dh01 = L2_diff(D_tilde[0,1],xh,yh,zh)
        err_Dh02 = L2_diff(D_tilde[0,2],xh,yh,zh)
        err_Dh10 = L2_diff(D_tilde[1,0],xh,yh,zh)
        err_Dh11 = L2_diff(D_tilde[1,1],xh,yh,zh)
        err_Dh12 = L2_diff(D_tilde[1,2],xh,yh,zh)
        err_Dh20 = L2_diff(D_tilde[2,0],xh,yh,zh)
        err_Dh21 = L2_diff(D_tilde[2,1],xh,yh,zh)
        err_Dh22 = L2_diff(D_tilde[2,2],xh,yh,zh)

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
        if t < initial_perturbation_time:
            fh_random[0][:] = full_rand(nx, ny, nz, initial_perturbation_size)
            fh_random[1][:] = full_rand(nx, ny, nz, initial_perturbation_size)
            fh_random[2][:] = full_rand(nx, ny, nz, initial_perturbation_size)
            fh += fh_random
        Uh, ph = stokes(fh)

        # update the time variables
        t += dt
    print('   Time: {:0.4f}'.format(t), 'of', tmax, '\r', end='')

    end_time = time.time()
    print ('\n\nSimulation took {:0.3f}'.format(time.time() - start_time), 'seconds')

    print('END OF SIMULATION')

    m00_loop[q_ind] = (curve_fit(f,t_total,np.log(err_Dh00))[0])[0]
    m01_loop[q_ind] = (curve_fit(f,t_total,np.log(err_Dh01))[0])[0]
    m02_loop[q_ind] = (curve_fit(f,t_total,np.log(err_Dh02))[0])[0]
    m10_loop[q_ind] = (curve_fit(f,t_total,np.log(err_Dh10))[0])[0]
    m11_loop[q_ind] = (curve_fit(f,t_total,np.log(err_Dh11))[0])[0]
    m12_loop[q_ind] = (curve_fit(f,t_total,np.log(err_Dh12))[0])[0]
    m20_loop[q_ind] = (curve_fit(f,t_total,np.log(err_Dh20))[0])[0]
    m21_loop[q_ind] = (curve_fit(f,t_total,np.log(err_Dh21))[0])[0]
    m21_loop[q_ind] = (curve_fit(f,t_total,np.log(err_Dh21))[0])[0]
    
    m00_exp[q_ind] = -omega
    m01_exp[q_ind] = -(omega+gamma)
    m02_exp[q_ind] = -(omega+gamma)
    m10_exp[q_ind] = -(omega+gamma)
    m11_exp[q_ind] = -omega
    m12_exp[q_ind] = -(omega+gamma)
    m20_exp[q_ind] = -(omega+gamma)
    m21_exp[q_ind] = -(omega+gamma)
    m21_exp[q_ind] = -omega

file_err_Dh00 = '/mnt/home/jflores/Doi-OnsagerModel/sigma/err_Dh00.txt'
file_err_Dh01 = '/mnt/home/jflores/Doi-OnsagerModel/sigma/err_Dh01.txt'
file_err_Dh02 = '/mnt/home/jflores/Doi-OnsagerModel/sigma/err_Dh02.txt'
file_err_Dh10 = '/mnt/home/jflores/Doi-OnsagerModel/sigma/err_Dh10.txt'
file_err_Dh11 = '/mnt/home/jflores/Doi-OnsagerModel/sigma/err_Dh11.txt'
file_err_Dh12 = '/mnt/home/jflores/Doi-OnsagerModel/sigma/err_Dh12.txt'
file_err_Dh20 = '/mnt/home/jflores/Doi-OnsagerModel/sigma/err_Dh20.txt'
file_err_Dh21 = '/mnt/home/jflores/Doi-OnsagerModel/sigma/err_Dh21.txt'
file_err_Dh22 = '/mnt/home/jflores/Doi-OnsagerModel/sigma/err_Dh22.txt'

np.savetxt(file_err_Dh00,m_00_loop)
np.savetxt(file_err_Dh01,m_01_loop)
np.savetxt(file_err_Dh02,m_02_loop)
np.savetxt(file_err_Dh10,m_10_loop)
np.savetxt(file_err_Dh11,m_11_loop)
np.savetxt(file_err_Dh12,m_12_loop)
np.savetxt(file_err_Dh20,m_20_loop)
np.savetxt(file_err_Dh21,m_21_loop)
np.savetxt(file_err_Dh22,m_22_loop)
