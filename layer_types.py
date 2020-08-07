#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of pyFresnel.
# Copyright (c) 2012-2016, Robert Steed
# Author: Robert Steed (rjsteed@talk21.com)
# License: GPL
# last modified 15.08.2016

#frequencies should be natural for compatability with the transfer matrix code.

import numpy as N
import os.path
from scipy.interpolate import interp1d,splrep,splev
sqrt=N.sqrt

eps0=8.8541E-12 #Farads/metres -vacuum permittivity.
m_e=9.1094e-31 #Kg - mass of electron
q=1.6022e-19 #C - unit charge.
c=299792458  #m/s - speed of light
pi=N.pi

class MaterialLayer(object):
    """Like a dielectric, plasma, quantum well etc"""
    def __init__(self,d,coh):
        """This base class has a circular definition for n and epsilon. One 
        function must be overridden in the derived class!
        
        d - layer thickness (m)
        coh - is layer coherent or incoherent? - could also be a number between 0 and pi to describe a partially coherent layer
        """        
        if type(self) == MaterialLayer:
            raise Exception("<MaterialLayer> must be subclassed.")
        self.d=d
        self.coh=coh
        
    def epsilon(self,w):
        return self.n(w)**2
    
    def n(self,w):
        return sqrt(self.epsilon(w))
        
    def __len__(self):
        pass
        
    def __add__(self,other): # this might not work once we start using the Claussius-Claussis relation!
        """Add two derived instances of classes derived from material"""
        def new_epsilon(self2,w):
            return self.epsilon(w)+other.epsilon(w)
        newmat=MaterialLayer()
        newmat.epsilon=new_epsilon.__get__(newmat,MaterialLayer) # binds function to instance
        return newmat
    
    def __repr__(self):
        return "Layer"+"("+repr(self._n)+", "+repr(self.d)+", coh="+repr(self.coh)+" )"
        
        
class AnisoMaterialLayer(MaterialLayer):
    """Anisotropic Layer with optical axis aligned perpendicular to the layer."""
    
    def epsilonzz(self,w):
        return self.nzz(w)**2
    
    def nzz(self,w):
        return sqrt(self.epsilonzz(w))


######################################

class MaterialEps(MaterialLayer):
    """initialise using material_eps(epsilon). epsilon can be a (complex) number or array"""
    def __init__(self,eps,d,coh=True):
        MaterialLayer.__init__(self,d,coh)
        self.eps=eps
    
    def epsilon(self,w): #although these take a frequency variable, they don't use it and so only use if the data fits the w rangeyou know what you are doing.
        return self.eps
    
class Material_nk(MaterialLayer):
    """initialise using material_nk(nk). nk can be a (complex) number or array"""
    def __init__(self,nk,d,coh=True):
        MaterialLayer.__init__(self,d,coh)
        self.nk=nk
        
    def n(self,w): #although these take a frequency variable, they don't use it and so only use if the data fits the w range.
        return self.nk

######################################
### SOPRA data

class SopraLayer(MaterialLayer):
    """Sopra data files can be downloaded from http://www.sspectra.com/sopra.html
    This class loads a file and provides an interpolated refractive index function"""
    directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),"sopra")
    def __init__(self,matname,d,coh=True):
        """loads and interpolates a SOPRA data file"""
        MaterialLayer.__init__(self,d,coh)
        self.name=matname
        with file(os.path.join(self.directory,matname+'.MAT')) as fobj:
            output=[]
            for line in fobj:
                linelist=line.split('*')
                if linelist[0]=='DATA1':
                    output.append(list(map(float,linelist[2:5]))) # data stored as wavelength(nm),n,k
            data=N.array(output)
        """take an array of column data: 
        spectral axis, n, k
        and creates a function that spline interoplates over the
        refractive index data
        """    
        w=c*2*pi*1e9/data[:,0] #natural frequency from wavelength (nm)
        #nk=data[:,1]+1j*data[:,2]
        #nkfunc = interp1d(axis,nk) #basic
        #nkfunc2 = interp1d(axis,nk,kind='quadratic') # doesn't work.
        nkspline_real = splrep(w[::-1],data[::-1,1],s=0)
        nkspline_imag = splrep(w[::-1],data[::-1,2],s=0) #library doesn't work with complex numbers?
        self._n=lambda axis: splev(axis,nkspline_real,der=0)+1j*splev(axis,nkspline_imag,der=0)
        self.wupper=max(w)
        self.wlower=min(w)
    
    def n(self,w):
        """w is a natural frequency range, a check is performed to see whether the values
        are within the dataset's frequency range"""
        #check range
        if max(w)>self.wupper or min(w)<self.wlower:
            raise Exception("%s :frequency range outside of material's data range" %self.name)
        return self._n(w)

    def __repr__(self):
        return "Layer"+"("+self.name+", "+repr(self.d)+", coh="+repr(self.coh)+" )"
        
######################################

class LorentzModel(MaterialLayer):
    """Simple model of an absorbing oscillator / transition.
    Frequencies - whether we use real or natural frequency doesn't matter as long as we are consistant! 
    Remember that there is a difference of 2pi between the two: w=2*pi*f
    Note that normally the equations for the plasma frequency will give a natural frequency but that otherwise
    will be interested in real frequencies."""
    def __init__(self,w0,y,wp,f,eps_b,d,coh=True):
        """Everything should be in natural frequencies - w0 y wp
        f is the unitless oscillator strength
        eps_b is the background dielectric constant
        d is the layer thickness.
        Strictly, if we use real frequencies for everything, (including the plasma frequency(and the normal equations give a natural value))
        it should still work."""
        MaterialLayer.__init__(self,d,coh)
        self.w0=w0
        self.y=y
        self.wp=wp
        self.f=f
        self.eps_b=eps_b
    
    def epsilon(self,w):
        w0,y,wp,f,eps_b=self.w0,self.y,self.wp,self.f,self.eps_b
        eps=eps_b*(1+wp**2*f/(w0**2-w**2-2j*y*w))
        return eps
        
    @staticmethod   
    def wp(N,meff,eps_b):
        """N (m**-3) charge density
        meff (fraction of m_e) effective mass
        eps_b (unitless) background dielectric"""
        return sqrt(N*q**2/(meff*m_e*eps0*eps_b))
    
class DrudeModel(MaterialLayer):
    """Simple model of a plasma"""
    def __init__(self,y,wp,f,eps_b,d,coh=True):
        """Everything should be in natural frequencies - w0 y wp
        f is the unitless oscillator strength
        eps_b is the background dielectric constant
        d is the layer thickness."""
        MaterialLayer.__init__(self,d,coh)
        self.y=y
        self.wp=wp
        self.f=f
        self.eps_b=eps_b
    
    def epsilon(self,w):
        y,wp,f,eps_b=self.y,self.wp,self.f,self.eps_b
        eps=eps_b*(1-wp**2*f/(w**2+2j*y*w))
        return eps
    
    @staticmethod    
    def wp(N,meff,eps_b):
        """N (m**-3) charge density
        meff (fraction of m_e) effective mass
        eps_b (unitless) background dielectric"""
        return sqrt(N*q**2/(meff*m_e*eps0*eps_b))
               
"""        
class metal(MaterialLayer):
    ""Simplified refractive index model of a metal at low frequencies.
    sigma0 (/Ohm/m) dc conductivity
    eps_b (unitless) background dielectric""
    def __init__(self,sigma0,eps_b,simple_n=True):
        self.sigma0=sigma0
        self.eps_b=eps_b
        
        if simple_n==True: #over-riding more exact calculation of refractive index
            def n(self):
                sigma0,eps_b=self.sigma0,self.eps_b
                p=sqrt(sigma0/(2.0*eps0*w))
                return (1+1j)*p
                
            self.n=n.__get__(self,metal) # binds function to instance
    
    def epsilon(self):
        sigma0,eps_b=self.sigma0,self.eps_b
        eps=eps_b+1j*sigma0/eps0/w
        return eps
"""

class Metal(MaterialLayer):
    """Simplified refractive index model of a metal at low frequencies.
    sigma0 (/Ohm/m) dc conductivity
    eps_b (unitless) background dielectric"""
    def __init__(self,sigma0,eps_b,d,coh=True,simple_n=True):
        """sigma0 - dc conductivity
        eps_b is the background dielectric constant
        d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
        self.sigma0=sigma0
        self.eps_b=eps_b        
        if simple_n==False: #to use more exact calculation of refractive index
            self.n=super(Metal,self).n # reverting to function from base class
            
    def epsilon(self,w):
        sigma0,eps_b=self.sigma0,self.eps_b
        eps=eps_b+1j*sigma0/eps0/w
        return eps
    
    def n(self,w):
        sigma0,eps_b=self.sigma0,self.eps_b
        p=sqrt(sigma0/(2.0*eps0*w))
        return (1+1j)*p
    
    @staticmethod    
    def wp(N,meff,eps_b):
        """N (m**-3) charge density
        meff (fraction of m_e) effective mass
        eps_b (unitless) background dielectric"""
        return sqrt(N*q**2/(meff*m_e*eps0*eps_b))

    @staticmethod    
    def sigma0(N,meff,y):
        """N (m**-3) charge density
        meff (fraction of m_e) effective mass
        eps_b (unitless) background dielectric"""
        return N*q**2/(meff*m_e*2*y)
        
    @staticmethod    
    def sigma0_b(wp,y,eps_b):
        """N (m**-3) charge density
        meff (fraction of m_e) effective mass
        eps_b (unitless) background dielectric"""
        return wp**2*eps0*eps_b/(2*y) 


class Metal2(MaterialLayer):
    """Simplified refractive index model of a metal at low frequencies.
    sigma0 (/Ohm/m) dc conductivity
    eps_b (unitless) background dielectric
    Actually, this is the Drude model reformulated."""
    def __init__(self,sigma0,y,eps_b,d,coh=True,simple_n=True):
        """sigma0 - dc conductivity
        y is the broadening and should be in natural frequency
        eps_b is the background dielectric constant
        d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
        self.sigma0=sigma0
        self.eps_b=eps_b
        self.y=y
        
    def epsilon(self,w):
        sigma0,y,eps_b=self.sigma0,self.y,self.eps_b
        sigma=sigma0*2*y/(2*y-1j*w)
        eps=eps_b+1j*sigma/eps0/w
        return eps
    
    @staticmethod    
    def wp(N,meff,eps_b):
        """N (m**-3) charge density
        meff (fraction of m_e) effective mass
        eps_b (unitless) background dielectric"""
        return sqrt(N*q**2/(meff*m_e*eps0*eps_b))

    @staticmethod    
    def sigma0(N,meff,y):
        """N (m**-3) charge density
        meff (fraction of m_e) effective mass
        eps_b (unitless) background dielectric"""
        return N*q**2/(meff*m_e*2*y)
        
    @staticmethod    
    def sigma0_b(wp,y,eps_b):
        """N (m**-3) charge density
        meff (fraction of m_e) effective mass
        eps_b (unitless) background dielectric"""
        return wp**2*eps0*eps_b/(2*y)

######################################

class Gold(MaterialLayer): #taken from paper by Etchegoin 2006
    def __init__(self,d,coh=True):
        """d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
        
    def epsilon(self,w):
        epsinf=1.53
        wp=12.9907004642E15 #Hz (natural)
        Gammap=110.803033371e12 #Hz (natural)
        C1=3.78340272066E15 #Hz (natural)
        w1=4.02489651134E15 #Hz (natural)
        Gamma1=0.818978942308E15 #Hz (natural)
        C2=7.73947471764E15 #Hz (natural)
        w2=5.69079023356E15 #Hz (natural)
        Gamma2=2.00388464607E15 #Hz (natural)
        sr2=sqrt(2)
        G1=C1*( (1-1j)/sr2/(w1 - w - 1j*Gamma1) + (1+1j)/sr2/(w1 + w + 1j*Gamma1) )
        G2=C2*( (1-1j)/sr2/(w2 - w - 1j*Gamma2) + (1+1j)/sr2/(w2 + w + 1j*Gamma2) )
        eps= epsinf - wp**2/(w**2+1j*w*Gammap) + G1 + G2
        
        return eps        
    
class Gold_Test(MaterialLayer): #an experiment to see difference of above model from plasma + 2 Lorentz oscillators
    def __init__(self,d,coh=True):
        """d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
    
    def epsilon(self,w):
        epsinf=1.53
        wp=12.9907004642E15 #Hz (natural)
        Gammap=110.803033371e12 #Hz (natural)
        C1=3.78340272066E15 #Hz (natural)
        w1=4.02489651134E15 #Hz (natural)
        Gamma1=0.818978942308E15 #Hz (natural)
        C2=7.73947471764E15 #Hz (natural)
        w2=5.69079023356E15 #Hz (natural)
        Gamma2=2.00388464607E15 #Hz (natural)
        #sr2=sqrt(2)
        G1=C1*( 1.0/(w1 - w - 1j*Gamma1) + 1.0/(w1 + w + 1j*Gamma1) )
        G2=C2*( 1.0/(w2 - w - 1j*Gamma2) + 1.0/(w2 + w + 1j*Gamma2) )
        eps= epsinf - wp**2/(w**2+1j*w*Gammap) + G1 + G2
        
        return eps
        


class Gold_THz(MaterialLayer):
    """Simple gold model for THz frequencies"""
    def __init__(self,d,coh=True):
        """d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
        #self.w=w #natural frequency
        
    def epsilon(self,w):
        w=1e-12*w
        eps= 1.0 - 0.6e8/(w * (w + 100.0j))
        return  eps    

######################################

class GaAs(MaterialLayer): #GaAs dielectric including phonon band.
    pass
    
class GaAs_THz(MaterialLayer): 
    """GaAs dielectric including phonon band for THz frequencies"""
    def __init__(self,n=0.0,d=None,coh=True):
        """n = doping (1E18 cm-3)
        d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
        #self.w=w #natural frequency
        self.doping=n
        
    def epsilon(self,w):
        w=1e-12*w
        eps= 10.4  + 5161.4 / (2620.0 - w**2 -0.2j*w)
        
        #including effect of doping
        n=self.doping
        if n!=0.0:
            if n<1.0: T=10.0/(1.0 - 2.0*N.log10(n))
            else: T=10.0
            eps-= 47436.84*n / (w*(w+1j*T))
        return  eps   

class GaAs_THz_C(MaterialLayer): 
    """GaAs dielectric including phonon band. Works for THz frequencies"""
    def __init__(self,n=0.0,d=None,coh=True):
        """n = doping (1E18 cm-3)
        d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
        #self.w=w #natural frequency
        self.doping=n
        
    def epsilon(self,w):
        w=1e-12*w
        eps= 10.88  + 5029.042 / (2552.81 - w**2 -0.377j*w)
        
        #including effect of doping
        n=self.doping
        if n!=0.0:
            if n<1.0: T=10.0/(1.0 - 2.0*N.log10(n))
            else: T=10.0
            eps-= 47436.84*n / (w*(w+1j*T))
        return  eps 

class AlGaAs(MaterialLayer): #AlGaAs dielectric including phonon band.
    pass

class QW_ISBT_unconventional(AnisoMaterialLayer): #includes wp, f12, damping, no depolarization shift, background dielectric
    """epsilon=eps_well+wp**2*f12/(w0**2-w**2-2j*y*w))
    wp**2=N*q**2/(meff*m_e*eps0*eps_well)
    w0 - frequency (?)
    y - scattering rate (?)
    wp - plasma frequency (?)
    eps_well - background dielectric constant (unitless)
    This class doesn't include the background dielectric constant within the 
    plasma frequency which allows more exactly for frequency dependence in the
    background dielectric constant but it's better normally to follow convention
    in order to avoid confusion.
    """
    def __init__(self,w0,y,f12,wp,eps_well,d,coh=True):
        """w0 -transition frequency (natural frequency)
        y - scattering rate (natural units)
        wp - plasma frequency (natural units)
        f12 is the unitless oscillator strength
        eps_well is the background dielectric constant
        d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
        self.w0=w0
        self.y=y
        self.f12=f12
        self.wp=wp
        self.eps_well=eps_well
        
    def epsilonzz(self,w):
        w0=self.w0; y=self.y; f12=self.f12; wp=self.wp; eps_well=self.eps_well
        eps=eps_well+wp**2*f12/(w0**2-w**2-2j*y*w)
        return eps
        
    def epsilon(self,w):
        return self.eps_well
    
    @staticmethod
    def wp(N,meff=0.067):
        """N  (cm**-3) 3D charge density
        eps_well (unitless) the background dielectric constant around the frequency of the transition
        meff (unitless - fraction of electron mass) effective mass of the electrons"""
        N=N*100**3 #converts density to m**-3
        return sqrt(N*q**2/(meff*m_e*eps0)) #doesn't include eps_well like other definitions
        
class QW_ISBT(AnisoMaterialLayer): #includes wp, f12, damping, no depolarization shift, background dielectric
    """epsilon=eps_well*(1.0+wp**2*f12/(w0**2-w**2-2j*y*w))
    wp**2=N*q**2/(meff*m_e*eps0*eps_well)
    w0 - frequency (?)
    y - scattering rate (?)
    wp - plasma frequency (?)
    eps_well - background dielectric constant (unitless)
    """
    def __init__(self,w0,y,f12,wp,eps_well,d,coh=True):
        """w0 -transition frequency (natural frequency)
        y - scattering rate (natural units)
        wp - plasma frequency (natural units)
        f12 is the unitless oscillator strength
        eps_well is the background dielectric constant
        d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
        self.w0=w0; self.y=y; 
        self.f12=f12; self.wp=wp; self.eps_well=eps_well
        
    def epsilonzz(self,w):
        w0=self.w0; y=self.y; f12=self.f12; wp=self.wp; eps_well=self.eps_well
        eps=eps_well*(1.0+wp**2*f12/(w0**2-w**2-2j*y*w))
        return eps
        
    def epsilon(self,w):
        return self.eps_well
    
    @staticmethod
    def wp(N,eps_well,meff=0.067):
        """N  (cm**-3) 3D charge density
        eps_well (unitless) the background dielectric constant around the frequency of the transition
        meff (unitless - fraction of electron mass) effective mass of the electrons"""
        N=N*100**3 #converts density to m**-3
        return sqrt(N*q**2/(meff*m_e*eps0*eps_well))
        
class QW_ISBT_gain(AnisoMaterialLayer): #includes wp, f12, damping, no depolarization shift, background dielectric
    """epsilon=eps_well*(1.0+wp**2*f12/(w0**2-w**2-2j*y*w))
    wp**2=N*q**2/(meff*m_e*eps0*eps_well)
    w0 - frequency (?)
    y - scattering rate (?)
    wp - plasma frequency (?)
    eps_well - background dielectric constant (unitless)
    """
    def __init__(self,w0,y,f12,wp,eps_well,d,coh=True):
        """w0 -transition frequency (natural frequency)
        y - scattering rate (natural units)
        wp - plasma frequency (natural units)
        f12 is the unitless oscillator strength
        eps_well is the background dielectric constant
        d is the layer thickness, coh is the layer coherence"""
        MaterialLayer.__init__(self,d,coh)
        self.w0=w0; self.y=y; 
        self.f12=f12; self.wp=wp; self.eps_well=eps_well
        
    def epsilonzz(self,w):
        w0=self.w0; y=self.y; f12=self.f12; wp=self.wp; eps_well=self.eps_well
        eps=eps_well*(1.0+wp**2*f12/(w0**2-w**2-2j*y*w))
        return N.conjugate(eps)
        
    def epsilon(self,w):
        return self.eps_well
    
    @staticmethod
    def wp(N,eps_well,meff=0.067):
        """N  (cm**-3) 3D charge density
        eps_well (unitless) the background dielectric constant around the frequency of the transition
        meff (unitless - fraction of electron mass) effective mass of the electrons"""
        N=N*100**3 #converts density to m**-3
        return sqrt(N*q**2/(meff*m_e*eps0*eps_well))
        
        
if __name__=="__main__":
    
    import pylab as pl
    pl.figure(1)
    ax1=pl.subplot(311)
    ax2=pl.subplot(312, sharex=ax1)
    ax3=pl.subplot(313,sharex=ax1)
    w=pl.arange(0,5e12,5e9)
    L=LorentzModel(w0=1e12,y=5e10,wp=8e11,f=0.96,eps_b=1.0,d=None)
    ax1.plot(w,L.epsilon(w).real,label="epsilon real")
    ax1.plot(w,L.epsilon(w).imag,label="epilon imaginary")
    ax2.plot(w,L.n(w).real,label="refractive index")
    ax2.plot(w,L.n(w).imag,label="kappa")
    ax3.plot(w,2*w*L.n(w).imag/c,label="absorption coefficient")
    for ax in ax1,ax2,ax3:
        ax.axvline(L.w0) #w0
        ax.axvline(sqrt(L.w0**2-L.y**2)) #damping shifted peak
        #We don't see a depolarisation shift in this geometry so no point confusing you...yet.
        #ax.axvline(sqrt(L.w0**2+L.wp**2*L.f)) #depolarisation shifted peak #2
        #ax.axvline(sqrt(L.w0**2+L.wp**2)) #depolarisation shifted peak
        #ax.axvline(sqrt(L.w0**2-L.y**2+L.wp**2*L.f)) #depolarisation + damping shifted peak #2 
        #ax.axvline(sqrt(L.w0**2-L.y**2+L.wp**2)) #depolarisation + dampiing shifted peak
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title("Various properties of an example Lorentzian Oscillator")
    ax3.set_xlabel("Frequency (real) (Hz)")
    ax3.text(1.4e12,4000,"It is interesting that the absorption coefficient plotted here is \n \
nothing like the profile, we would see if we modelled the \n \
absorption of a slab of this material"  )
    #
    pl.figure(2)
    ax1=pl.subplot(311)
    ax2=pl.subplot(312, sharex=ax1)
    ax3=pl.subplot(313,sharex=ax1)
    w=pl.arange(0,5e12,5e9)
    D=DrudeModel(y=5e10,wp=8e11,f=0.96,eps_b=1.0,d=None)
    ax1.plot(w,D.epsilon(w).real,label="epsilon real")
    ax1.plot(w,D.epsilon(w).imag,label="epilon imaginary")
    ax2.plot(w,D.n(w).real,label="refractive index")
    ax2.plot(w,D.n(w).imag,label="kappa")
    ax3.plot(w,2*w*D.n(w).imag/c,label="absorption")
    for ax in ax1,ax2,ax3:
        ax.axvline(D.wp) #w0
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title("Various properties of an example Drude model")
    ax3.set_xlabel("Frequency (real) (Hz)")
    ax3.text(1.4e12,2000,"It is interesting that the absorption coefficient plotted here is \n \
nothing like the profile, we would see if we modelled the \n \
absorption of a slab of this material"  )    
    #
    pl.figure(3)
    ax1=pl.subplot(311)
    ax2=pl.subplot(312, sharex=ax1)
    ax3=pl.subplot(313,sharex=ax1)
    w=pl.arange(5e9,5e12,5e9)
    M=Metal(sigma0=45.2e6,eps_b=1.0,d=None) #gold
    ax1.plot(w,M.epsilon(w).real,label="epsilon real")
    ax1.plot(w,M.epsilon(w).imag,label="epilon imaginary")
    ax2.plot(w,M.n(w).real,label="refractive index")
    ax2.plot(w,M.n(w).imag,label="kappa")
    ax3.plot(w,2*w*M.n(w).imag/c,label="absorption")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title("Various properties of an example of a metallic material")
    ax3.set_xlabel("Frequency (real) (Hz)")
    ax3.text(1.4e12,15e6,"It is interesting that the absorption coefficient plotted here is \n \
nothing like the profile, we would see if we modelled the \n \
absorption of a slab of this material"  )  
    # Gold at optical frequencies
    pl.figure(4)
    ax1=pl.subplot(211)
    ax2=pl.subplot(212, sharex=ax1)
    f=pl.arange(300e12,1500e12,5e10)
    w=2*pi*f
    G=Gold(d=None) #
    G2=Gold_Test(d=None)
    for g in (G,G2):#,(G2):
        ax1.plot(c/f*1e9,g.epsilon(w).real,label="epsilon real")
        ax1.plot(c/f*1e9,g.epsilon(w).imag,label="epilon imaginary")
        ax2.plot(c/f*1e9,g.n(w).real,label="refractive index")
        ax2.plot(c/f*1e9,g.n(w).imag,label="kappa")
    ax1.legend()
    ax2.legend()
    ax1.set_title("Gold at optical frequencies")
    ax2.set_xlabel("Wavelength (nm)")
    
    #GaAs refractive index
    pl.figure(5)
    ax1=pl.subplot(111)
    f=pl.arange(300e9,20e12,100e9)
    w=2*pi*f
    GaAs=GaAs_THz(d=None)
    ax1.plot(f,GaAs.n(w).real,label="real part")
    ax1.plot(f,GaAs.n(w).imag,label="imag part")
    ax1.legend()
    ax1.set_title("Refractive index of GaAs for THz frequencies")
    ax1.set_xlabel("Frequency (real) (Hz)")
    ax1.text(9e12,10,"LO phonon interaction (polariton)")
    
    """
    #SOPRA refractive index
    pl.figure(6)
    ax1=pl.subplot(111)
    wavelength=pl.linspace(240,830,200) #nm
    w=c*2*pi*1e9/wavelength #frequency (natural)
    GaAs111=SopraLayer("GAAS111",d=None,coh=True)
    ax1.plot(wavelength,GaAs111.n(w).real,label="real part")
    ax1.plot(wavelength,GaAs111.n(w).imag,label="imag part")
    ax1.legend()
    ax1.set_title("Refractive index of GaAs 111K from SOPRA data")
    ax1.set_xlabel("Wavelength (nm)")
    """
    pl.show()
