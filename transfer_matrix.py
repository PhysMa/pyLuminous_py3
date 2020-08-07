#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of pyFresnel.
# Copyright (c) 2012-2016, Robert Steed
# Author: Robert Steed (rjsteed@talk21.com)
# License: GPL
# last modified 15.08.2016
"""A simple Optical Transfer Matrix code (there are so many out there).
It takes a description of the layers and calculates the transmission and reflection.
I will include a very special anisotropic/uniaxial case where the dielectric is
different along the perpendicular/ layer stack axis than to the in-plane directions;
this is so I can describe quantum well intersubband absorptions. If I have time,
I will include code to calculate the intensity for each layer (for modelling
saturation of structures)."""

"""
##Summary of maths


##Implementation
#Transfer matrix involves 2x2 matrices and multiplication but this needs to be
#done for each frequency/wavelength.

#numpy method 1
#A = N.empty((numpts,),dtype=object) #this creates an array of objects that can be filled with anything, including matrices

#numpy method2
#A = N.zeros((numpts,2,2)) #creates a 3d array
#Trick from http://jameshensman.wordpress.com/2010/06/14/multiple-matrix-multiplication-in-numpy/
#C = N.sum(np.transpose(A,(0,2,1)).reshape(100,2,2,1)*B.reshape(100,2,1,2),-3)#this can be used to matrix multiply 2 such arrays (very ugly isn't it)
#C = N.dot(A,B) #does this work? not quite.
#use
#N.diagonal(N.dot(A,B),axis1=0,axis2=2).swapawes(1,2).swapaxes(0,1)
#or
#N.rollaxis(N.diagonal(N.dot(A,B),axis1=0,axis2=2),2)
#or
#N.transpose(N.diagonal(N.dot(A,B),axis1=0,axis2=2),(2,0,1))
#but unecessary calculations are carried out by N.dot() so it is better to do
# N.sum(N.transpose(A,(0,2,1))[:,:,:,N.newaxis]*B[:,:,N.newaxis,:],axis=-3)

# see also einsum() and tensordot()

##Describing layers
#Can describe system as
#
#f1 = filter([Layer(eps0,d0),
#            Layer(eps1,d1),
#            Layer(eps3,d2),
#            ...
#            ],w,pol,theta)
# but can use any python list operation required to make the process more efficient, i.e. f1*3 will repeat the structure 3 times.
# F1.append(Layer(epsn,dn)) will append a layer.
"""

import numpy as N
from numpy.random import random_sample
print ("numpy imported")
from numbers import Number
import copy
c=299792458  #m/s - speed of light
pi=N.pi

def param_check(param,w):
    if hasattr(param,'n'): #this case is so that we are able to use the material classes defined in materials.py.
        param=param.n()
    if hasattr(param,'__call__'):
        x = param(w)
    elif hasattr(param,'__len__'): #Is n an array with the same length as w?
        if len(param)!=len(w):
            raise NameError("w and parameter array are not compatible")
        else:
            x = param
    elif isinstance(param,Number): #Is self.eps a number (integer/float/complex)
        x = N.repeat(param,len(w))
    else:
        raise TypeError("Don't know how to handle this input")
    return x

#add code to deal with classes from materials.py

class Layer():
    def __init__(self,n,d,coh=True):
        self._n = n # can be a function or array (if it has the same length as the spectral axis)
        self.d = d # thickness of the layer (m)
        self.coh = coh # is layer coherent or incoherent, could also be a number between 0 and pi to decribe a partially coherent layer

    def n(self,w):
        return param_check(self._n,w)

    def __repr__(self):
        return "Layer"+"("+repr(self._n)+", "+repr(self.d)+", coh="+repr(self.coh)+" )"

class LayerUniaxial():
    def __init__(self,nxx,nzz,d,coh=True):
        self._nxx = nxx # can be a function or array (if it has the same length as the spectral axis)
        self._nzz = nzz # can be a function or array (if it has the same length as the spectral axis)
        self.d = d
        self.coh = coh

    def n(self,w):
        return param_check(self._nxx,w)

    def nzz(self,w):
        return param_check(self._nzz,w)

    def __repr__(self):
        return "LayerUniaxial"+"("+repr(self._nxx)+", "+repr(self._nzz)+", "+repr(self.d)+", coh="+repr(self.coh)+" )"

class Layer_eps():
    def __init__(self,eps,d,coh=True):
        self.eps = eps # can be a function or array (if it has the same length as the spectral axis)
        self.d = d
        self.coh = coh

    def n(self,w):
        return N.sqrt(param_check(self.eps,w))

    def __repr__(self):
        return "Layer_eps"+"("+repr(self.eps)+", "+repr(self.d)+", coh="+repr(self.coh)+" )"

class LayerUniaxial_eps():
    def __init__(self,epsxx,epszz,d,coh=True):
        self.epsxx = epsxx # can be a function or array (if it has the same length as the spectral axis)
        self.epszz = epszz # can be a function or array (if it has the same length as the spectral axis)
        self.d = d
        self.coh = coh

    def n(self,w):
        return N.sqrt(param_check(self.epsxx,w))

    def nzz(self,w):
        return N.sqrt(param_check(self.epszz,w))

    def __repr__(self):
        return "LayerUniaxial_eps"+"("+repr(self.epsxx)+", "+repr(self.epszz)+", "+repr(self.d)+", coh="+repr(self.coh)+" )"



def interactive(func):
    """This allows us to change the objects parameters temporarily during a calculation
    which is more convenient when calling the class methods. However, it will only
    work if the function does not require an input parameter with the same name as
    an object data attribute"""
    def wrapping(self,*args,**kwargs):
        #I'm using self.__dict__ to be general but I could always use a custom defined dictionary/list like ['pol','w','angles'].
        kwargsA = {k:kwargs[k] for k in kwargs if k not in self.__dict__}
        kwargsB = {k:kwargs[k] for k in kwargs if k in self.__dict__}
        SaveState = copy.copy(self.__dict__)
        try:
            for k in kwargsB:
                print(('Temporarily changing %s for this calculation' %k))
                setattr(self,k,kwargsB[k])
            self._checks_n_axis()
            self.n0sinth2=(self[0].n(self.w) * N.sin(self.theta))**2
            return func(self,*args,**kwargsA)
        finally:
            self.__dict__=SaveState
    return wrapping



class Filter_base(list):
    def __init__(self,fltrlst,w=None,pol='TE',theta=0.0,*args,**kwargs):
        """Object to describe a stack of dielectric layers. fltrlist is a list of
        Layer objects (any object that has a method n(w) for the refractive index and
        and a property d for the layer thickness will be suitable). The order of the
        list is from front to back. w is the frequency (natural), pol is the
        polarisation and should be either 'TE' or 'TM'. theta is the angle of
        incidence (radians)."""
        list.__init__(self,fltrlst,*args,**kwargs)
        self.w=w
        self.pol=pol
        self.theta=theta
        self._checks_n_axis() # makes sure that w and theta are numpy arrays
        self.n0sinth2=(fltrlst[0].n(self.w) * N.sin(self.theta))**2

    def _checks_n_axis(self):
        w=self.w
        theta=self.theta
        # Some checks
        if w is None:
            w=N.array([1.0])
        elif isinstance(w,Number):
            w=N.array([w])
        elif hasattr(w,'__iter__'):
            w=N.array(w)
        self.w=w
        # Some checks
        if isinstance(theta,Number):
            theta=N.array([theta])
        elif hasattr(theta,'__iter__'):
            theta=N.array(theta)
        self.theta=theta
        #decide whether axis is frequency or angle of incidence
        if hasattr(theta,'__iter__') and len(theta)>1:
            if len(theta)==len(w):
                self.axis=w
                print ('Exceptionally both angle and frequency are arrays with the same length')
                print ('but setting self.axis to frequency')
            else:
                assert len(w)==1, "Can not vary both frequency and angle at the same time"
                self.axis=theta
        else:
            self.axis=w

    def __repr__(self):
        return repr(self[:]) + ", w= "+repr(self.w)+", pol= "+repr(self.pol)+", theta (radians) = "+repr(self.theta)+", theta (degrees) = "+repr(self.theta*180/pi)

    def __str__(self):
        globals = "w : %s\npol : %s\ntheta : %s theta(deg): %s" %(repr(self.w),repr(self.pol),repr(self.theta),repr(self.theta*180/pi))
        stack = "[\n"+",\n".join([repr(l) for l in self ])+"\n]"
        return globals + '\n' +stack

    def _lambda(self,layer_pair):
        """Calculates a variable needed to create the interface matrices"""
        # Special code to account for a special case of anisotropic medium
        if hasattr(layer_pair[1],'nzz') and self.pol=='TM':
            n1 = layer_pair[1].nzz
        else:
            n1 = layer_pair[1].n

        if hasattr(layer_pair[0],'nzz') and self.pol=='TM':
            n0 = layer_pair[0].nzz
        else:
            n0 = layer_pair[0].n

        w = self.w
        cos_ratio = N.sqrt(1+0j - self.n0sinth2 / n1(w)**2 ) # uses 1+0j to cast argument to complex so that we can describe totoal internal reflection.
        cos_ratio = cos_ratio/N.sqrt(1+0j - self.n0sinth2 / n0(w)**2 )
        n_ratio = layer_pair[1].n(w) / layer_pair[0].n(w)

        if self.pol=='TE':
            lmda = cos_ratio * n_ratio
        elif self.pol=='TM':
            lmda = cos_ratio / n_ratio
        else:
            raise NameError("_lambda(): pol should be 'TE' or 'TM'")
        return lmda

    def _k_z(self,layer):
        """z component of k vector for a layer. Returns a 1d array of phase vs frequency."""
        #special code to account for a special case of anisotropic medium
        if hasattr(layer,'nzz') and self.pol=='TM':
            n1 = layer.nzz
        else:
            n1 = layer.n

        w = self.w
        k = layer.n(w) * w / c
        costh = N.sqrt(1+0j - self.n0sinth2 / n1(w)**2 )

        return k * costh

    def _phase(self,layer):
        """Phase change for a layer. Returns a 1d array of phase vs frequency."""
        return self._k_z(layer)* layer.d

    def _mat_mult(self,A,B):
        """Multiplies two arrays of 2x2 matrices together using matrix multiplication.
        A & B are arrays nx2x2 where n is the frequency axis of the calculation"""
        #Trick from http://jameshensman.wordpress.com/2010/06/14/multiple-matrix-multiplication-in-numpy/
        return  N.sum(N.transpose(A,(0,2,1))[:,:,:,N.newaxis]*B[:,:,N.newaxis,:],axis=-3)



class Filter(Filter_base):
    def __init__(self,fltrlst,w=None,pol='TE',theta=0.0,*args,**kwargs):
        """Object to describe a stack of dielectric layers. fltrlist is a list of
        Layer objects (any object that has a method n(w) for the refractive index and
        and a property d for the layer thickness will be suitable). The order of the
        list is from front to back. w is the frequency (natural), pol is the
        polarisation and should be either 'TE' or 'TM'. theta is the angle of
        incidence (radians)."""
        Filter_base.__init__(self,fltrlst,w=w,pol=pol,theta=theta,*args,**kwargs)

    def _layer_array(self,layer):
        """array of 'matrices' describing the phase change across the layer wrt frequency"""
        phase=self._k_z(layer)*layer.d
        if hasattr(layer,'coh') and type(layer.coh)==float:
            coh=layer.coh
            nph=N.exp(-1j*(phase + coh*(2*random_sample(self.axis.shape)-1) ))
            pph=N.exp( 1j*(phase + coh*(2*random_sample(self.axis.shape)-1) ))
        else:
            nph=N.exp(-1j*phase)
            pph=N.exp(1j*phase)
        matrix = N.column_stack((nph,N.zeros_like(self.axis),N.zeros_like(self.axis),pph))
        matrix.shape = (len(self.axis),2,2)
        return matrix

    def _interface_array(self,layer_pair):
        """array of 'matrices' describing the interface between 2 layers wrt frequency"""
        lmda=self._lambda(layer_pair)
        matrix = 0.5*N.column_stack(((1+lmda),(1-lmda),(1-lmda),(1+lmda)))
        matrix.shape = (len(self.axis),2,2)
        return matrix

    def _layer_array2(self,layer_pair):
        """array of 'matrices' describing the interface and also phase change for the layer wrt frequency"""
        phase=self._k_z(layer_pair[1])*layer_pair[1].d
        if hasattr(layer_pair[1],'coh') and type(layer_pair[1].coh)==float:
            coh=layer_pair[1].coh
            nph=N.exp(-1j*(phase + coh*(2*random_sample(self.axis.shape)-1) ))
            pph=N.exp( 1j*(phase + coh*(2*random_sample(self.axis.shape)-1) ))
        else:
            nph=N.exp(-1j*phase)
            pph=N.exp(1j*phase)
        lmda=self._lambda(layer_pair)
        matrix = 0.5*N.column_stack(((1+lmda)*nph,(1-lmda)*pph,(1-lmda)*nph,(1+lmda)*pph))
        matrix.shape = (len(self.axis),2,2)
        return matrix

    @interactive
    def _calculate_M(self):
        """Calculates the system matrix"""
        I = N.array(((1.0,0.0),(0.0,1.0)))
        tmp = N.array( (I,)*len(self.axis) ) # running variable to hold the calculation results. Starts as an array of identity 'matrices'
        ziplist = list(zip(self,self[1:]))
        for layer_pair in ziplist[:-1]:
            nl = self._layer_array2(layer_pair)
            tmp = self._mat_mult(tmp,nl)
        #final interface before substrate/end of filter
        layer_pair=(self[-2],self[-1])
        tmp = self._mat_mult(tmp,self._interface_array(layer_pair))
        return tmp

    @interactive
    def calculate_r_t(self):
        """Calculates the reflection and transmission coefficients"""
        axis = self.axis
        w = self.w
        M = self._calculate_M()
        t = 1.0/M[:,0,0]
        if self.pol=='TM': #extra polarisation sensitive term for t
            t*=self[0].n(w)/self[-1].n(w)
        r = M[:,1,0]/M[:,0,0]
        return (axis,r,t)

    @interactive
    def calculate_pol_phaseshift(self,deg=False):
        axis,rs,ts =self.calculate_r_t(pol='TE')
        axis,rp,tp =self.calculate_r_t(pol='TM')
        r_phase=N.angle(rs,deg) - N.angle(rp,deg)
        t_phase=N.angle(ts,deg) - N.angle(tp,deg)
        return (axis,r_phase,t_phase)

    @interactive
    def calculate_R_T(self):
        """Calculates the Reflectivity and Transmission of the stack"""
        axis = self.axis
        M = self._calculate_M()
        t = 1.0/M[:,0,0]
        T = t*t.conjugate()*self._lambda((self[0],self[-1]))
        r = M[:,1,0]/M[:,0,0]
        R = r*r.conjugate()
        return (axis,R.real,T.real)

    def _mat_vec_mult(self,A,v):
        """Multiplies an array of 2x2 matrices to an array of 2x1 vectors using matrix multiplication.
        A is an arrays nx2x2 where n is the frequency axis of the calculation.
        v is a vector nx2x1."""
        return  N.sum(A*v[:,N.newaxis,:],axis=-1)

    def _calculate_UV(self):
        """In the transfer matrix model, each layer's electric field can be described
        as the summation of two waves, one travelling in each direction. I call the
        waves U & V in my derivation, hence the method name. This method calculates
        the amplitudes of these waves for each layer. However, to calculate the total
        electric field within the layer, we would need to take account of the wave's
        directions as well."""
        w=self.w
        #It's quite easy to get the wave amplitudes, we just need to cache the results
        #of our matrix calculation at each step and then normalise the results at the end.
        #Compared to calculate_M(), we have to do the calculation backwards though.

        # The final state vector is given by one wave travelling away from the system.
        Uf = N.array((1,0)) #unnormalised solution.
        Ufw = N.array( (Uf,)*len(self.axis) ) # An array of vectors to describe different frequencies.

        # Treat last interface separately.
        last_interface = self._interface_array((self[-2],self[-1])) #remember that this class is derived from a list
        current_UV = self._mat_vec_mult(last_interface,Ufw)
        if self.pol=='TM': current_UV*=(self[-1].n(w)/self[-2].n(w))[:,N.newaxis] #Most transfer matrix calculation's
        #drop this term and shift its effect into the calculation of the transmissivity but here we have to remember it.

        allUV= [Ufw,current_UV] #list to hold results.

        # Main Loop
        reverse_layer_order = self[-2::-1] # we are doing are calculation in reverse.
        for layer_pair in list(zip(reverse_layer_order[1:],reverse_layer_order)):
            layer_matrix = self._layer_array2(layer_pair)
            current_UV = self._mat_vec_mult(layer_matrix,current_UV)
            if self.pol=='TM': current_UV*=(layer_pair[1].n(w)/layer_pair[0].n(w))[:,N.newaxis]
            allUV.append(current_UV)

        # Checking final matrix against normal calculation
        M = self._calculate_M()
        S = self._mat_vec_mult(M,Ufw)
        if self.pol=='TM': S*=(self[-1].n(w)/self[0].n(w))[:,N.newaxis]
        assert N.allclose(current_UV,S)

        # Normalise Results
        A = current_UV[:,0]
        allUVnormed = [uv/A[:,N.newaxis] for uv in allUV]
        return allUVnormed

    def _Eangle(self,layer):
        """Calculates the angle of the Efield within each layer"""
        w = self.w
        if self.pol=='TE':
            angles = 0.0+0.0j
        elif hasattr(layer,'nzz') and self.pol=='TM': #because the Evector is not perpendicular to the k-vector
            nzz = layer.nzz
            angles = N.arcsin(N.sqrt(self.n0sinth2)/nzz(w)+0j) #allow for complex angles with 0j
            #there will be an extra factor for Ez of n_xx/n_zz that I will have to fit in somewhere.
        elif self.pol=='TM':
            n1 = layer.n
            angles = N.arcsin(N.sqrt(self.n0sinth2)/n1(w)+0j) #allow for complex angles with 0j
        else:
            raise NameError("pol should be 'TE' or 'TM'")

        return angles

    @interactive
    def calculate_E(self,z):
        """Calculate the electric field within the thin film filter.
        z is the position vector, the origin is located at the end of the filter."""
        # As we are definining the origin at the end of the filter, we need to reverse things
        flist=list(reversed(self))
        # In this calculation, the 0th layer is the final semi-infinite medium/substrate.
        # Likewise the last layer is the semi-infinite initial medium of the problem.

        # Wave amplitudes for each layer.
        allUV_list = self._calculate_UV() #nb. already in reverse order

        # k_z and angle of Electric field to stack direction
        k_z_list = [self._k_z(layer) for layer in flist]
        angles_list = [self._Eangle(layer) for layer in flist]

        # Calculate Electric Field

        #using numpy arrays in order to do some indexing tricks
        allUV = N.array(allUV_list)  # allUV[layer no.,w,U/V] i.e. U = allUV[:,:,0],V = allUV[:,:,1]
        k_z = N.array(k_z_list)
        angles = N.array(angles_list)

        #find layer number and relative position within layer for each z
        #This is the thickness of each layer (the 0th layer is in fact the final semi-infinte medium and here is assigned a zero thickness).
        thicknesses=[0.0]+[layer.d for layer in flist if layer.d!=None]#+[0.0]
        interfaces=N.cumsum(thicknesses) # the positions of the interfaces.
        zlayers = N.searchsorted(interfaces,z) #the layer number for each z position,
        #however, the distance is defined from the back of the filter stack and the layer numbers are reversed relative to 'normal' but not compared to everything else in this calculation.

        #find relative positions within layers.
        #Since the final(0th) layer is normally infinite, it is defined differently from the others and so we need to treat it differently
        offsets = zlayers.copy()
        offsets[zlayers!=0]-=1
        dz = z - interfaces[offsets]

        #
        Uz = allUV[zlayers,:,0]*N.exp(-1j*k_z[zlayers,:]*dz[:,N.newaxis])
        Vz = allUV[zlayers,:,1]*N.exp(1j*k_z[zlayers,:]*dz[:,N.newaxis])

        if self.pol=='TE':
            Ey = Uz+Vz
            return {'Ey':Ey,'Ez':0.0,'Ex':0,'Dz':0.0}

        elif self.pol=='TM':
            epszz = N.array([getattr(layer,'nzz',layer.n)(self.w)**2 for layer in flist])
            epszzz = epszz[zlayers]

            nzz = N.array([getattr(layer,'nzz',layer.n)(self.w) for layer in flist])
            nzzz = nzz[zlayers]
            nxx = N.array([layer.n(self.w) for layer in flist])
            nxxz = nxx[zlayers]
            ratio=nxxz/nzzz # this ratio is needed to get the Ez field correct within the uniaxial layers.

            anglesz = angles[zlayers]
            Ez = (Uz+Vz)*N.sin(anglesz)*ratio
            Ex = (Uz-Vz)*N.cos(anglesz)
            Dz = epszzz*Ez
            return {'Ey':0.0,'Ez':Ez,'Ex':Ex,'Dz':Dz}

        else:
            raise NameError("calculate_E(): pol should be 'TE' or 'TM'")

    @interactive
    def calculate_Abs(self,z):
        mod2=lambda a : a*a.conj() #modulus squared

        Efields=self.calculate_E(z)
        if self.pol=='TM':
            Fz =mod2(Efields['Ez']) + mod2(Efields['Ex'])
        elif self.pol=='TE':
            Fz = mod2(Efields['Ey'])
        else:
            raise Exception("Absorption(): pol should be 'TE' or 'TM'")
        #
        # As we are definining the origin at the end of the filter, we need to reverse things
        flist=list(reversed(self))
        # In this calculation, the 0th layer is the final semi-infinite medium/substrate.
        # Likewise the last layer is the semi-infinite initial medium of the problem.
        layer0=self[0]
        #However layer0 will be the initial layer of the problem where the light starts from.

        tmp=self.pol
        self.pol='TE' #need this so _lambda() calculates the correct form of the obliqueness factor
        lmdas=[self._lambda((layer0,layer)) for layer in flist]
        self.pol=tmp  #return polarisation to previous state.
        #
        kzs = [self._k_z(layer) for layer in flist]
        #
        qs = [lmda.real*2*kz.imag for lmda,kz in list(zip(lmdas,kzs))]

        #using numpy arrays in order to do some indexing tricks
        qs = N.array(qs)
        #find layer number and relative position within layer for each z
        #This is the thickness of each layer (the 0th layer is in fact the final semi-infinte medium and here is assigned a zero thickness).
        thicknesses=[0.0]+[layer.d for layer in flist if layer.d!=None]#+[0.0]
        interfaces=N.cumsum(thicknesses) # the positions of the interfaces.
        zlayers = N.searchsorted(interfaces,z) #the layer number for each z position,

        qsz=qs[zlayers,:] #change q for each layer into q relative to z array.

        dA = qsz * Fz #Absorption per dz.
        return dA



if __name__ == "__main__":
    pi=N.pi
    f=N.linspace(1.0e10,10e12,200)

    ########Define interference filter##########
    f1 = Filter([Layer_eps(1.0,None),
            Layer_eps(3.50+0.5j,8.6e-6),
            Layer_eps(12.25,None)],
            w=2*pi*f,
            pol='TM',
            theta=pi/4)

    #print f1
    #print repr(f1)
    #print len(f1)
    #print f1.calculate_M()[0]
    #print f1._lambda((f1[0],f1[-1]))
    #print f1.calculate_r_t()[0]

    ########Calculate Reflection/Transmission###
    w,R,T=f1.calculate_R_T(pol='TM')
    w,R2,T2=f1.calculate_R_T(pol='TE')

    import matplotlib.pyplot as P

    ax1 = P.subplot(111)
    ax1.plot(f,R,label="TM Reflection")
    ax1.plot(f,T,label="TM Transmission")
    ax1.plot(f,R2,label="TE Reflection")
    ax1.plot(f,T2,label="TE Transmission")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_title("Antireflection coating for GaAS or Silicon")
    ax1.legend()

    ########Calculate E field###################
    mod2=lambda a : a*a.conj() #modulus squared

    z = N.linspace(-30.0e-6,60e-6,400) #(m)
    zaxis = z*1e6 #(um)
    fi = 100 #choose index for frequency (this is equiv. to 5.5THz for this script).

    Efields=f1.calculate_E(z=z,pol='TM')

    Efields2=f1.calculate_E(z,pol='TE')

    Absorption=f1.calculate_Abs(z,pol='TM')

    Absorption2=f1.calculate_Abs(z,pol='TE')

    #check TM transmission coefficient
    f1.pol='TE' #this is so we calculate the n2cos(theta2)/(n1cos(theta1)) factor (rather than the n1cos(theta2)/(n2cos(theta1))
    print(('TM Transmission at %g THz = ' %(f[fi]*1e-12), f1._lambda((f1[0],f1[-1]))[fi]*(mod2(Efields['Ez'][0,fi])+mod2(Efields['Ex'][0,fi]))))

    #check TE transmission coefficient
    print(('TE Transmission at %g THz = ' %(f[fi]*1e-12), f1._lambda((f1[0],f1[-1]))[fi]*mod2(Efields2['Ey'][0,fi])))

    #######

    f2 = P.figure(2)
    ax2 = P.subplot(111)
    ax2.plot(zaxis,mod2(Efields['Ex'][:,fi]),label="|Ex|**2")
    ax2.plot(zaxis,mod2(Efields['Ez'][:,fi]),label="|Ez|**2")
    ax2.plot(zaxis,mod2(Efields['Dz'][:,fi]),label="|Dz|**2")
    ax2.plot(zaxis,mod2(Efields2['Ey'][:,fi]),label="|Ey|**2")
    ax3 = P.twinx(ax2)
    ax3.plot(zaxis,Absorption[:,fi],label="Absorption (TM)")
    ax3.plot(zaxis,Absorption2[:,fi],label="Absorption (TE)")
    ax2.set_xlabel("distance (um)")
    ax2.set_ylabel("|E|**2")
    ax2.legend()
    ax3.legend(loc=4)

    P.show()
