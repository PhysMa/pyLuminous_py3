#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of pyFresnel.
# Copyright (c) 2012-2016, Robert Steed
# Author: Robert Steed (rjsteed@talk21.com)
# License: GPL
# last modified 15.08.2016
"""Incoherent Optical Transfer Matrix code. 
It takes a description of the layers and calculates the transmission and reflection. 
I will include a very special anisotropic/uniaxial case where the dielectric is 
different along the perpendicular/ layer stack axis than to the in-plane directions; 
this is so I can describe quantum well intersubband absorptions. If I have time, 
I will include code to calculate the intensity for each layer (for modelling
saturation of structures)."""

import numpy as N
print("numpy imported")
from numbers import Number
import copy
c=299792458  #m/s - speed of light
pi=N.pi

from .transfer_matrix import Filter_base,Filter,interactive,Layer_eps,Layer

class IncoherentFilter(Filter_base):
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
        alphad=2*self._k_z(layer).imag*layer.d
        nph=N.exp(-alphad)
        pph=N.exp( alphad)
        matrix = N.column_stack((pph,N.zeros_like(self.axis),N.zeros_like(self.axis),nph))
        matrix.shape = (len(self.axis),2,2)
        return matrix
        
    def _interface_array(self,layer_pair):
        """array of 'matrices' describing the interface between 2 layers wrt frequency"""
        mod2=lambda a : a*a.conj() #modulus squared
        lmda=self._lambda(layer_pair)
        matrix = 0.25*N.column_stack(( mod2(1+lmda),-mod2(1-lmda),mod2(1-lmda),mod2(1+lmda) - 2*mod2(1-lmda))) 
        matrix.shape = (len(self.axis),2,2)
        matrix/= self._lambda(layer_pair).real[:,N.newaxis,N.newaxis] #here is a factor of 1/lmbda which I'm including because I no longer think that the term cancels out between layers when there are absorbing layers.
        return matrix
    """    
    def _layer_array2(self,layer_pair):
        ""array of 'matrices' describing the interface and also phase change for the layer wrt frequency""
        mod2=lambda a : a*a.conj() #modulus squared
        alphad=2*self._k_z(layer_pair[1]).imag*layer_pair[1].d
        nph=N.exp(-alphad)
        pph=N.exp( alphad)
        lmda=self._lambda(layer_pair)
        matrix = 0.25*N.column_stack(( mod2(1+lmda)*nph,-mod2(1-lmda)*pph,mod2(1-lmda)*nph, (16*mod2(lmda) - mod2(1-lmda)**2)/mod2(1+lmda)*pph )) #mod2(l+lmda) - 2*mod2(1-lmda)*pph
        #There should be a factor of 1/lmda but these cancel out except for the initial and last layer.
        matrix.shape = (len(self.axis),2,2)
        return matrix
    """    
    def _thin_film(self,filter):
        """Calculates the thin film's matrix for inclusion in an incoherent calculation"""
        mod2=lambda a : a*a.conj() #modulus squared
        tm=filter._calculate_M() #transfer matrix for coherent thin film system
        obliquenessfac = filter._lambda((filter[0],filter[-1])).real
        incoherent_matrix = mod2(tm)
        incoherent_matrix[:,0,1]*= -1
        incoherent_matrix[:,1,1] = (obliquenessfac**2 - mod2(tm[:,0,1]*tm[:,1,0]).real)/mod2(tm[:,0,0])
        incoherent_matrix/= obliquenessfac[:,N.newaxis,N.newaxis] #here is a factor of 1/lmbda which I'm including because I'm not sure that the terms cancel out for absorbing layers.
        return incoherent_matrix 

    @interactive
    def _calculate_M(self):
        """Calculates the incoherent system matrix."""
        #create a new list, with single layer entries to indicate layer matrices, pair enties to indicate normal interfaces 
        #and lists to indicate thin film structures.
        grammar=[[self[0]]] #we will always start with the interface between the initial medium and the first layer
        for layer in self[1:-1]:
            if hasattr(layer,'n') and hasattr(layer,'d') and hasattr(layer,'coh') and layer.coh==False: #incoherent layer
                grammar[-1].append(layer) #finish previous interface or thinfilm filter
                grammar.append([layer]) #for layer matrix
                grammar.append([layer]) #for next interface or thin film stack
            elif hasattr(layer,'n') and hasattr(layer,'d'): #normal layer
                grammar[-1].append(layer) #proceed to create a thin film filter stack.
            elif isinstance(layer,list):
                if layer[0].d ==None: layer = layer[1:] #Q.what to do about initial or final layers?
                if layer[-1].d ==None: layer = layer[:1]
                grammar[-1].extend(layer) #add filter stack to current stack 
            else:
                raise NameError("unknown type of entry in filter list")
        #last layer
        grammar[-1].append(self[-1]) #this works for both ending with a incoherent layer or a thin film
 
        print("Incoherent Transfer Matrix, will process the following grammar")
        for token in grammar: print(token)
        print()
        
       ##Calculation
        I = N.array(((1.0,0.0),(0.0,1.0)))
        tmp = N.array( (I,)*len(self.axis) ) # running variable to hold the calculation results. Starts as an array of identity 'matrices'

        #print grammar
        
        for entry in grammar:
            if len(entry)==2:
                #print 'incoherent interface calculation'
                tmp=self._mat_mult(tmp,self._interface_array(entry))
            elif len(entry)==1:
                #print 'incoherent layer calculation'
                tmp=self._mat_mult(tmp,self._layer_array(entry[0]))
            elif len(entry)>2:
                #print 'thin film calculation'
                fltr = Filter(entry,w=self.w,pol=self.pol,theta=self.theta) #Create a thin film object
                fltrM = self._thin_film(fltr) #Find the matrix for the thin film system.
                tmp=self._mat_mult(tmp,fltrM)
            else:
                error

        return tmp
    
    @interactive
    def calculate_R_T(self):
        """Calculate the transmittance and reflectance of the system"""
        axis = self.axis
        M = self._calculate_M()
        T = 1.0/M[:,0,0]
        R = M[:,1,0]/M[:,0,0]
        return (axis,R.real,T.real)
    
    """
    def _calculate_M(self):
        ""Calculates the incoherent system matrix.""
        I = N.array(((1,0),(0,1)))
        tmp = N.array( (I,)*len(self.axis) ) # running variable to hold the calculation results. Starts as an array of identity 'matrices'
        
        stack=[self[0]]
        for layer in self[1:-1]:            
            if layer == coherent: ??
                stack.append(layer)
            elif type(layer)==type(list): ?? #Filter is derived from list and so these should be caught by this case too 
                stack.extend(layer)
            else: #found an incoherent layer or end layer
                stack.append(layer)
                if len(stack)>2: #Must be a thin film system
                    fltr = Filter(stack,w=self.w,pol=self.pol,theta=self.theta,*args,**kwargs) #Create a thin film object
                    fltrM = fltr.calculate_incoherent_M() #Find the matrix for the thin film system.
                    tmp=self._mat_mult(tmp,fltrM)
                    #
                    tmp=self._mat_mult(tmp,self._layer_array(layer))
                elif len(stack)==2: #stack layer must be incoherent too or this is the first layer 
                    tmp=self._mat_mult(tmp,self._layer_array2(stack)
                else:
                    raise(NameError("error in _calculate_M: stack should have a length of at least 2")
                stack=[layer] #resetting temporary variable                
        #Last layer
        stack.append(self[-1])
        if len(stack)>2: #Must be a thin film system
            fltr = Filter(stack,w=self.w,pol=self.pol,theta=self.theta,*args,**kwargs) #Create a thin film object
            fltrM = fltr.calculate_incoherent_M() #Find the matrix for the thin film system.
            tmp=self._mat_mult(tmp,fltrM)
        elif len(stack)==2: #stack layer must be incoherent too or this is the first layer 
            tmp=self._mat_mult(tmp,self._interface_array(stack)
        else:
            raise(NameError("error in _calculate_M: stack should have a length of at least 2")
        
        return tmp
    """
    
    """    
  
    def _mat_vec_mult(self,A,v):
        ""Multiplies an array of 2x2 matrices to an array of 2x1 vectors using matrix multiplication.
        A is an arrays nx2x2 where n is the frequency axis of the calculation.
        v is a vector nx2x1.""
        return  N.sum(A*v[:,N.newaxis,:],axis=-1)
            
    def _calculate_UV(self):
        ""In the transfer matrix model, each layer's electric field can be described
        as the summation of two waves, one travelling in each direction. I call the
        waves U & V in my derivation, hence the method name. This method calculates
        the amplitudes of these waves for each layer. However, to calculate the total
        electric field within the layer, we would need to take account of the wave's
        directions as well.""
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
        for layer_pair in zip(reverse_layer_order[1:],reverse_layer_order):
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
        ""Calculates the angle of the Efield within each layer""
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
        ""Calculate the electric field within the thin film filter.
        z is the position vector, the origin is located at the end of the filter.""
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
            raise NameError("pol should be 'TE' or 'TM'")
    
    @interactive
    def Absorption(self,z):
        Efields=self.calculate_E(self,z)
     
    """     
if __name__ == "__main__":
    pi=N.pi
    f=N.linspace(1.0e10,10e12,200)
    
    ########Define interference filter##########
    f1 = IncoherentFilter([Layer_eps(1.0,None),
            Layer_eps(3.50,8.6e-6,coh=False),
            Layer_eps(12.25,None)],
            w=2*pi*f,
            pol='TE',
            theta=0.0)
        
    ########Calculate Reflection/Transmission###
    w,R,T=f1.calculate_R_T(pol='TM')
    w,R2,T2=f1.calculate_R_T(pol='TE')
    
    #print f1._lambda((f1[0],f1[-1]))
    
    import matplotlib.pyplot as P
    
    ax1 = P.subplot(111)
    ax1.plot(f,R,label="TM Reflection")
    ax1.plot(f,T,label="TM Transmission")
    ax1.plot(f,R2,label="TE Reflection")
    ax1.plot(f,T2,label="TE Transmission")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_title("Antireflection coating for GaAS or Silicon")
    ax1.legend()
        
    P.show()