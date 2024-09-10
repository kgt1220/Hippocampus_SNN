#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import seaborn as sns
from scipy import signal

#import winsound as sd
import pickle
import sys
import os


# In[ ]:


class Neuron:
    def __init__(self, W, g, g_L, q, types):
        # Initialization
        self.types = types
        self.f_time = 0
        self.dt = 1 # ms
        self.trace = 500
        self.etime = 0
        self.rtime = 0
            
        self.pre_weights = W
        self.g = g
        self.g_L = g_L
        self.q = q

        self.N_types = len(self.pre_weights)
        
        # Currents boundary
        self.BDe = 5
        self.BDi = 1
        if self.types == 'DGM':
            self.BDe = 4
            self.BDi = 10
        elif self.types == 'DGH':
            self.BDe = 5
            self.BDi = 1
        elif self.types == 'DGB':
            self.BDe = 7
            self.BDi = 7
        elif self.types == 'DG':
            self.BDe = 5
            self.BDi = 10
        elif self.types == 'CA3':
            self.BDe = 2
            self.BDi = 10
        elif self.types == 'CA3i':
            self.BDe = 5
            self.BDi = 10
        elif self.types == 'CA1':
            self.BDe = 4
            self.BDi = 10
        
        self.BDr = 20
        self.BDp = 0.75
        self.alpha = 1.3

        # Izhichevich neuron's parameters
        if types == 'CA3i' or types == 'DGH' or types == 'DGB' or types == 'DGM':
            self.C = 20
            self.k = 1
            self.V_rest = -55
            self.V_th = -40
            self.V_peak = 25
            self.a = 0.15
            self.b = 8
            self.c = -55
            self.d = 200
        elif types == 'CA3' or types == 'Input' or types == 'Output' or types == 'DG' or types == 'CA1':
            self.C = 80
            self.k = 3
            self.V_rest = -60
            self.V_th = -50
            self.V_peak = 50
            self.a = 0.01
            self.b = 5
            self.c = -60
            self.d = 10
        else:
            self.C = 200
            self.k = 1.6
            self.V_rest = -60
            self.V_th = -50
            self.V_peak = 40
            self.a = 0.1
            self.b = 0
            self.c = -60
            self.d = 10
            
        self.u = 0
        self.V = np.full(2, self.V_rest)
        self.fire_record = np.zeros(2)     
                
        self.V_ex = 0
        self.V_inA = -70       
        self.V_inB = -90
        
        # Conductance parameters 
        self.tau_AM = 5 # ms
        self.tau_NM = 5
        if types == 'DGB':
            self.tau_AM = 5 
            self.tau_NM = 5 
            self.tau_GA = 6 
            self.tau_GB = 6 
        elif types == 'DGM':
            self.tau_AM = 5 
            self.tau_NM = 5
            self.tau_GA = 15 
            self.tau_GB = 15 
        elif types == 'DG':
            self.tau_AM = 5 
            self.tau_NM = 5
            self.tau_GA = 15
            self.tau_GB = 15
        elif types == 'CA3':
            self.tau_AM = 5
            self.tau_NM = 30
            self.tau_GA = 8
            self.tau_GB = 30
        elif types == 'CA3i':
            self.tau_AM = 5
            self.tau_NM = 30
            self.tau_GA = 8
            self.tau_GB = 30
        elif types == 'CA1':
            self.tau_AM = 5
            self.tau_NM = 30
        
    # Conductance for excitatory neurons
    def calculate_conductance(self, inputs, n):
        temp = np.where(inputs == 0)[0]
        inputs_N = np.zeros(len(inputs), dtype=int)
        inputs_N[temp] = 1
        
        self.g[n] = self.g[n] + inputs*(self.q[n])
        self.g[n] = inputs*self.g[n] + inputs_N*(1-self.dt/self.tau_AM)*self.g[n]
        self.g[n][np.where(self.g[n] < 0.0001)[0]] = 0
        
        if len(self.g_L) !=0:
            self.g_L[n] = self.g_L[n] + inputs*(self.q[n])
            self.g_L[n] = inputs*self.g_L[n] + inputs_N*(1-self.dt/self.tau_NM)*self.g_L[n]
            self.g_L[n][np.where(self.g_L[n] < 0.0001)[0]] = 0
            
    # Conductance for inhibitory neurons
    def calculate_conductance_in(self, inputs, n, N):
        temp = np.where(inputs == 0)[0]
        inputs_N = np.zeros(len(inputs), dtype=int)
        inputs_N[temp] = 1
        
        self.g[n] = self.g[n] + inputs*(self.q[n])
        self.g[n] = inputs*self.g[n] + inputs_N*(1-self.dt/self.tau_GA)*self.g[n]
        self.g[n][np.where(self.g[n] < 0.0001)[0]] = 0
        
        if len(self.g_L) !=0:
            self.g_L[n] = self.g_L[n] + inputs*(self.q[n])
            self.g_L[n] = inputs*self.g_L[n] + inputs_N*(1-self.dt/self.tau_GB)*self.g_L[n]
            self.g_L[n][np.where(self.g_L[n] < 0.0001)[0]] = 0
    
    # Conductance with noise
    def calculate_conductance_ran(self, inputs, n, N):
        temp = np.where(inputs == 0)[0]
        inputs_N = np.zeros(len(inputs), dtype=int)
        inputs_N[temp] = 1
        
        noise = np.full(len(self.g[n]),0.5)
        if self.partial_time % 10 == 0:
            noise = np.random.normal(0.5,1,len(self.g[n]))

        self.g[n] = self.g[n] + inputs*(self.q[n])*noise
        self.g[n] = inputs*self.g[n] + inputs_N*(1-self.dt/self.tau_AM)*self.g[n]
        self.g[n][np.where(self.g[n] < 0.0001)[0]] = 0
        
        if len(self.g_L) !=0:
            self.g_L[n] = self.g_L[n] + inputs*(self.q[n])*noise
            self.g_L[n] = inputs*self.g_L[n] + inputs_N*(1-self.dt/self.tau_NM)*self.g_L[n]
            self.g_L[n][np.where(self.g_L[n] < 0.0001)[0]] = 0  
            
    def solve(self, inputs, t, ER, N, En_win, Re_win):
        # Initialization
        self.V = np.roll(self.V, -1)
        self.summ_AM = np.array([])
        self.summ_NM = np.array([])
        self.summ_GA = np.array([])
        self.summ_GB = np.array([])
        self.pp_AM = np.array([])
        self.pp_NM = np.array([])
        self.Rc_AM = np.array([])
        self.Rc_NM = np.array([])
        self.noise_AM = np.array([])
        self.noise_NM = np.array([])
        
        # Current update
        for n in range(self.N_types):
            # Separate encoding and retrieval regidly
            if ER and self.etime == 0 or not ER and self.rtime == 0:
                self.g[n] = 0*self.g[n]
                self.g_L[n] = 0*self.g_L[n]
                self.V[0] = self.V_rest
                self.u = 0
                    
            if n == 0:
                if ER:
                    inputs_ph = inputs[n]
                # Activation of input layer when retrieval
                elif not ER and self.types == 'Input':
                    inputs_ph = inputs[n]
                else:
                    inputs_ph = 0*inputs[n]

                self.calculate_conductance(inputs_ph, n)
                self.summ_AM = np.append(self.summ_AM, np.dot(self.g[n], self.pre_weights[n]))
                
            elif 0 < n <= 4:
                if self.types == 'DGM' or self.types == 'DGB' or self.types == 'DG':
                    # Activated connections when encoding
                    if ER:
                        inputs_ph = inputs[n]
                    else:
                        inputs_ph = 0*inputs[n]
                        
                    if self.types == 'DGB':
                        self.calculate_conductance(inputs_ph, n)
                        self.summ_AM = np.append(self.summ_AM, np.dot(self.g[n], self.pre_weights[n]))
                    else:
                        self.calculate_conductance_in(inputs_ph, n, N)
                        self.summ_GA = np.append(self.summ_GA, np.dot(self.g[n], self.pre_weights[n]))
                elif self.types == 'CA3' and n == 3:
                    # Always receive inhibitory inputs
                    inputs_ph = inputs[n]
                    
                    self.calculate_conductance_in(inputs_ph, n, N)
                    self.summ_GA = np.append(self.summ_GA, np.dot(self.g[n], self.pre_weights[n]))
                    self.summ_GB = np.append(self.summ_GB, 0.1*np.dot(*self.g_L[n], self.pre_weights[n]))
                else:
                    # Activated connections when retrieval
                    if ER:
                        inputs_ph = 0*inputs[n]
                    else:
                        inputs_ph = inputs[n]

                    if self.types == 'CA3' and n == 1:
                        self.calculate_conductance(inputs_ph, n)
                        self.pp_AM = np.append(self.pp_AM, np.dot(self.g[n], self.pre_weights[n]))
                        self.pp_NM = np.append(self.pp_NM, 0.5*np.dot(self.g_L[n], self.pre_weights[n])) 

                    elif self.types == 'CA3' and n == 2:
                        self.calculate_conductance(inputs_ph, n)
                        self.Rc_AM = np.append(self.Rc_AM, np.dot(self.g[n], self.pre_weights[n]))
                        self.Rc_NM = np.append(self.Rc_NM, 0.5*np.dot(self.g_L[n], self.pre_weights[n])) 
                        
                    elif self.types == 'CA3' and n == 4:
                        self.calculate_conductance(inputs_ph, n)
                        self.noise_AM = np.append(self.noise_AM, np.dot(self.g[n], self.pre_weights[n]))
                        self.noise_NM = np.append(self.noise_NM, 0.5*np.dot(self.g_L[n], self.pre_weights[n]))  
                        
                    elif self.types == 'CA3i' and n == 2:
                        self.calculate_conductance_in(inputs_ph, n, N)
                        self.summ_GA = np.append(self.summ_GA, np.dot(self.g[n], self.pre_weights[n]))
                        self.summ_GB = np.append(self.summ_GB, 0.1*np.dot(self.g_L[n], self.pre_weights[n]))
                        
                    elif self.types == 'CA3i' and n == 1:
                        self.calculate_conductance(inputs_ph, n)
                        self.summ_AM = np.append(self.summ_AM, np.dot(self.g[n], self.pre_weights[n]))
                        self.summ_NM = np.append(self.summ_NM, 0.5*np.dot(self.g_L[n], self.pre_weights[n]))    
                        
                    else:
                        # CA1, output
                        self.calculate_conductance(inputs_ph, n)
                        self.summ_AM = np.append(self.summ_AM, np.dot(self.g[n], self.pre_weights[n]))
                        self.summ_NM = np.append(self.summ_NM, 0.5*np.dot(self.g_L[n], self.pre_weights[n]))                                   
        if ER:
            self.etime += 1
        else:
            self.rtime += 1
            
        if self.etime == En_win:
            self.etime = 0
            
        if self.rtime == Re_win:
            self.rtime = 0
        
        # Current summation
        temp = ((self.V[0]+80)/60)**2
        self.total_pp = self.alpha*self.BDp*np.tanh(self.pp_AM.sum()/self.BDp)*(self.V_ex - self.V[0]) + self.alpha*self.BDp*np.tanh(self.pp_NM.sum()/self.BDp)*(temp/(1+temp))*(self.V_ex - self.V[0])
        self.total_Rc = self.alpha*self.BDr*np.tanh(self.Rc_AM.sum()/self.BDr)*(self.V_ex - self.V[0]) + self.alpha*self.BDr*np.tanh(self.Rc_NM.sum()/self.BDr)*(temp/(1+temp))*(self.V_ex - self.V[0])
        self.total_N = self.noise_AM.sum()*(self.V_ex - self.V[0]) + self.noise_NM.sum()*(temp/(1+temp))*(self.V_ex - self.V[0])
        self.total_ex = self.alpha*self.BDe*np.tanh(self.summ_AM.sum()/self.BDe)*(self.V_ex - self.V[0]) + self.alpha*self.BDe*np.tanh(self.summ_NM.sum()/self.BDe)*(temp/(1+temp))*(self.V_ex - self.V[0])
        self.total_in = self.alpha*self.BDi*np.tanh(self.summ_GA.sum()/self.BDi)*(self.V_inA - self.V[0]) + self.alpha*self.BDi*np.tanh(self.summ_GB.sum()/self.BDi)*(self.V_inB - self.V[0]) 
        self.I = self.total_ex + self.total_in + self.total_pp + self.total_Rc + self.total_N

        # Voltage update
        if self.V[0] >= self.V_peak:
            self.V[1] = self.c
            self.u = self.u + self.d
        else:
            if self.types == 'CA3' or self.types == 'CA3i':
                self.V[1] = self.V[0] + self.dt*(self.k*(self.V[0]-self.V_rest)*(self.V[0]-self.V_th) - self.u + self.I - self.W)/self.C
                self.wts = self.I - self.W
            else:
                self.V[1] = self.V[0] + self.dt*(self.k*(self.V[0]-self.V_rest)*(self.V[0]-self.V_th) - self.u + self.I)/self.C
            self.u = self.u + self.dt*self.a*(self.b*(self.V[0]-self.V_rest) - self.u)

        if self.V[1] >= self.V_peak:
            self.fired = 1
            self.f_time = t
            self.V[1] = self.V_peak
        else:
            self.fired = 0
            self.f_time = 0

        self.fire_record = np.roll(self.fire_record, -1)
        self.fire_record[1] = self.fired
        return self.fired
    
    # Recording firing trace
    def update_trace(self, ER):
        if ER:
            if self.fire_record[0] == 1:
                self.trace = 0
            elif self.fire_record[0] == 0:             
                self.trace += 1
        else:
            self.trace = 500

