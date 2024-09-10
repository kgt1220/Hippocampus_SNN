# import
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import seaborn as sns
from scipy import signal

import pickle
import sys
import os

class SNN:
    def __init__(self, params, neuron_class, synapse, learned_weights, AL):
        # The number of neurons in each layer
        self.N_I = params.N_I
        self.N_O = params.N_O
        
        self.N_DG = params.N_DG
        self.N_DGH = params.N_DGH
        self.N_DGM = params.N_DGM
        self.N_DGB = params.N_DGB
        
        self.N_CA3 = params.N_CA3
        self.N_CA3i = params.N_CA3i
        self.N_CA1 = params.N_CA1
        
        # For delay of transmission
        self.d_I = params.d_I
        self.d_O = params.d_O
        self.d_DG = params.d_DG
        self.d_CA3 = params.d_CA3
        self.d_CA3i = params.d_CA3i
        self.outIn_L = np.zeros((self.d_I, self.N_I))
        self.outO_L = np.zeros((self.d_O, self.N_O))
        self.outDG_L = np.zeros((self.d_DG, self.N_DG))
        self.outCA3_L = np.zeros((self.d_CA3, self.N_CA3))
        self.outCA3i_L = np.zeros((self.d_CA3i, self.N_CA3i))

        # Outer functions
        self.neuron = neuron_class
        self.synapse = synapse()
        
        # Already learned weights
        self.AL = AL
        self.AL_ppCA3 = learned_weights[0]
        self.AL_Rc = learned_weights[1]
        self.AL_CA33i = learned_weights[2]
        self.AL_Sc = learned_weights[3]
        
        # Carry out inner functions related to initial setup
        self.setup_synapse(params)
        self.setup_layer()
        
        self.priorCA1 = np.zeros(self.N_CA1)
        
        # STDP parameters
        self.gra = 1 
        self.grc = 40     
        self.q_max = 3

    # Initialize connection weights
    def setup_synapse(self, params):
        # Environment to input
        self.W_EI, self.q_EI = self.synapse.take_weights(self.N_I, self.N_I, 'O', params.w_EI, 'strong', 1)
        self.g_EI_AM = np.zeros((self.N_I, self.N_I))
        self.g_EI_NM = np.zeros((self.N_I, self.N_I))
        # From perforant path
        # receive inputs by the number
        random.seed(8)
        self.W_ppDG, self.q_ppDG = self.synapse.take_weights(self.N_I, self.N_DG, 'Post_view', params.w_ppDG, 'strong', params.c_ppDG)
        self.g_ppDG_AM = np.zeros((self.N_I,self.N_DG))
        self.g_ppDG_NM = np.zeros((self.N_I,self.N_DG))
        random.seed(100)
        self.W_ppCA3, self.q_ppCA3 = self.synapse.take_weights(self.N_I, self.N_CA3, 'Post_view', params.w_ppCA3, 'weak', params.c_ppCA3)
        self.g_ppCA3_AM = np.zeros((self.N_I, self.N_CA3))
        self.g_ppCA3_NM = np.zeros((self.N_I, self.N_CA3))
        self.stp_x_ppCA3 = np.full((self.N_I, self.N_CA3), 1, dtype=np.float64)
        random.seed(125) 
        self.W_ppCA1, self.q_ppCA1 = self.synapse.take_weights(self.N_O, self.N_CA1, 'ppCA1', params.w_ppCA1, 'strong', params.c_ppCA1)
        self.g_ppCA1_AM = np.zeros((self.N_O, self.N_CA1))
        self.g_ppCA1_NM = np.zeros((self.N_O, self.N_CA1))
        
        # Mossy fiber
        random.seed(150)
        self.W_mf, self.q_mf = self.synapse.take_weights(self.N_DG, self.N_CA3, 'Pre_view', params.w_mf, 'strong', params.c_mf)
        self.g_mf_AM = np.zeros((self.N_DG, self.N_CA3))
        self.g_mf_NM = np.zeros((self.N_DG, self.N_CA3))
        random.seed(160)
        self.W_mfi, self.q_mfi = self.synapse.take_weights(self.N_DG, self.N_CA3i, 'Pre_view', params.w_mfi, 'strong', params.c_mfi)
        self.g_mfi_AM = np.zeros((self.N_DG, self.N_CA3i))
        self.g_mfi_NM = np.zeros((self.N_DG, self.N_CA3i))
        
        # CA3
        random.seed(200)
        self.W_Rc, self.q_Rc = self.synapse.take_weights(self.N_CA3, self.N_CA3, 'Pre_view', params.w_Rc, 'weak', params.c_Rc)
        self.g_Rc_AM = np.zeros((self.N_CA3, self.N_CA3))
        self.g_Rc_NM = np.zeros((self.N_CA3, self.N_CA3))
        self.stp_x_Rc = np.full((self.N_CA3, self.N_CA3), 1, dtype=np.float64)
        random.seed(260)
        self.W_CA33i, self.q_CA33i = self.synapse.take_weights(self.N_CA3, self.N_CA3i, 'Pre_view', params.w_CA33i, 'EtoI', params.c_CA33i)
        self.g_CA33i_AM = np.zeros((self.N_CA3, self.N_CA3i))
        self.g_CA33i_NM = np.zeros((self.N_CA3, self.N_CA3i))  
        self.stp_x_CA33i =np.full((self.N_CA3, self.N_CA3i), 1, dtype=np.float64)
        random.seed(290)
        self.W_3iCA3, self.q_3iCA3 = self.synapse.take_weights(self.N_CA3i, self.N_CA3, 'Pre_view', params.w_3iCA3, 'strong', params.c_3iCA3)
        self.g_3iCA3_GA = np.zeros((self.N_CA3i, self.N_CA3))
        self.g_3iCA3_GB = np.zeros((self.N_CA3i, self.N_CA3))
        self.stp_x_3iCA3 =np.full((self.N_CA3i, self.N_CA3), 1, dtype=np.float64)
        random.seed(300)
        self.W_3i3i, self.q_3i3i = self.synapse.take_weights(self.N_CA3i, self.N_CA3i, 'Pre_view', params.w_3i3i, 'strong', params.c_3i3i)
        self.g_3i3i_GA = np.zeros((self.N_CA3i, self.N_CA3i))
        self.g_3i3i_GB = np.zeros((self.N_CA3i, self.N_CA3i))
        self.stp_x_3i3i =np.full((self.N_CA3i, self.N_CA3i), 1, dtype=np.float64)
        for i in range(params.N_CA3i):
            for j in range(params.N_CA3i):
                if i == j:
                    self.q_3i3i[i,j] = 0
        
        # Noise
        self.W_NoCA3, self.q_NoCA3 = self.synapse.take_weights(self.N_CA3, self.N_CA3, 'O', params.w_NoCA3, 'strong', 1)
        self.g_NoCA3_AM = np.zeros((self.N_CA3, self.N_CA3))
        self.g_NoCA3_NM = np.zeros((self.N_CA3, self.N_CA3))

        # DG
        random.seed(280) 
        self.W_IH, self.q_IH = self.synapse.take_weights(self.N_I, self.N_DGH, 'O', params.w_IH, 'strong', params.c_IH)               
        self.g_IH_AM = np.zeros((self.N_I, self.N_DGH))
        self.g_IH_NM = np.zeros((self.N_I, self.N_DGH))
        random.seed(79)
        self.W_IB, self.q_IB = self.synapse.take_weights(self.N_I, self.N_DGB, 'Post_view_with_random', params.w_IB, 'strong', params.c_IB)
        self.g_IB_AM = np.zeros((self.N_I, self.N_DGB))
        self.g_IB_NM = np.zeros((self.N_I, self.N_DGB))
        random.seed(53)
        self.W_IM, self.q_IM = self.synapse.take_weights(self.N_I, self.N_DGM, 'Post_view_with_random', params.w_IM, 'strong', params.c_IM) 
        self.g_IM_AM = np.zeros((self.N_I, self.N_DGM))
        self.g_IM_NM = np.zeros((self.N_I, self.N_DGM))
        random.seed(66)
        self.W_HM, self.q_HM = self.synapse.take_weights(self.N_DGH, self.N_DGM, 'Post_view', params.w_HM, 'strong', params.c_HM)        
        self.g_HM_GA = np.zeros((self.N_DGH, self.N_DGM))
        self.g_HM_GB = np.zeros((self.N_DGH, self.N_DGM))
        random.seed(14) 
        # 원래는 1:1임
        self.W_BDG, self.q_BDG = self.synapse.take_weights(self.N_DGB, self.N_DG, 'Post_view', params.w_BDG, 'strong', params.c_BDG)
        self.g_BDG_GA = np.zeros((self.N_DGB, self.N_DG))
        self.g_BDG_GB = np.zeros((self.N_DGB, self.N_DG))
        
        # Make connection based on HB
        random.seed(290) 
        self.W_MB, self.q_MB = self.synapse.take_weights(self.N_DGM, self.N_DGB, 'Post_view_with_random', params.w_MB, 'strong', params.c_MB)
        self.g_MB_AM = np.zeros((self.N_DGM, self.N_DGB))
        self.g_MB_NM = np.zeros((self.N_DGM, self.N_DGB))        

        # Scaffer collateral
        random.seed(300)
        self.W_Sc, self.q_Sc = self.synapse.take_weights(self.N_CA3, self.N_CA1, 'Post_view', params.w_Sc, 'weak', params.c_Sc)
        self.g_Sc_AM = np.zeros((self.N_CA3,self.N_CA1))
        self.g_Sc_NM = np.zeros((self.N_CA3,self.N_CA1))
        self.q_Sc_init = self.q_Sc.copy()
        
        # Initialize CA1 to deepEC connection based on the deepEC to CA1 connection.
        self.W_CA1O = np.zeros((self.N_CA1, self.N_O))
        self.q_CA1O = np.zeros((self.N_CA1, self.N_O))
        for n in range(self.N_CA1):
            for m in range(self.N_O):
                if self.W_ppCA1[m,n] != 0:
                    self.W_CA1O[n,m] = params.w_CA1O
                    self.q_CA1O[n,m] = 3
        self.g_CA1O_AM = np.zeros((self.N_CA1,self.N_O))
        self.g_CA1O_NM = np.zeros((self.N_CA1,self.N_O))
        
        # Already_learned
        if self.AL:
            self.q_ppCA3 = self.AL_ppCA3
            self.q_Rc = self.AL_Rc
            self.q_CA33i = self.AL_CA33i
            self.q_Sc = self.AL_Sc
    
    # Setup layers
    def setup_layer(self):
        self.layers = []
        # Input layer
        layer_neurons = np.array([])
        for n in range(self.N_I):
            W = [self.W_EI[:,n]]
            g = [self.g_EI_AM[:,n]]
            g_L = [self.g_EI_NM[:,n]]
            q = [self.q_EI[:,n]]
            layer_neurons = np.append(layer_neurons, self.neuron(W, g, g_L, q, 'Input'))
        self.layers.append(layer_neurons)
        
        # Output layer
        layer_neurons = np.array([])
        for n in range(self.N_O):
            W = [self.W_EI[:,n], self.W_CA1O[:,n]]
            g = [self.g_EI_AM[:,n], self.g_CA1O_AM[:,n]]
            g_L = [self.g_EI_NM[:,n], self.g_CA1O_NM[:,n]]
            q = [self.q_EI[:,n], self.q_CA1O[:,n]]
            layer_neurons = np.append(layer_neurons, self.neuron(W, g, g_L, q, 'Output'))
        self.layers.append(layer_neurons)
        
        # DG Hilar cells
        layer_neurons = np.array([])
        for n in range(self.N_DGH):
            W = [self.W_IH[:,n]]
            g = [self.g_IH_AM[:,n]]
            g_L = [self.g_IH_NM[:,n]]
            q = [self.q_IH[:,n]]
            layer_neurons = np.append(layer_neurons, self.neuron(W, g, g_L, q, 'DGH'))
        self.layers.append(layer_neurons)
        
        # DG mossy cells
        layer_neurons = np.array([])
        for n in range(self.N_DGM):
            W = [self.W_IM[:,n], self.W_HM[:,n]]
            g = [self.g_IM_AM[:,n], self.g_HM_GA[:,n]]
            g_L = [self.g_IM_NM[:,n], self.g_HM_GB[:,n]]
            q = [self.q_IM[:,n], self.q_HM[:,n]]
            layer_neurons = np.append(layer_neurons, self.neuron(W, g, g_L, q, 'DGM'))
        self.layers.append(layer_neurons)
        
        # DG basket cells
        layer_neurons = np.array([])
        for n in range(self.N_DGB):
            W = [self.W_MB[:,n], self.W_IB[:,n]]
            g = [self.g_MB_AM[:,n], self.g_IB_AM[:,n]]
            g_L = [self.g_MB_NM[:,n], self.g_IB_NM[:,n]]
            q = [self.q_MB[:,n], self.q_IB[:,n]]
            layer_neurons = np.append(layer_neurons, self.neuron(W, g, g_L, q, 'DGB'))
        self.layers.append(layer_neurons)
        
        # DG granule cells
        layer_neurons = np.array([])
        for n in range(self.N_DG):
            W = [self.W_ppDG[:,n], self.W_BDG[:,n]]
            g = [self.g_ppDG_AM[:,n], self.g_BDG_GA[:,n]]
            g_L = [self.g_ppDG_NM[:,n], self.g_BDG_GB[:,n]]
            q = [self.q_ppDG[:,n], self.q_BDG[:,n]]
            layer_neurons = np.append(layer_neurons, self.neuron(W, g, g_L, q, 'DG'))
        self.layers.append(layer_neurons)
        
        # CA3 layer
        layer_neurons = np.array([])
        for n in range(self.N_CA3):
            W = [self.W_mf[:,n], self.W_ppCA3[:,n], self.W_Rc[:,n], self.W_3iCA3[:,n], self.W_NoCA3[:,n]]
            g = [self.g_mf_AM[:,n], self.g_ppCA3_AM[:,n], self.g_Rc_AM[:,n], self.g_3iCA3_GA[:,n], self.g_NoCA3_AM[:,n]]
            g_L = [self.g_mf_NM[:,n], self.g_ppCA3_NM[:,n], self.g_Rc_NM[:,n], self.g_3iCA3_GB[:,n], self.g_NoCA3_NM[:,n]]
            q = [self.q_mf[:,n], self.q_ppCA3[:,n], self.q_Rc[:,n], self.q_3iCA3[:,n], self.q_NoCA3[:,n]]
            layer_neurons = np.append(layer_neurons, self.neuron(W, g, g_L, q, 'CA3'))
        self.layers.append(layer_neurons)
        
        # CA3 interneurons
        layer_neurons = np.array([])
        for n in range(self.N_CA3i):
            W = [self.W_mfi[:,n], self.W_CA33i[:,n], self.W_3i3i[:,n]]
            g = [self.g_mfi_AM[:,n], self.g_CA33i_AM[:,n], self.g_3i3i_GA[:,n]]
            g_L = [self.g_mfi_NM[:,n], self.g_CA33i_NM[:,n], self.g_3i3i_GB[:,n]]
            q = [self.q_mfi[:,n], self.q_CA33i[:,n], self.q_3i3i[:,n]]
            layer_neurons = np.append(layer_neurons, self.neuron(W, g, g_L, q, 'CA3i'))
        self.layers.append(layer_neurons)
        
        # CA1 layer
        layer_neurons = np.array([])
        for n in range(self.N_CA1):
            W = [self.W_ppCA1[:,n], self.W_Sc[:,n]]
            g = [self.g_ppCA1_AM[:,n], self.g_Sc_AM[:,n]]
            g_L = [self.g_ppCA1_NM[:,n], self.g_Sc_NM[:,n]]
            q = [self.q_ppCA1[:,n], self.q_Sc[:,n]]
            layer_neurons = np.append(layer_neurons, self.neuron(W, g, g_L, q, 'CA1'))
        self.layers.append(layer_neurons)
        
    # Updating weights through STDP
    # Direct perforant path                        
    def STDP_on_dpp(self):
        for ca3 in range(self.N_CA3):
            if self.layers[6][ca3].fired == 1:
                for i in range(self.N_I):
                    if self.layers[6][ca3].pre_weights[1][i] != 0:
                        if 0 <= self.layers[0][i].trace < 60:
                            delta = self.layers[0][i].trace
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                        elif self.layers[0][i].fired == 1:
                            delta = 0
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                        else:
                            LTP = 0
                        self.layers[6][ca3].q[1][i] += self.qe_max*LTP
                        if self.layers[6][ca3].q[1][i] >= self.q_max:
                            self.layers[6][ca3].q[1][i] = self.q_max
        for i in range(self.N_I):
            if self.layers[1][i].fired == 1:
                for ca3 in range(self.N_CA3):
                    if self.layers[6][ca3].pre_weights[1][i] != 0:
                        if 0 <= self.layers[6][ca3].trace < 60:
                            delta = self.layers[6][ca3].trace
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                            self.layers[6][ca3].q[1][i] += LTP
                            if self.layers[6][ca3].q[1][i] >= self.q_max:
                                self.layers[6][ca3].q[1][i] = self.q_max
 
    # Recurrent colleteral (excitatory to excitatory)
    def STDP_on_Rc(self):
        # n: post, m : pre
        for n in range(self.N_CA3):
            if self.layers[6][n].fired == 1:
                for m in range(self.N_CA3):
                    if self.layers[6][n].pre_weights[2][m] != 0:
                        if 0 <= self.layers[6][m].trace < 60:
                            delta = self.layers[6][m].trace
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                        elif self.layers[6][m].fired == 1 and n <= m:
                            delta = 0
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                        else:
                            LTP = 0
                        self.layers[6][n].q[2][m] += 1.25*LTP
                        self.layers[6][m].q[2][n] += 1.25*LTP

                        if self.layers[6][n].q[2][m] >= self.q_max:
                            self.layers[6][n].q[2][m] = self.q_max
                        if self.layers[6][n].q[2][m] <= 0:
                            self.layers[6][n].q[2][m] = 0
                        if self.layers[6][m].q[2][n] >= self.q_max:
                            self.layers[6][m].q[2][n] = self.q_max
                        if self.layers[6][m].q[2][n] <= 0:
                            self.layers[6][m].q[2][n] = 0  
                            
    # Recurrent colleteral (excitatory to inhibitory)
    def STDP_on_Rci(self):
        for ca3i in range(self.N_CA3i):
            if self.layers[7][ca3i].fired == 1:
                for ca3 in range(self.N_CA3):
                    if self.layers[7][ca3i].pre_weights[1][ca3] != 0:
                        if 0<= self.layers[6][ca3].trace < 60:
                            delta = self.layers[6][ca3].trace
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                        elif self.layers[6][ca3].fired == 1:
                            delta = 0
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                        else:
                            LTP = 0    
                        self.layers[7][ca3i].q[1][ca3] += LTP
                        if self.layers[7][ca3i].q[1][ca3] >= self.q_max:
                            self.layers[7][ca3i].q[1][ca3] = self.q_max
                        elif self.layers[7][ca3i].q[1][ca3] <= 0:
                            self.layers[7][ca3i].q[1][ca3] = 0
        for ca3 in range(self.N_CA3):
            if self.layers[6][ca3].fired == 1:
                for ca3i in range(self.N_CA3i):
                    if self.layers[7][ca3i].pre_weights[1][ca3] != 0:
                        if 0 <= self.layers[7][ca3i].trace < 60:
                            delta = self.layers[7][ca3i].trace
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                            self.layers[7][ca3i].q[1][ca3] += LTP
                            if self.layers[7][ca3i].q[1][ca3] >= self.q_max:
                                self.layers[7][ca3i].q[1][ca3] = self.q_max
                            elif self.layers[7][ca3i].q[1][ca3] <= 0:
                                self.layers[7][ca3i].q[1][ca3] = 0
    
    # Schaffer colleteral
    def STDP_on_Sc(self):
        for ca1 in range(self.N_CA1):
            if self.layers[8][ca1].fired == 1:
                for ca3 in range(self.N_CA3):
                    if self.layers[8][ca1].pre_weights[1][ca3] != 0:
                        if 0<= self.layers[6][ca3].trace < 60:
                            delta = self.layers[6][ca3].trace
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                        elif self.layers[6][ca3].fired == 1:
                            delta = 0    
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                        else:
                            LTP = 0        
                        self.layers[8][ca1].q[1][ca3] += 1.5*LTP
                        if self.layers[8][ca1].q[1][ca3] >= self.q_max:
                            self.layers[8][ca1].q[1][ca3] = self.q_max
        for ca3 in range(self.N_CA3):
            if self.layers[6][ca3].fired == 1:
                for ca1 in range(self.N_CA1):
                    if self.layers[8][ca1].pre_weights[1][ca3] != 0:
                        if 0 <= self.layers[8][ca1].trace < 60:
                            delta = self.layers[8][ca1].trace
                            LTP = self.gra*np.exp(-delta**2/(2*self.grc**2))
                            self.layers[8][ca1].q[1][ca3] += 1.5*LTP
                            if self.layers[8][ca1].q[1][ca3] >= self.q_max:
                                self.layers[8][ca1].q[1][ca3] = self.q_max
       
    def solve(self, inputs, CA3inputs, Noise, ER, time_step, only_DG, only_CA3, En_win, Re_win):
        t = 0.001*time_step
        
        if ER and (time_step+1) % En_win == 0 or not ER and (time_step+1) % Re_win == 0:
            self.outIn_L = np.zeros((self.d_I, self.N_I))
            self.outO_L = np.zeros((self.d_O, self.N_O))
            self.outDG_L = np.zeros((self.d_DG, self.N_DG))
            self.outCA3_L = np.zeros((self.d_CA3, self.N_CA3))
            self.outCA3i_L = np.zeros((self.d_CA3i, self.N_CA3i))
        
        # Env to input layer
        outIn = np.array([])
        for n, neuron in enumerate(self.layers[0]):
            fired = neuron.solve([inputs], t, ER, n, En_win, Re_win)
            neuron.update_trace(ER)
            outIn = np.append(outIn, fired)
        self.outIn_L = np.roll(self.outIn_L, -self.N_I)
        self.outIn_L[-1] = outIn
            
        # Env to output layer
        outO = np.array([])
        for n, neuron in enumerate(self.layers[1]):
            fired = neuron.solve([inputs, self.priorCA1], t, ER, n, En_win, Re_win)
            outO = np.append(outO, fired)
        self.outO_L = np.roll(self.outO_L, -self.N_O)
        self.outO_L[-1] = outO
            
        if not only_CA3:
            # To the DG Hilar cell
            outDGH = np.array([])
            for n, neuron in enumerate(self.layers[2]):
                fired = neuron.solve([outIn], t, ER, n, En_win, Re_win)
                outDGH = np.append(outDGH, fired)
        
            # To the DG mossy cell
            outDGM = np.array([])
            for n, neuron in enumerate(self.layers[3]):
                fired = neuron.solve([self.outIn_L[3], outDGH], t, ER, n, En_win, Re_win)
                outDGM = np.append(outDGM, fired)
            
            # To the DG basket cell
            outDGB = np.array([])
            for n, neuron in enumerate(self.layers[4]):
                fired = neuron.solve([outDGM, outIn, outDGH], t, ER, n, En_win, Re_win)
                outDGB = np.append(outDGB, fired)

            # To DG layer
            outDG = np.array([])
            for n, neuron in enumerate(self.layers[5]):
                fired = neuron.solve([self.outIn_L[0], outDGB], t, ER, n, En_win, Re_win)
                outDG = np.append(outDG, fired)
            self.outDG_L = np.roll(self.outDG_L, -self.N_DG)
            self.outDG_L[-1] = outDG
        if not only_DG:
            if only_CA3:
                self.outDG_L = np.roll(self.outDG_L, -self.N_DG)
                self.outDG_L[-1] = CA3inputs
                
            # To the CA3 layer
            outCA3 = np.array([])
            for n, neuron in enumerate(self.layers[6]):
                fired = neuron.solve([self.outDG_L[0], outIn, self.outCA3_L[0], self.outCA3i_L[0], Noise], t, ER, n, En_win, Re_win)
                neuron.update_trace(ER)
                outCA3 = np.append(outCA3, fired)
            self.outCA3_L = np.roll(self.outCA3_L, -self.N_CA3)
            self.outCA3_L[-1] = outCA3

            # To the CA3 interneuron
            outCA3i = np.array([])
            for n, neuron in enumerate(self.layers[7]):
                fired = neuron.solve([self.outDG_L[-1], self.outCA3_L[-1], self.outCA3i_L[0]], t, ER, n, En_win, Re_win)
                neuron.update_trace(ER)
                outCA3i = np.append(outCA3i, fired)
            self.outCA3i_L = np.roll(self.outCA3i_L, -self.N_CA3i)
            self.outCA3i_L[-1] = outCA3i

            # To the CA1 layer
            outCA1 = np.array([])
            for n, neuron in enumerate(self.layers[8]):
                fired = neuron.solve([self.outO_L[0], self.outCA3_L[-1]], t, ER, n, En_win, Re_win)
                neuron.update_trace(ER)
                outCA1 = np.append(outCA1, fired)
            self.priorCA1 = outCA1.copy()
            
            if not self.AL:
                self.STDP_on_dpp()
                self.STDP_on_Rc()
                self.STDP_on_Rci()
                self.STDP_on_Sc()
        return 

