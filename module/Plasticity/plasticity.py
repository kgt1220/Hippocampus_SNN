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

class Synapse:
    def __init__(self):
        self.q_s = 3 #0.04
        self.q_w = 0 #0.001
        self.q_EI = 0.5

    def take_weights(self, layer1, layer2, syn_type, strength, cond_str, N):
        # One-to-one
        if syn_type == 'O':
            weights = np.zeros((layer1, layer2))
            q = np.zeros((layer1, layer2))
            for n in range(layer1):
                for m in range(layer2):
                    if n == m:
                        weights[n][m] = strength
                        if cond_str == 'strong':
                            q[n][m] = self.q_s
                        elif cond_str == 'weak':
                            q[n][m] = self.q_w
                            
        # Initialize connection in the perspective of the postsynaptic neurons
        elif syn_type == 'Post_view':
            # normal, proper number from input
            weights = np.zeros((layer1, layer2))
            q = np.zeros((layer1, layer2))
            for m in range(layer2):
                input_idx = self.choice_neurons(layer1, N)
                weights[input_idx,m] = strength*np.abs(np.random.normal(1,0.01,len(input_idx)))
                if cond_str == 'strong':
                    q[input_idx,m] = self.q_s
                elif cond_str == 'weak':
                    q[input_idx,m] = self.q_w
                    
        # Initialize connection in the perspective of the presynaptic neurons
        elif syn_type == 'Pre_view':
            # normal, proper number from input
            weights = np.zeros((layer1, layer2))
            q = np.zeros((layer1, layer2))
            for n in range(layer1):
                input_idx = self.choice_neurons(layer2, N)
                weights[n,input_idx] = strength*np.abs(np.random.normal(1,0.01,len(input_idx)))
                if cond_str == 'strong':
                    q[n,input_idx] = self.q_s
                elif cond_str == 'weak':
                    q[n,input_idx] = self.q_w
                elif cond_str == 'EtoI':
                    q[n,input_idx] = self.q_EI
        
        # Choose the number of presynaptic neurons randomly
        elif syn_type == 'Post_view_with_random':
            weights = np.zeros((layer1, layer2))
            q = np.zeros((layer1, layer2))
            for m in range(layer2):
                N_ran = np.random.randint(N)+1
                input_idx = self.choice_neurons(layer1, N_ran)
                weights[input_idx,m] = strength*np.abs(np.random.normal(1,0.01,len(input_idx)))
                if cond_str == 'strong':
                    q[input_idx,m] = self.q_s
                elif cond_str == 'weak':
                    q[input_idx,m] = self.q_w
        
        elif syn_type == 'ppCA1': 
            weights = np.zeros((layer1, layer2))
            q = np.zeros((layer1, layer2))
            for m in range(layer2):
                N_ran = np.random.randint(N)+1
                input_idx = self.choice_neurons(layer1, N_ran)
                weights[input_idx,m] = strength*np.abs(np.random.normal(1,0.01,len(input_idx)))/N_ran
                if cond_str == 'strong':
                    q[input_idx,m] = self.q_s
                elif cond_str == 'weak':
                    q[input_idx,m] = self.q_w
                    
        return weights, q
    
    # Choose N neurons without repetition
    def choice_neurons(self, layer, N):
        n_index = []
        num = random.randrange(0,layer)
        #N_R = np.random.randint(1, N)
        for n in range(N):
            while num in n_index:
                num = random.randrange(0,layer)
            n_index.append(num)
        n_index.sort()

        return n_index

