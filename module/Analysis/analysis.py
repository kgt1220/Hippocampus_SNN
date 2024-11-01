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

def cal_tendency(bef,af):
    tend_temp = 0
    if len(bef) !=0:
        tend_temp += len(af)
    return tend_temp

def cal_tendency2(BEF,AF):
    tend_A = cal_tendency(BEF[0], AF[0])
    tend_B = cal_tendency(BEF[1], AF[1])
    #tend_C = cal_tendency(BEF[2], AF[2])
    #tend_D = cal_tendency(BEF[3], AF[3])
    #tendency_temp = (tend_A+tend_B+tend_C+tend_D)/(len(AF[0])+len(AF[1])+len(AF[2])+len(AF[3]))
    tendency_temp = 0
    if len(AF[0])+len(AF[1]) !=0:
        tendency_temp = (tend_A+tend_B)/(len(AF[0])+len(AF[1]))
    return tendency_temp


# In[ ]:


def cal_winner_onCA3(win,af_py):
    win_temp = 0
    if len(win)/len(af_py) >= 0.5:
        win_temp = 1
    return win_temp


# In[ ]:


def Analysis_results(ret, timing, en_num, en_num2):
    tempa = set()
    for i, comp in enumerate(ret):
        print('Firing neuron in %d at %dth : ' % (en_num,timing), comp)
        tempa |= set(np.where(network.q_CA33i[comp,:] > 2)[0])
    print('Inh neuron by comp : ', tempa)

    if len(tempa) !=0:
        tempc = set()
        for i, comp in enumerate(tempa):
            tempb = np.where(network.q_3iCA3[comp,Af_py[en_num2]] !=0)[0]
            tempc |= set(tempb)
        tempd = []
        for j, comp2 in enumerate(tempc):
            tempd.append(Af_py[en_num2][comp2])
        print('Should be inhibited neuron on %d : ' % en_num, tempd)
    else:
        print('No inh neurons are firing')


# In[ ]:


def Analysis_reason(N, n):
    temp = np.where(CA3_FT[N,:] !=0)[0]
    for i, comp in enumerate(network.layers[6][N].WTS):
        if 0.001*(temp[n]-20) <= comp[2] < 0.001*temp[n]:
            if comp[0] == 1:
                print('by pp : ', comp, temp[n])
            elif comp[0] == 2:
                print('by rc : ', comp, temp[n])
            elif comp[0] == 5:
                print('by noise : ', comp, temp[n])
            else:
                print(comp)

