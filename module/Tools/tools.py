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

def choose_ran(number, min_num, max_num):
    index = []
    num = random.randrange(min_num, max_num)
    for n in range(number):
        while num in index:
            num = random.randrange(min_num, max_num)
        index.append(num)
    index.sort()
    return index


# In[ ]:

# Plot voltage of one neuron divided by encoding and retrieval
def plotV_ER(E, R, title):
    fig = plt.figure(figsize = (9,3))
    ax = fig.add_subplot(1,1,1)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    ax.set_xlabel('time(ms)')
    ax.set_ylabel('voltage(mV)')
    ax1.plot(E)
    ax1.set_ylim([-100,50])
    ax2.plot(R)
    ax2.set_ylim([-100,50])
    plt.suptitle(title)
    plt.show()

# Plot current of one neuron divided by encoding and retrieval
def plotT_ER(E, R, title, BD):
    fig = plt.figure(figsize = (9,3))
    ax = fig.add_subplot(1,1,1)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    ax.set_xlabel('time(ms)')
    ax1.plot(E)
    ax1.set_ylim([-BD,BD])
    ax2.plot(R)
    ax2.set_ylim([-BD,BD])
    plt.suptitle(title)
    plt.show()


# In[ ]:

# The number of firing
def prepare_data_FR(FR_temp, N, layers):
    for i in range(N):
        FR_temp[i] = FR_temp[i] + layers[i].fire_record[-1]
    return FR_temp


# In[ ]:

def trans_OH(Data, Dic_Idx, Dic_OH):
    shape = np.shape(Data)
    Data_OH = np.zeros(shape)
    Data_E = []
    Data_R = []
    Idx = []
    Val = []
    for n in range(shape[1]):
        Idx.append(list(np.where(Data[:,n] !=0)[0]))
        Val.append(np.max(Data[:,n]))
    
    for t, comp in enumerate(Idx):
        for k, v in Dic_Idx.items():
            if comp == list(v):
                Data_OH[:,t] = Val[t]*Dic_OH[k]
    
    return Data_OH


# In[ ]:


def generate_binvec(layer, N):
    idx = []
    num = random.randrange(0,layer)
    #N_R = np.random.randint(1, N)
    for n in range(N):
        while num in idx:
            num = random.randrange(0,layer)
        idx.append(num)
    idx.sort()
    data = np.zeros(layer)
    for n, comp in enumerate(idx):
        data[comp] = 1
    return data


# In[ ]:


def cal_overlap(total,A,B):
    output = len(A-(total- B))*2 / (len(A)+len(B))
    return output


# In[ ]:


def engram_filter(N, Datalen, DGlen_list):
    DGlen_idx = []
    for idx, comp in enumerate(DGlen_list):
        if 0 < comp < N:
            DGlen_idx.append(idx)
    Total_idx = set(range(0,Datalen))
    DGlen_idx_excluded = list(Total_idx-set(DGlen_idx))
    return DGlen_idx, DGlen_idx_excluded


# In[ ]:


def filter_SP(OverI, OverO, Pair_idx, DGlen_idx_excluded):
    Pair_idx_excluded = []
    # 제거해야할 인풋(engram 부적절) 중에
    for i, comp in enumerate(DGlen_idx_excluded):
        # 오버랩 계산된 애들 중 이 부적절 인풋이 들어가면 일단 모아놓음
        for j, comp2 in enumerate(Pair_idx):
            if comp in comp2:
                Pair_idx_excluded.append(comp2)
            
    OverI_v2 = []
    OverO_v2 = []
    Pair_idx_v2 = []
    
    # target SP 중에서 부적절 engram과 계산된 건 제거
    for k, comp3 in enumerate(Pair_idx):
        if comp3 not in Pair_idx_excluded:
            OverI_v2.append(OverI[k])
            OverO_v2.append(OverO[k])
            Pair_idx_v2.append(comp3)
    return OverI_v2, OverO_v2, Pair_idx_v2


# In[ ]:


def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 750     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

