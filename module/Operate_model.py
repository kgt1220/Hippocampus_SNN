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

from Tools.tools import *
from Plasticity.plasticity import *
from Neuron.neuron import *
from Neuron.neuron_No_dpp import *
from Model.model import *
from Analysis.analysis import *

# Data 
def run_model(En_win, Re_win, Num_tar, Num_cue, Repeat, Target, Cue, WTS, params, already_learned, learned_weights, Direct_pp, Filt_idata, Af_py, Af_in, Noise_on=True, plot_I=False, plot_O=False, plot_H=False, plot_M=False, plot_B=False, plot_DG=False, plot_CA3=False, plot_CA3i=False, plot_CA1=False):
    # Initialization
    Freq = 20    
    n = 0
    ER = False
    Onff = True
    Rtime = 0
    
    Phase_win = En_win+Re_win
    if already_learned:
        Phase_num = Num_cue*Repeat
    else:
        Phase_num = Num_tar*Repeat
    En_st = []
    for n in range(Phase_num):
        En_st.append(n*Phase_win)
    total_time = En_st[-1]+Phase_win
    TLen = len(Target)
    Noise_sum = np.zeros(params.N_CA3)
    
    WTS_I = WTS[0]
    WTS_DGH = WTS[1]
    WTS_DGM = WTS[2]
    WTS_DGB = WTS[3]
    WTS_DG = WTS[4]
    WTS_CA3 = WTS[5]
    WTS_CA3i = WTS[6]
    WTS_CA1 = WTS[7]
    
    Tendency1 = []
    Tendency2 = []
    Fail = 0
    If_fail = 0
    Success = 0
    Winfin = np.zeros(len(Target))
    
    CA3_R_list = []
    CA1_R_list = []
    
    In_FT_list =  []
    Out_FT_list = []
    H_FT_list = []
    M_FT_list = []
    B_FT_list = []
    DG_FT_list = []

    CA3_FT_list = []
    CA3i_FT_list = []
    CA1_FT_list = []
    
    In_V_list =  []
    Out_V_list = []
    H_V_list = []
    M_V_list = []
    B_V_list = []
    DG_V_list = []

    CA3_V_list = []
    CA3i_V_list = []
    CA1_V_list = []
    
    # Suppression direct perforant path
    if Direct_pp:
        network = SNN(params, Neuron, Synapse, learned_weights, already_learned)
    else:
        network = SNN(params, Neuron_No_dpp, Synapse, learned_weights, already_learned)

    for T in range(total_time):
        if n != len(En_st) and En_st[n] <= T < En_st[n] + En_win:
            ER = True
            # encoding 시작점
            if T == En_st[n]:
                # initiation
                In_V =  np.zeros((params.N_I, Phase_win))
                Out_V = np.zeros((params.N_O, Phase_win))
                H_V = np.zeros((params.N_DGH, Phase_win))
                M_V = np.zeros((params.N_DGM, Phase_win))
                B_V = np.zeros((params.N_DGB, Phase_win))
                DG_V = np.zeros((params.N_DG, Phase_win))
                
                CA3_V = np.zeros((params.N_CA3, Phase_win))
                CA3i_V = np.zeros((params.N_CA3i, Phase_win))
                CA1_V = np.zeros((params.N_CA1, Phase_win))
                
                In_FT =  np.zeros((params.N_I, Phase_win))
                Out_FT = np.zeros((params.N_O, Phase_win))
                H_FT = np.zeros((params.N_DGH, Phase_win))
                M_FT = np.zeros((params.N_DGM, Phase_win))
                B_FT = np.zeros((params.N_DGB, Phase_win))
                DG_FT = np.zeros((params.N_DG, Phase_win))
                
                CA3_FT = np.zeros((params.N_CA3, Phase_win))
                CA3i_FT = np.zeros((params.N_CA3i, Phase_win))
                CA1_FT = np.zeros((params.N_CA1, Phase_win))

                V_E_I = np.zeros((len(WTS_I),En_win))
                V_R_I = np.zeros((len(WTS_I),Re_win))
                V_E_O = np.zeros((len(WTS_I),En_win))
                V_R_O = np.zeros((len(WTS_I),Re_win))

                V_E_DGH = np.zeros((len(WTS_DGH),En_win))
                V_R_DGH = np.zeros((len(WTS_DGH),Re_win))
                V_E_DGM = np.zeros((len(WTS_DGM),En_win))
                V_R_DGM = np.zeros((len(WTS_DGM),Re_win))
                V_E_DGB = np.zeros((len(WTS_DGB),En_win))
                V_R_DGB = np.zeros((len(WTS_DGB),Re_win))
                V_E_DG = np.zeros((len(WTS_DG),En_win))
                V_R_DG = np.zeros((len(WTS_DG),Re_win))

                V_E_CA3 = np.zeros((len(WTS_CA3),En_win))
                V_R_CA3 = np.zeros((len(WTS_CA3),Re_win))
                V_E_CA3i = np.zeros((len(WTS_CA3i),En_win))
                V_R_CA3i = np.zeros((len(WTS_CA3i),Re_win))

                V_E_CA1 = np.zeros((len(WTS_CA1),En_win))
                V_R_CA1 = np.zeros((len(WTS_CA1),Re_win))

                Tex_E_I = np.zeros((len(WTS_I),En_win))
                Tin_E_I = np.zeros((len(WTS_I),En_win))
                Tex_R_I = np.zeros((len(WTS_I),Re_win))
                Tin_R_I = np.zeros((len(WTS_I),Re_win))

                Tex_E_O = np.zeros((len(WTS_I),En_win))
                Tin_E_O = np.zeros((len(WTS_I),En_win))
                Tex_R_O = np.zeros((len(WTS_I),Re_win))
                Tin_R_O = np.zeros((len(WTS_I),Re_win))

                Tex_E_DGM = np.zeros((len(WTS_DGM),En_win))
                Tin_E_DGM = np.zeros((len(WTS_DGM),En_win))
                Tex_R_DGM = np.zeros((len(WTS_DGM),Re_win))
                Tin_R_DGM = np.zeros((len(WTS_DGM),Re_win))

                Tex_E_DGB = np.zeros((len(WTS_DGB),En_win))
                Tin_E_DGB = np.zeros((len(WTS_DGB),En_win))
                Tex_R_DGB = np.zeros((len(WTS_DGB),Re_win))
                Tin_R_DGB = np.zeros((len(WTS_DGB),Re_win))

                Tex_E_DG = np.zeros((len(WTS_DG),En_win))
                Tin_E_DG = np.zeros((len(WTS_DG),En_win))
                Tex_R_DG = np.zeros((len(WTS_DG),Re_win))
                Tin_R_DG = np.zeros((len(WTS_DG),Re_win))

                Tex_E_CA3 = np.zeros((len(WTS_CA3),En_win))
                Tin_E_CA3 = np.zeros((len(WTS_CA3),En_win))
                Tpp_R_CA3 = np.zeros((len(WTS_CA3),Re_win))
                Trc_R_CA3 = np.zeros((len(WTS_CA3),Re_win))
                Tno_R_CA3 = np.zeros((len(WTS_CA3),Re_win))
                Tin_R_CA3 = np.zeros((len(WTS_CA3),Re_win))

                Tex_E_CA3i = np.zeros((len(WTS_CA3i),En_win))
                Tin_E_CA3i = np.zeros((len(WTS_CA3i),En_win))
                Tex_R_CA3i = np.zeros((len(WTS_CA3i),Re_win))
                Tin_R_CA3i = np.zeros((len(WTS_CA3i),Re_win))

                Tex_E_CA1 = np.zeros((len(WTS_CA1),En_win))
                Tin_E_CA1 = np.zeros((len(WTS_CA1),En_win))
                Tex_R_CA1 = np.zeros((len(WTS_CA1),Re_win))
                Tin_R_CA1 = np.zeros((len(WTS_CA1),Re_win))              
                
                t = 0
                Te = 0
                Tr = 0
        else:
            ER = False
   
        # Encoding
        env = np.zeros(params.N_I)   
        if T % Freq == 0 and En_st[n] <= T < En_st[n]+En_win:
            if not already_learned:
                env = Filt_idata[:,Target[n]]

        # Retrieval
        elif T % Freq == 0 and En_st[n]+En_win <= T < En_st[n]+Phase_win:
            idx_n = n // Repeat
            if Num_cue == 1:
                env = Cue
            else:
                env = Cue[:,idx_n]

        # Noise : 0.0035 = 3.5Hz
        np.random.seed()
        Noise = np.random.poisson(0.000, params.N_CA3)
        if Noise_on:    
            if not ER:
                np.random.uniform()
                Noise = np.random.poisson(0.0035, params.N_CA3)

        network.solve(env, CA3input, Noise, ER, T, En_win, Re_win)

        # Just before finishing one cycle
        if n != len(En_st) and En_st[n] <= T < En_st[n] + Phase_win:
            In_FT_temp = []
            In_V_temp = []
            for i_ft in range(params.N_I):
                In_FT_temp.append(network.layers[0][i_ft].f_time)
                In_V_temp.append(network.layers[0][i_ft].V[1])
            In_FT[:,t] = In_FT_temp
            In_V[:,t] = In_V_temp

            Out_FT_temp = []
            Out_V_temp = []
            for o_ft in range(params.N_O):
                Out_FT_temp.append(network.layers[1][o_ft].f_time)
                Out_V_temp.append(network.layers[1][o_ft].V[1])
            Out_FT[:,t] = Out_FT_temp     
            Out_V[:,t] = Out_V_temp     

            H_FT_temp = []
            H_V_temp = []
            for h_ft in range(params.N_DGH):
                H_FT_temp.append(network.layers[2][h_ft].f_time)
                H_V_temp.append(network.layers[2][h_ft].V[1])
            H_FT[:,t] = H_FT_temp
            H_V[:,t] = H_V_temp

            M_FT_temp = []
            M_V_temp = []
            for m_ft in range(params.N_DGM):
                M_FT_temp.append(network.layers[3][m_ft].f_time)
                M_V_temp.append(network.layers[3][m_ft].V[1])
            M_FT[:,t] = M_FT_temp
            M_V[:,t] = M_V_temp

            B_FT_temp = []
            B_V_temp = []
            for b_ft in range(params.N_DGB):
                B_FT_temp.append(network.layers[4][b_ft].f_time)
                B_V_temp.append(network.layers[4][b_ft].V[1])
            B_FT[:,t] = B_FT_temp
            B_V[:,t] = B_V_temp

            DG_FT_temp = []
            DG_V_temp = []
            for g_ft in range(params.N_DG):
                DG_FT_temp.append(network.layers[5][g_ft].f_time)
                DG_V_temp.append(network.layers[5][g_ft].V[1])
            DG_FT[:,t] = DG_FT_temp
            DG_V[:,t] = DG_V_temp
                
            CA3_FT_temp = []
            CA3_V_temp = []
            for c3_ft in range(params.N_CA3):
                CA3_FT_temp.append(network.layers[6][c3_ft].f_time)
                CA3_V_temp.append(network.layers[6][c3_ft].V[1])
            CA3_FT[:,t] = CA3_FT_temp
            CA3_V[:,t] = CA3_V_temp

            CA3i_FT_temp = []
            CA3i_V_temp = []
            for c3i_ft in range(params.N_CA3i):
                CA3i_FT_temp.append(network.layers[7][c3i_ft].f_time)
                CA3i_V_temp.append(network.layers[7][c3i_ft].V[1])
            CA3i_FT[:,t] = CA3i_FT_temp
            CA3i_V[:,t] = CA3i_V_temp

            CA1_FT_temp = []
            CA1_V_temp = []
            for c1_ft in range(params.N_CA1):
                CA1_FT_temp.append(network.layers[8][c1_ft].f_time)
                CA1_V_temp.append(network.layers[8][c1_ft].V[1])
            CA1_FT[:,t] = CA1_FT_temp
            CA1_V[:,t] = CA1_V_temp
                
            t += 1
            
            if ER:                
                for I in range(len(WTS_I)):
                    V_E_I[I,Te] = network.layers[0][WTS_I[I]].V[1]
                    V_E_O[I,Te] = network.layers[1][WTS_I[I]].V[1]
                    Tex_E_I[I,Te] = network.layers[0][WTS_I[I]].total_ex
                    Tin_E_I[I,Te] = network.layers[0][WTS_I[I]].total_in
                    Tex_E_O[I,Te] = network.layers[1][WTS_I[I]].total_ex
                    Tin_E_O[I,Te] = network.layers[1][WTS_I[I]].total_in
                
                for H in range(len(WTS_DGH)):
                    V_E_DGH[H,Te] = network.layers[2][WTS_DGH[H]].V[1]

                for M in range(len(WTS_DGM)):
                    V_E_DGM[M,Te] = network.layers[3][WTS_DGM[M]].V[1]
                    Tex_E_DGM[M,Te] = network.layers[3][WTS_DGM[M]].total_ex
                    Tin_E_DGM[M,Te] = network.layers[3][WTS_DGM[M]].total_in

                for B in range(len(WTS_DGB)):
                    V_E_DGB[B,Te] = network.layers[4][WTS_DGB[B]].V[1]
                    Tex_E_DGB[B,Te] = network.layers[4][WTS_DGB[B]].total_ex
                    Tin_E_DGB[B,Te] = network.layers[4][WTS_DGB[B]].total_in

                for DG in range(len(WTS_DG)):
                    V_E_DG[DG,Te] = network.layers[5][WTS_DG[DG]].V[1]
                    Tex_E_DG[DG,Te] = network.layers[5][WTS_DG[DG]].total_ex
                    Tin_E_DG[DG,Te] = network.layers[5][WTS_DG[DG]].total_in
                
                for CA3 in range(len(WTS_CA3)):
                    V_E_CA3[CA3,Te] = network.layers[6][WTS_CA3[CA3]].V[1]
                    Tex_E_CA3[CA3,Te] = network.layers[6][WTS_CA3[CA3]].total_ex
                    Tin_E_CA3[CA3,Te] = network.layers[6][WTS_CA3[CA3]].total_in
                    
                for CA3i in range(len(WTS_CA3i)):
                    V_E_CA3i[CA3i,Te] = network.layers[7][WTS_CA3i[CA3i]].V[1]   
                    Tex_E_CA3i[CA3i,Te] = network.layers[7][WTS_CA3i[CA3i]].total_ex
                    Tin_E_CA3i[CA3i,Te] = network.layers[7][WTS_CA3i[CA3i]].total_in

                for CA1 in range(len(WTS_CA1)):
                    V_E_CA1[CA1,Te] = network.layers[8][WTS_CA1[CA1]].V[1]
                    Tex_E_CA1[CA1,Te] = network.layers[8][WTS_CA1[CA1]].total_ex
                    Tin_E_CA1[CA1,Te] = network.layers[8][WTS_CA1[CA1]].total_in
                Te += 1
            else:
                for i in range(params.N_CA3):                
                    CA3_R[i] = CA3_R[i] + network.layers[6][i].fire_record[1]
                for i in range(params.N_O):
                    Out_R[i] = Out_R[i] + network.layers[1][i].fire_record[1]  
                
                for I in range(len(WTS_I)):
                    V_R_I[I,Tr] = network.layers[0][WTS_I[I]].V[1]
                    V_R_O[I,Tr] = network.layers[1][WTS_I[I]].V[1]
                    Tex_R_I[I,Tr] = network.layers[0][WTS_I[I]].total_ex
                    Tin_R_I[I,Tr] = network.layers[0][WTS_I[I]].total_in
                    Tex_R_O[I,Tr] = network.layers[1][WTS_I[I]].total_ex
                    Tin_R_O[I,Tr] = network.layers[1][WTS_I[I]].total_in
                
                for H in range(len(WTS_DGH)):
                    V_R_DGH[H,Tr] = network.layers[2][WTS_DGH[H]].V[1]

                for M in range(len(WTS_DGM)):
                    V_R_DGM[M,Tr] = network.layers[3][WTS_DGM[M]].V[1]
                    Tex_R_DGM[M,Tr] = network.layers[3][WTS_DGM[M]].total_ex
                    Tin_R_DGM[M,Tr] = network.layers[3][WTS_DGM[M]].total_in

                for B in range(len(WTS_DGB)):
                    V_R_DGB[B,Tr] = network.layers[4][WTS_DGB[B]].V[1]
                    Tex_R_DGB[B,Tr] = network.layers[4][WTS_DGB[B]].total_ex
                    Tin_R_DGB[B,Tr] = network.layers[4][WTS_DGB[B]].total_in

                for DG in range(len(WTS_DG)):
                    V_R_DG[DG,Tr] = network.layers[5][WTS_DG[DG]].V[1]
                    Tex_R_DG[DG,Tr] = network.layers[5][WTS_DG[DG]].total_ex
                    Tin_R_DG[DG,Tr] = network.layers[5][WTS_DG[DG]].total_in

                for CA3 in range(len(WTS_CA3)):
                    V_R_CA3[CA3,Tr] = network.layers[6][WTS_CA3[CA3]].V[1]
                    Tpp_R_CA3[CA3,Tr] = network.layers[6][WTS_CA3[CA3]].total_pp
                    Trc_R_CA3[CA3,Tr] = network.layers[6][WTS_CA3[CA3]].total_Rc
                    Tno_R_CA3[CA3,Tr] = network.layers[6][WTS_CA3[CA3]].total_N
                    Tin_R_CA3[CA3,Tr] = network.layers[6][WTS_CA3[CA3]].total_in

                for CA3i in range(len(WTS_CA3i)):
                    V_R_CA3i[CA3i,Tr] = network.layers[7][WTS_CA3i[CA3i]].V[1]
                    Tex_R_CA3i[CA3i,Tr] = network.layers[7][WTS_CA3i[CA3i]].total_ex
                    Tin_R_CA3i[CA3i,Tr] = network.layers[7][WTS_CA3i[CA3i]].total_in                

                for CA1 in range(len(WTS_CA1)):
                    V_R_CA1[CA1,Tr] = network.layers[8][WTS_CA1[CA1]].V[1]
                    Tex_R_CA1[CA1,Tr] = network.layers[8][WTS_CA1[CA1]].total_ex
                    Tin_R_CA1[CA1,Tr] = network.layers[8][WTS_CA1[CA1]].total_in
                Tr += 1

        # After one cycle
        if n != len(En_st) and T == En_st[n] + Phase_win-1:
            # firing timing figure
            lineSize = list(np.full(params.N_I,0.5))
            plt.figure()
            plt.eventplot(In_FT, linelengths = lineSize)
            plt.xlim([0.0001+0.001*En_st[n], 0.001*T])
            plt.xticks(np.linspace(0.001*En_st[n],0.001*(En_st[n]+Phase_win),11))
            plt.grid(True)
            plt.xlabel('time(s)')
            plt.ylabel('Cell')
            plt.title('Input layer')
            plt.show()

            lineSize = list(np.full(params.N_O,0.5))
            plt.figure()
            plt.eventplot(Out_FT, linelengths = lineSize)
            plt.xlim([0.0001+0.001*En_st[n], 0.001*T])
            plt.xticks(np.linspace(0.001*En_st[n],0.001*(En_st[n]+Phase_win),11))
            plt.grid(True)
            plt.xlabel('time(s)')
            plt.ylabel('Cell')
            plt.title('Output layer')
            plt.show()

            lineSize = list(np.full(params.N_DG,1))
            plt.figure(figsize = (10,10))
            plt.eventplot(DG_FT, linelengths = lineSize, linewidths = 3)
            plt.xlim([0.0001+0.001*En_st[n], 0.001*T])
            plt.xticks(np.linspace(0.001*En_st[n],0.001*(En_st[n]+Phase_win),11))
            plt.grid(True)
            plt.xlabel('time(s)')
            plt.ylabel('Cell')
            plt.title('DG layer')
            plt.show()

            lineSize = list(np.full(params.N_CA3,2))
            plt.figure(figsize = (15,15))
            plt.eventplot(CA3_FT, linelengths = lineSize, linewidths = 3)
            plt.xlim([0.0001+0.001*En_st[n], 0.001*T])
            plt.xticks(np.linspace(0.001*En_st[n],0.001*(En_st[n]+Phase_win),11))
            plt.grid(True)
            plt.xlabel('time(s)')
            plt.ylabel('Cell')
            plt.title('CA3 layer')
            plt.show()

            lineSize = list(np.full(params.N_CA1,0.5))
            plt.figure()
            plt.eventplot(CA1_FT, linelengths = lineSize)
            plt.xlim([0.0001+0.001*En_st[n], 0.001*T])
            plt.xticks(np.linspace(0.001*En_st[n],0.001*(En_st[n]+Phase_win),11))
            plt.grid(True)
            plt.xlabel('time(s)')
            plt.ylabel('Cell')
            plt.title('CA1 layer')
            plt.show()
            
            In_V_list.append(In_V)
            Out_V_list.append(Out_V)
            
            H_V_list.append(H_V)
            M_V_list.append(M_V)
            B_V_list.append(B_V)
            DG_V_list.append(DG_V)
            
            CA3_V_list.append(CA3_V)
            CA3i_V_list.append(CA3i_V)
            CA1_V_list.append(CA1_V)
            
            In_FT_list.append(In_FT)
            Out_FT_list.append(Out_FT)
            
            H_FT_list.append(H_FT)
            M_FT_list.append(M_FT)
            B_FT_list.append(B_FT)
            DG_FT_list.append(DG_FT)
            
            CA3_FT_list.append(CA3_FT)
            CA3i_FT_list.append(CA3i_FT)
            CA1_FT_list.append(CA1_FT)

            CA3_R_list.append(CA3_R)
            Out_R_list.append(Out_R)
            
            # Plot voltages and currents for each neuron
            if plot_I:
                for I in range(len(WTS_I)):
                    plotV_ER(V_E_I[I,:], V_R_I[I,:], 'firing of a I_%d at input %d' % (WTS_I[I],I))
                    plotT_ER(Tex_E_I[I,:]+Tin_E_I[I,:], Tex_R_I[I,:]+Tin_R_I[I,:], 'total of a I_%d at input %d' % (WTS_I[I],I))
                    plotT_ER(Tex_E_I[I,:], Tex_R_I[I,:], 'total_ex of a I_%d at input %d' % (WTS_I[I],I))
                    plotT_ER(Tin_E_I[I,:], Tin_R_I[I,:], 'total_in of a I_%d at input %d' % (WTS_I[I],I))
                print('---------------------------------------------------------------------------------')
            if plot_O:
                for O in range(len(WTS_I)):
                    plotV_ER(V_E_O[O,:], V_R_O[O,:], 'firing of a O_%d at input %d' % (WTS_I[O],O))
                    plotT_ER(Tex_E_O[O,:]+Tin_E_O[O,:], Tex_R_O[O,:]+Tin_R_O[O,:], 'total of a O_%d at input %d' % (WTS_I[O],O))
                    plotT_ER(Tex_E_O[O,:], Tex_R_O[O,:], 'total_ex of a O_%d at input %d' % (WTS_I[O],O))
                    plotT_ER(Tin_E_O[O,:], Tin_R_O[O,:], 'total_in of a O_%d at input %d' % (WTS_I[O],O))
                print('---------------------------------------------------------------------------------')
            if plot_H:
                for H in range(len(WTS_DGH)):
                    plotV_ER(V_E_DGH[H,:], V_R_DGH[H,:], 'firing of a DGH_%d at input %d' % (WTS_DGH[H],H))
            print('---------------------------------------------------------------------------------')
            if plot_M:
                for M in range(len(WTS_DGM)):
                    plotV_ER(V_E_DGM[M,:], V_R_DGM[M,:], 'firing of a DGM_%d at input %d' % (WTS_DGM[M],M))
                    plotT_ER(Tex_E_DGM[M,:]+Tin_E_DGM[M,:], Tex_R_DGM[M,:]+Tin_R_DGM[M,:], 'total of a DGM_%d at input %d' % (WTS_DGM[M],M), 800)
                    plotT_ER(Tex_E_DGM[M,:], Tex_R_DGM[M,:], 'total_ex of a DGM_%d at input %d' % (WTS_DGM[M],M), 800)
                    plotT_ER(Tin_E_DGM[M,:], Tin_R_DGM[M,:], 'total_in of a DGM_%d at input %d' % (WTS_DGM[M],M), 800)
                print('---------------------------------------------------------------------------------')
            if plot_B:
                for B in range(len(WTS_DGB)):
                    plotV_ER(V_E_DGB[B,:], V_R_DGB[B,:], 'firing of a DGB_%d at input %d' % (WTS_DGB[B],B%4))
                    plotT_ER(Tex_E_DGB[B,:]+Tin_E_DGB[B,:], Tex_R_DGB[B,:]+Tin_R_DGB[B,:], 'total of a DGB_%d at input %d' % (WTS_DGB[B],B%4), 800)
                    plotT_ER(Tex_E_DGB[B,:], Tex_R_DGB[B,:], 'total_ex of a DGB_%d at input %d' % (WTS_DGB[B],B%4), 800)
                    plotT_ER(Tin_E_DGB[B,:], Tin_R_DGB[B,:], 'total_in of a DGB_%d at input %d' % (WTS_DGB[B],B%4), 800)
                print('---------------------------------------------------------------------------------')
            if plot_DG:
                for DG in range(len(WTS_DG)):
                    plotV_ER(V_E_DG[DG,:], V_R_DG[DG,:], 'firing of a DG_%d at input %d' % (WTS_DG[DG],DG%4))
                    plotT_ER(Tex_E_DG[DG,:]+Tin_E_DG[DG,:], Tex_R_DG[DG,:]+Tin_R_DG[DG,:], 'total of a DG_%d at input %d' % (WTS_DG[DG],DG%4), 800)
                    plotT_ER(Tex_E_DG[DG,:], Tex_R_DG[DG,:], 'total_ex of a DG_%d at input %d' % (WTS_DG[DG],DG%4), 800)
                    plotT_ER(Tin_E_DG[DG,:], Tin_R_DG[DG,:], 'total_in of a DG_%d at input %d' % (WTS_DG[DG],DG%4), 800)
                print('---------------------------------------------------------------------------------')
            if plot_CA3:
                for CA3 in range(len(WTS_CA3)):
                    plotV_ER(V_E_CA3[CA3,:], V_R_CA3[CA3,:], 'firing of a CA3_%d at input %d' % (WTS_CA3[CA3],CA3//13))
                    plotT_ER(Tex_E_CA3[CA3,:]+Tin_E_CA3[CA3,:], Tpp_R_CA3[CA3,:]+Trc_R_CA3[CA3,:]+Tno_R_CA3[CA3,:]+Tin_R_CA3[CA3,:], 'total of a CA3_%d at input %d' % (WTS_CA3[CA3],CA3//13), 800)
                    plotT_ER(Tex_E_CA3[CA3,:], Tpp_R_CA3[CA3,:], 'pp_current of a CA3_%d at input %d' % (WTS_CA3[CA3],CA3//13), 800)
                    plotT_ER(Tex_E_CA3[CA3,:], Trc_R_CA3[CA3,:], 'rc_current of a CA3_%d at input %d' % (WTS_CA3[CA3],CA3//13), 800)
                    plotT_ER(Tex_E_CA3[CA3,:], Tno_R_CA3[CA3,:], 'noise_current of a CA3_%d at input %d' % (WTS_CA3[CA3],CA3//13), 800)
                    plotT_ER(Tin_E_CA3[CA3,:], Tin_R_CA3[CA3,:], 'in_current of a CA3_%d at input %d' % (WTS_CA3[CA3],CA3//13), 800)
                print('---------------------------------------------------------------------------------')    
            if plot_CA3i:
                for CA3i in range(len(WTS_CA3i)):
                    plotV_ER(V_E_CA3i[CA3i,:], V_R_CA3i[CA3i,:], 'firing of a CA3i_%d at input %d' % (WTS_CA3i[CA3i],CA3i//3))
                    plotT_ER(Tex_E_CA3i[CA3i,:]+Tin_E_CA3i[CA3i,:], Tex_R_CA3i[CA3i,:]+Tin_R_CA3i[CA3i,:], 'total of a CA3i_%d at input %d' % (WTS_CA3i[CA3i],CA3i//3), 600)
                    plotT_ER(Tex_E_CA3i[CA3i,:], Tex_R_CA3i[CA3i,:], 'ex_current of a CA3i_%d at input %d' % (WTS_CA3i[CA3i],CA3i//3), 600)
                    plotT_ER(Tin_E_CA3i[CA3i,:], Tin_R_CA3i[CA3i,:], 'in_current of a CA3i_%d at input %d' % (WTS_CA3i[CA3i],CA3i//3), 600)
                print('---------------------------------------------------------------------------------')
            if plot_CA1:
                for CA1 in range(len(WTS_CA1)):
                    plotV_ER(V_E_CA1[CA1,:], V_R_CA1[CA1,:], 'firing of a CA1_%d at input %d' % (WTS_CA1[CA1],CA1))
                    plotT_ER(Tex_E_CA1[CA1,:]+Tin_E_CA1[CA1,:], Tex_R_CA1[CA1,:]+Tin_R_CA1[CA1,:], 'total of a CA1_%d at input %d' % (WTS_CA1[CA1],CA1), 600)
                    plotT_ER(Tex_E_CA1[CA1,:], Tex_R_CA1[CA1,:], 'total_ex of a CA1_%d at input %d' % (WTS_CA1[CA1],CA1), 600)
                    plotT_ER(Tin_E_CA1[CA1,:], Tin_R_CA1[CA1,:], 'total_in of a CA1_%d at input %d' % (WTS_CA1[CA1],CA1), 600)
            n += 1
            
    FT = [In_FT_list, Out_FT_list, H_FT_list, M_FT_list, B_FT_list, DG_FT_list, CA3_FT_list, CA3i_FT_list, CA1_FT_list]
    V = [In_V_list, Out_V_list, H_V_list, M_V_list, B_V_list, DG_V_list, CA3_V_list, CA3i_V_list, CA1_V_list]
    print('Success : ', Success)
    print('Fail : ', Fail)
    print('If_fail : ', If_fail)
    print('Output bias (mean, std) : ', np.mean(Winfin), np.std(Winfin)) 

    return network, Fail, If_fail, Success, Winfin, V, FT, CA3_R_list, Out_R_list
# In[ ]:




