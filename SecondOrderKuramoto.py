# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:11:19 2018

@author: Joseph Wild
"""

# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


'''User Input Values'''
'''The first 55 lines of the script are for the user to input values for the network of Kuramoto Oscillators.
   The values need to be hard coded into the simulator, unless changes are made to allow an external file of 
   values to be read in. 
   
   The script is currently set up for a 2-bus, load and generator system, with arbitrary values that couple
   For different plots, change titles and axes names in lines 199-122, and lines 137-140
   '''
   

NodeNumber = 2          ##Numer of Nodes in Network

edges = [(0,1)]         ##Edges of the form (node1,node2) if a uniform coupling strength is used,
                        ##or (node1,node2,couplingstrength12) if setting individual coupling strengths between nodes

tstop = 10          ##Setting Length of Solution, with solutions plotted at least every 1ms
if tstop < 1:
    increments = 1000
else:
    increments - 1000*tstop
t = np.linspace(0,tstop, increments) 

CouplingStrength = 55  ##General coupling strength. Remove if using individual strengths between nodes

'''For power network scenarios, following Dorfler et al. (2012) "Synchronization in Complex Oscillator Networks
    and Smart Grids", set K = |V(1)||V(2)|/X(12) for node voltages and line reactance'''

DGen = 1                ##Damping Coefficients for Generators and Loads
DLoad = 0.1

Inertia = 1             ##Common Inertia Term (M) for Generators

Generators = [0]        ##Defining Which Nodes are Generators

ThetaInit = [0,0]      ##Initial State as ThetaInit = [Theta0, Theta1,...,ThetaN-1], FreqInit = [Freq0, Freq1,..., FreqN-1]
FreqInit = [50,-50]    ##Match Initial Power

NodePlots = [0,1]       ##Define Which Nodes are Desired for Plotting

PowerInput = [50,-50] ##Natural Frequency Term for System. Replaced with Power Input for Power Network
'''This "PowerInput" term takes the place of the "Natural Frequency" term in the Second-Order Kuramoto Model'''

 

'''Simulation Code'''
'''The code below takes the user inputs above and simulates the Second-Order Kuramoto Model for such values. '''

'''Combine Initial Conds for ODE Solver'''
def initconditions(ThetaInit, FreqInit):
    y0 = []
    for i in ThetaInit:
        y0.append(i)
    for i in FreqInit:
        y0.append(i)  
    return y0

'''Create Coupling Matrix to Determine the Summation. If individual coupling strengths are desired between nodes
    replace the "CouplingStrength" terms with "i[2]" instead'''
def summationterms(edges,CouplingStrength):
    Matrix = np.zeros((NodeNumber,NodeNumber))
    for i in edges:
        Matrix[i[0]][i[1]] += CouplingStrength
        Matrix[i[1]][i[0]] += CouplingStrength
    return Matrix

'''Build Each Summation Equation - Called During the Solver for Each Iteration'''
def Summation(y,Matrix,equation):
    SummationTerm = 0
    Node = 0
    for i in Matrix[equation]:
        SummationTerm = SummationTerm + i*np.sin(y[equation]-y[Node])
        Node = Node + 1
    return SummationTerm    
    
'''Kuramoto Equations'''
def kuramoto(y,t,NodeNumber, DGen, DLoad, PowerInput, Generators, Inertia, Matrix):
    derivatives = []    
    ddydt = np.zeros((NodeNumber))
    dydt = np.zeros((NodeNumber))
    for equation in range(NodeNumber):
        SummationTerm = Summation(y,Matrix,equation)
        if equation in Generators: 
            '''Second-Order Term for Generators Only'''
            ddydt[equation] = (-DGen*y[NodeNumber+equation] + PowerInput[equation] - SummationTerm)/Inertia
            '''First-Order Term Passes Previous, Updated in Next Time Instance From Second-Order Term'''
            derivatives.append(y[NodeNumber+equation]) 
        else: 
            '''First-Order Terms for Loads Only'''
            dydt[equation] = (PowerInput[equation] - SummationTerm)/DLoad
            '''First-Order Term Added'''
            derivatives.append(dydt[equation])
    for i in range(NodeNumber):
                derivatives.append(ddydt[i])
    return derivatives                                           ###return is form [dydt0,...,dydtN-1,ddydt0,...ddydtN-1]

'''Plotting Outputs'''       
def plotting(NodePlots, sol, increments,NodeNumber,y0,tstop):
    PlotColours = ['r','b','g','m','y','c','k']             ###different colours for multiple plots at once, can be changed to allow more 
    for i in NodePlots:
        Freq = []
        Freq.append(y0[i+NodeNumber])
        for x in range(0, increments-2, 1):
            Freq.append((sol[x+1][i]-sol[x][i])*(increments/tstop))
        plt.plot(t,Freq, color=PlotColours[i%7]) 
        plt.title('Rate of Change Relative to Each Other')  #Frequency of oscillators for non-power network
        plt.gca().legend(('Generator','Load'))              #Change these for plots
        plt.ylabel('Relative Change per Time')
        plt.xlabel('Time (s)')        

'''Initial Terms in Desired Form [Theta0,...,ThetaN-1, Omega0,...,OmegaN-1]'''
y0 = initconditions(ThetaInit, FreqInit)
   
'''Create Matrix from Edges Given '''
Matrix = summationterms(edges,CouplingStrength)

'''Solve the ODE at each Iteration Time'''
sol = odeint(kuramoto,y0,t, args = (NodeNumber, DGen, DLoad, PowerInput, Generators, Inertia, Matrix))

'''Plot Angles if Desired'''
for i in range(NodeNumber):
    PlotColours = ['g','m','r','b']
    plt.plot(t,sol[:,i], color=PlotColours[i])
    plt.title('Angles')
    plt.gca().legend(('Generator','Load'))
    plt.ylabel('Relative Change per Time')
    plt.xlabel('Time (s)') 
plt.show()  

'''Plot Frequencies'''
t = t[1:len(t)]
plotting(NodePlots, sol, increments,NodeNumber,y0,tstop)
plt.show()
