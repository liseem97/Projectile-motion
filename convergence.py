#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:43:28 2018

@author: auroragrefsrud
"""

import numpy as np
import matplotlib.pyplot as plt

g=9.81
C = -g


#Initial conditions
X0 = 0
Y0 = 0
theta = np.deg2rad(45)
V=700
U0 = V*np.cos(theta)
V0 = V*np.sin(theta)
print(U0," ",V0)

t_min = 0
t_max = 120
dt = 0.01             #time step / tau
N = int(t_max/dt)


#RUNGE-KUTTA (4th order)
def F(X_, Y_, U_, V_):          #dX/dt
    return U_

def G(X_, Y_, U_, V_):          #dY/dt
    return V_

def H(X_, Y_, U_, V_):          #dU/dt
    return 0

def I(X_, Y_, U_, V_):          #dV/dt
    return C

def RK(X0, Y0, U0, V0, t_min, t_max, tau):    
    dt_RK = tau
    N_RK = int(t_max/dt_RK)
    t_RK = np.linspace(t_min, t_max, N_RK)
    
    X_RK = np.zeros(N_RK)   #Position x
    Y_RK = np.zeros(N_RK)   #Position y
    U_RK = np.zeros(N_RK)   #Velocity x
    V_RK = np.zeros(N_RK)   #Velocity y
    E_RK = np.zeros(N_RK)   #Total energy/mass
    Ek_RK = np.zeros(N_RK)  #Kinetic energy
    Ep_RK = np.zeros(N_RK)  #Potential energy

    
    X_RK[0] = X0
    Y_RK[0] = Y0
    U_RK[0] = U0
    V_RK[0] = V0
    Ek_RK[0] = 0.5 * np.sqrt(U_RK[0]**2 + V_RK[0]**2)**2
    Ep_RK[0] = 0
    E_RK[0] = Ek_RK[0] + Ep_RK[0]

    
    for n in range(N_RK-1):
        k_x1 = dt_RK * F( X_RK[n], Y_RK[n], U_RK[n], V_RK[n] )
        k_y1 = dt_RK * G( X_RK[n], Y_RK[n], U_RK[n], V_RK[n] )
        k_u1 = dt_RK * H( X_RK[n], Y_RK[n], U_RK[n], V_RK[n] )
        k_v1 = dt_RK * I( X_RK[n], Y_RK[n], U_RK[n], V_RK[n] )
        
        k_x2 = dt_RK * F( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 )
        k_y2 = dt_RK * G( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 )
        k_u2 = dt_RK * H( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 )
        k_v2 = dt_RK * I( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 )
        
        k_x3 = dt_RK * F( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 )
        k_y3 = dt_RK * G( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 )
        k_u3 = dt_RK * H( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 )
        k_v3 = dt_RK * I( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 )
        
        k_x4 = dt_RK * F( X_RK[n] + k_x3, Y_RK[n] + k_y3, U_RK[n] + k_u3, V_RK[n] + k_v3 )
        k_y4 = dt_RK * G( X_RK[n] + k_x3, Y_RK[n] + k_y3, U_RK[n] + k_u3, V_RK[n] + k_v3 )
        k_u4 = dt_RK * H( X_RK[n] + k_x3, Y_RK[n] + k_y3, U_RK[n] + k_u3, V_RK[n] + k_v3 )
        k_v4 = dt_RK * I( X_RK[n] + k_x3, Y_RK[n] + k_y3, U_RK[n] + k_u3, V_RK[n] + k_v3 )
        
        X_RK[n+1] = X_RK[n] + k_x1/6 + k_x2/3 + k_x3/3 + k_x4/6
        Y_RK[n+1] = Y_RK[n] + k_y1/6 + k_y2/3 + k_y3/3 + k_y4/6
        U_RK[n+1] = U_RK[n] + k_u1/6 + k_u2/3 + k_u3/3 + k_u4/6
        V_RK[n+1] = V_RK[n] + k_v1/6 + k_v2/3 + k_v3/3 + k_v4/6

        Ek_RK[n+1] = 0.5 * np.sqrt(U_RK[n+1]**2 + V_RK[n+1]**2)**2
        Ep_RK[n+1] = -C*Y_RK[n+1]
        E_RK[n+1] = Ek_RK[n+1] + Ep_RK[n+1]
        
    
    return X_RK, Y_RK, U_RK, V_RK, E_RK, Ek_RK, Ep_RK, t_RK


#analytical solution
def analytical(X0, Y0, U0, V0, times):
    X_A = np.zeros(len(times))
    Y_A = np.zeros(len(times))
    U_A = np.zeros(len(times))
    V_A = np.zeros(len(times))
    Ek_A  = np.zeros(len(times))
    Ep_A = np.zeros(len(times))
    E_A = np.zeros(len(times))
    i = 0
    for i in range(0, len(times)):
        Y_A[i] = -0.5*g*times[i]**2 + times[i]*V0 + Y0
        X_A[i] = V0*times[i] + X0
        V_A[i] = C*times[i] +U0
        U_A[i] = V0
        Ek_A[i] = 0.5 * np.sqrt(U_A[i]**2 + V_A[i]**2)**2
        Ep_A[i] = -C*Y_A[i]
        E_A[i] = Ek_A[i]+Ep_A[i]
    return X_A, Y_A, U_A, V_A, E_A
   

###code to run
timesteps = np.linspace(0.0001,0.1,100)
positions = np.zeros(len(timesteps))
for i in range (len(timesteps)):
    print(i)
    dt = timesteps[i]
    RK_test = RK(X0, Y0, U0, V0, t_min, t_max, dt) 
    AN = analytical(X0, Y0, U0, V0, RK_test[-1])
    positions[i] = abs(RK_test[0][-1]-AN[0][-1])
    #print("time: ", RK_test[-1][-1], " y-position: ", RK_test[1][-1], " and ", AN[1][-1])
    
#fig = plt.figure(figsize=(8, 2))
#plt.title("Energy as a function of time")
#plt.plot(RK_test[-1], RK_test[4]/1000, label= "Total energy", color = "crimson")
#plt.plot(RK_test[-1], RK_test[5]/1000, label = "Kinetic energy", color = "blue")
#plt.plot(RK_test[-1], RK_test[6]/1000, label= "Potential energy", color = "orange")
#plt.xlabel(r"time [s]")
#plt.ylabel(r"$Etot$ [kJ / kg]")
#plt.legend(loc = 4)
#plt.grid()
##plt.savefig("/Users/elveb/Documents/1_Ep.pdf")
#plt.show()

plt.figure()
plt.title("Difference in x-position compared to analytical solution")
plt.plot(timesteps, positions)
plt.xlabel(r"$dt$ [s]")
plt.ylabel(r"$difference$ [m]")
plt.grid()
#plt.savefig("/Users/elveb/Documents/1_RK_pos.pdf")
plt.show()
