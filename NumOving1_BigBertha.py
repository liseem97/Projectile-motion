# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 21:46:37 2018

@author: lise
"""


import numpy as np
import matplotlib.pyplot as plt

#eneste endring fra adiabatic er V_start

g = 9.81 #m/s^2
C = -g
m1 = 50 #kg
m2 =  820 #106 #kg
B2_m1 = 4*10**(-5) #m^-1
B2_m2 = (B2_m1 * m1 )/ m2 #m^-1

a = 6.5 * 10**(-3) #K/m
alpha = 2.5 #air 
T_0 = 288.15 #K (tilsvarer 15 grader celcius)



#Initial conditions
X0 = 0
Y0 = 0
theta = np.deg2rad(43.75) #Har samme som adiabatisk modell
V_start = 400 #1640
U0 = V_start*np.cos(theta)
V0 = V_start*np.sin(theta)
print(U0," ",V0)

t_min = 0
t_max = 200
dt = 0.1             #time step / tau
N = int(t_max/dt)


#RUNGE-KUTTA (4th order)
def F(X_, Y_, U_, V_):          #dX/dt
    return U_

def G(X_, Y_, U_, V_):          #dY/dt
    return V_

def H(X_, Y_, U_, V_):          #dU/dt
    V = (U_**2+V_**2)**(1/2)
    AD = (1 - ((a*Y_)/T_0))**alpha #airdensity adiabatic
    return -B2_m2*AD*U_*V

def I(X_, Y_, U_, V_):          #dV/dt
    V = (U_**2+V_**2)**(1/2)
    AD = (1 - ((a*Y_)/T_0))**alpha #airdensity adiabatic
    return C - B2_m2*AD*V_*V

def RK(X0, Y0, U0, V0, t_min, t_max, tau):    
    dt_RK = tau
    N_RK = int(t_max/dt_RK)
    t_RK = np.linspace(t_min, t_max, N_RK)
    
    X_RK = np.zeros(N_RK)   #Position x
    Y_RK = np.zeros(N_RK)   #Position y
    U_RK = np.zeros(N_RK)   #Velocity x
    V_RK = np.zeros(N_RK)   #Velocity y

    
    X_RK[0] = X0
    Y_RK[0] = Y0
    U_RK[0] = U0
    V_RK[0] = V0

    
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
        
        if Y_RK[n+1] < 0 and Y_RK[n] > 0:
            r = - Y_RK[n] / Y_RK[n+1]
            x_l = (X_RK[n] + r*X_RK[n+1])/(r + 1)
            t_l = (t_RK[n] + t_RK[n+1])/2
        
    
    return X_RK, Y_RK, U_RK, V_RK, x_l, t_RK, t_l


#analytical solution OBS DENNE ER UTEN DRAG OG LUFTFUKTIGHET
#def analytical(X0, Y0, U0, V0, times):
#    X_A = np.zeros(len(times))
#    Y_A = np.zeros(len(times))
#    i = 0
#    for i in range(0, len(times)):
#        Y_A[i] = -0.5*g*times[i]**2 + times[i]*V0 + Y0
#        X_A[i] = V0*times[i] + X0
#        
#    return X_A, Y_A
   

###code to run
     
RK = RK(X0, Y0, U0, V0, t_min, t_max, dt) 
#AN = analytical(X0, Y0, U0, V0, RK[5])

y_best = np.amax(RK[1])


print("Landingpoint: ", RK[4], "m ")
print("Time of fligth: ", RK[6], "s ")
print("Best height: ", y_best , "m ")

#RK   
plt.figure()
plt.title("Projectile path of Big Bertha")
plt.plot(RK[0]/1000, RK[1]/1000, color = "darkblue")
#plt.plot(AN[0], AN[1], color = "red", label = "Analytical path")
#plt.plot(RK[4], [0], color = "darkorange", marker = "o", markersize = 5)
plt.legend()
plt.xlabel(r"$x$ [km]")
plt.ylabel(r"$y$ [km]")
#plt.axis([0,150,0,40])
plt.grid()
#plt.savefig("pathBB.pdf")
plt.show()

plt.figure()
plt.title("Projectile height as function of time")
plt.plot(RK[5], RK[1]/1000, color = "darkblue")
#plt.plot(AN[0], AN[1], color = "red", label = "Analytical path")
plt.legend()
plt.xlabel(r"$time$ [s]")
plt.ylabel(r"$y$ [km]")
#plt.axis([0,180,0,40])
plt.grid()
#plt.savefig("heighttimeBB.pdf")
plt.show()

#plt.figure()
#plt.title("Velocity")
#plt.plot(RK[5], RK[2], color = "darkblue", label = "X-velocity")
#plt.plot(RK[5], RK[3], color = "red", label = "Y-velocity")
##plt.plot([0], [0], color = "darkorange", marker = "o", markersize = 19, label = "Fixed sun")
#plt.legend(loc = 4)
#plt.xlabel(r"$x$ [s]")
#plt.ylabel(r"$y$ [m/s]")
##plt.axis("equal")
#plt.grid()
##plt.savefig("/Users/elveb/Documents/1_RK_pos.pdf")
#plt.show()

