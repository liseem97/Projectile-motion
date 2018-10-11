# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:09:59 2018

@author: lise
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



g = 9.81 #m/s^2
C = -g
m1 = 50 #kg
m2 =  106 #kg
B2_m1 = 4*10**(-5) #m^-1
B2_m2 = (B2_m1 * m1 )/ m2 #m^-1

a = 6.5 * 10**(-3) #K/m
alpha = 2.5 #air 
T_0 = 288.15 #K (tilsvarer 15 grader celcius)



#Initial conditions
#Earth
R = 6371*10**3

#Crépy
Nc = np.deg2rad(49.60500)
Ec = np.deg2rad(3.514722)

Xc = R*np.cos(Nc)*np.cos(Ec)
Yc = R*np.cos(Nc)*np.sin(Ec)
Zc = R*np.sin(Nc)

print("Crepy: ", Xc/1000, Yc/1000, Zc/1000)
Lc= np.sqrt(Xc**2+Yc**2+Zc**2)

#Paris
Np = np.deg2rad(48.5667)
Ep = np.deg2rad(2.35083)

Xp = R*np.cos(Np)*np.cos(Ep)
Yp = R*np.cos(Np)*np.sin(Ep)
Zp = R*np.sin(Np)

print("Paris: ", Xp/1000, Yp/1000, Zp/1000)

Lp = np.sqrt(Xp**2+Yp**2+Zp**2)

beta = np.arccos((Xc*Xp+Yc*Yp+Zc*Zp)/(Lc*Lp)) 
d = beta * R
print("Distance between places: ", d/1000 )

theta = beta/2+np.pi/2 #Har samme som adiabatisk modell
phi = np.deg2rad(40)

V_start = 1640
U0 = V_start*np.cos(theta)*np.sin(phi)
V0 = V_start*np.sin(theta)*np.sin(phi)
W0 = V_start*np.cos(phi)

print("starting velocity: ", U0, V0, W0, " velocity: ", np.sqrt(U0**2+V0**2+W0**2))

print("Starting position: ", np.sqrt(Xc**2+Yc**2+Zc**2))


t_min = 0
t_max = 4000
dt = 0.1             #time step / tau
N = int(t_max/dt)


#RUNGE-KUTTA (4th order)
def E(X_,Y_,Z_,U_,V_,W_):          #dX/dt
    return U_

def F(X_,Y_,Z_,U_,V_,W_):          #dY/dt
    return V_

def G(X_,Y_,Z_,U_,V_,W_):           #dZ/dt
    return W_

def H(X_,Y_,Z_,U_,V_,W_):          #dU/dt
    V = (U_**2+V_**2+W_**2)**(1/2)
    h=np.sqrt(X_**2+Y_**2+Z_**2)-R
    AD = (1 -((a*h)/T_0))**alpha #airdensity adiabatic
    return C -B2_m2*U_*V*AD

def I(X_,Y_,Z_,U_,V_,W_):          #dV/dt
    V = (U_**2+V_**2+W_**2)**(1/2)
    h=np.sqrt(X_**2+Y_**2+Z_**2)-R
    AD = (1 - ((a*h)/T_0))**alpha #airdensity adiabatic
    return C -B2_m2*V_*V*AD

def J(X_,Y_,Z_,U_,V_,W_):           #dW/dt
    V = (U_**2+V_**2+W_**2)**(1/2)
    h=np.sqrt(X_**2+Y_**2+Z_**2)-R
    AD = (1 - ((a*h)/T_0))**alpha
    return C -B2_m2*W_*V*AD

def RK(X0, Y0, Z0, U0, V0, W0, t_min, t_max, tau):  
    dt_RK = tau
    N_RK = int(t_max/dt_RK)
    t_RK = np.linspace(t_min, t_max, N_RK)
    x_l=0
    y_l=0
    t_l=0
    
    X_RK = np.zeros(N_RK)   #Position x
    Y_RK = np.zeros(N_RK)   #Position y
    Z_RK = np.zeros(N_RK)   #Position z
    U_RK = np.zeros(N_RK)   #Velocity x
    V_RK = np.zeros(N_RK)   #Velocity y
    W_RK = np.zeros(N_RK)   #Velocity z
    
    X_RK[0] = X0
    Y_RK[0] = Y0
    Z_RK[0] = Z0
    U_RK[0] = U0
    V_RK[0] = V0
    W_RK[0] = W0

    
    for n in range(N_RK-1):
        k_x1 = dt_RK * E( X_RK[n], Y_RK[n], Z_RK[n], U_RK[n], V_RK[n] , W_RK[n] )
        k_y1 = dt_RK * F(  X_RK[n], Y_RK[n], Z_RK[n], U_RK[n], V_RK[n] , W_RK[n] )
        k_z1 = dt_RK * G(  X_RK[n], Y_RK[n], Z_RK[n], U_RK[n], V_RK[n] , W_RK[n] )
        k_u1 = dt_RK * H( X_RK[n], Y_RK[n], Z_RK[n], U_RK[n], V_RK[n] , W_RK[n] )
        k_v1 = dt_RK * I( X_RK[n], Y_RK[n], Z_RK[n], U_RK[n], V_RK[n] , W_RK[n] )
        k_w1 = dt_RK * J(  X_RK[n], Y_RK[n], Z_RK[n], U_RK[n], V_RK[n] , W_RK[n] )
        
        
        k_x2 = dt_RK * E( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, Z_RK[n] + k_z1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 , W_RK[n] + k_w1/2 )
        k_y2 = dt_RK * F( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, Z_RK[n] + k_z1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 , W_RK[n] + k_w1/2 )
        k_z2 = dt_RK * G( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, Z_RK[n] + k_z1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 , W_RK[n] + k_w1/2 )
        k_u2 = dt_RK * H( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, Z_RK[n] + k_z1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 , W_RK[n] + k_w1/2 )
        k_v2 = dt_RK * I( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, Z_RK[n] + k_z1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 , W_RK[n] + k_w1/2 )
        k_w2 = dt_RK * J( X_RK[n] + k_x1/2, Y_RK[n] + k_y1/2, Z_RK[n] + k_z1/2, U_RK[n] + k_u1/2, V_RK[n] + k_v1/2 , W_RK[n] + k_w1/2 )
        
        
        k_x3 = dt_RK * E( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, Z_RK[n] + k_z2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 , W_RK[n] + k_w2/2 )
        k_y3 = dt_RK * F( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, Z_RK[n] + k_z2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 , W_RK[n] + k_w2/2 )
        k_z3 = dt_RK * G( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, Z_RK[n] + k_z2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 , W_RK[n] + k_w2/2 )
        k_u3 = dt_RK * H( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, Z_RK[n] + k_z2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 , W_RK[n] + k_w2/2 )
        k_v3 = dt_RK * I( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, Z_RK[n] + k_z2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 , W_RK[n] + k_w2/2 )
        k_w3 = dt_RK * J( X_RK[n] + k_x2/2, Y_RK[n] + k_y2/2, Z_RK[n] + k_z2/2, U_RK[n] + k_u2/2, V_RK[n] + k_v2/2 , W_RK[n] + k_w2/2 )
        
        k_x4 = dt_RK * E( X_RK[n] + k_x3, Y_RK[n] + k_y3, Z_RK[n] + k_z3, U_RK[n] + k_u3, V_RK[n] + k_v3, W_RK[n] + k_w3 )
        k_y4 = dt_RK * F( X_RK[n] + k_x3, Y_RK[n] + k_y3, Z_RK[n] + k_z3, U_RK[n] + k_u3, V_RK[n] + k_v3, W_RK[n] + k_w3 )
        k_z4 = dt_RK * G( X_RK[n] + k_x3, Y_RK[n] + k_y3, Z_RK[n] + k_z3, U_RK[n] + k_u3, V_RK[n] + k_v3, W_RK[n] + k_w3 )
        k_u4 = dt_RK * H( X_RK[n] + k_x3, Y_RK[n] + k_y3, Z_RK[n] + k_z3, U_RK[n] + k_u3, V_RK[n] + k_v3, W_RK[n] + k_w3 )
        k_v4 = dt_RK * I( X_RK[n] + k_x3, Y_RK[n] + k_y3, Z_RK[n] + k_z3, U_RK[n] + k_u3, V_RK[n] + k_v3, W_RK[n] + k_w3 )
        k_w4 = dt_RK * J( X_RK[n] + k_x3, Y_RK[n] + k_y3, Z_RK[n] + k_z3, U_RK[n] + k_u3, V_RK[n] + k_v3, W_RK[n] + k_w3 )
        
        X_RK[n+1] = X_RK[n] + k_x1/6 + k_x2/3 + k_x3/3 + k_x4/6
        Y_RK[n+1] = Y_RK[n] + k_y1/6 + k_y2/3 + k_y3/3 + k_y4/6
        Z_RK[n+1] = Z_RK[n] + k_z1/6 + k_z2/3 + k_z3/3 + k_z4/6
        
        U_RK[n+1] = U_RK[n] + k_u1/6 + k_u2/3 + k_u3/3 + k_u4/6
        V_RK[n+1] = V_RK[n] + k_v1/6 + k_v2/3 + k_v3/3 + k_v4/6
        W_RK[n+1] = W_RK[n] + k_w1/6 + k_w2/3 + k_w3/3 + k_w4/6
        
        
        if np.sqrt(X_RK[n+1]**2+Y_RK[n+1]**2+Z_RK[n+1]**2) < R and np.sqrt(X_RK[n]**2+Y_RK[n]**2+Z_RK[n]**2) > R:
            r = - np.sqrt(X_RK[n]**2+Y_RK[n]**2+Z_RK[n]**2) / np.sqrt(X_RK[n+1]**2+Y_RK[n+1]**2+Z_RK[n+1]**2)
            x_l = (X_RK[n] + r*X_RK[n+1])/(r + 1)
            y_l = (Y_RK[n] + r*Y_RK[n+1])/(r + 1)
            t_l = (t_RK[n] + t_RK[n+1])/2
        
    
    return X_RK, Y_RK, Z_RK, U_RK, V_RK, W_RK, x_l, y_l, t_RK, t_l


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
    

     
RK = RK(Xc, Yc, Zc, U0, V0, W0, t_min, t_max, dt) 

print("landing position", RK[6]/1000,RK[7]/1000, " last position", RK[0][1], RK[1][1])
#AN = analytical(X0, Y0, U0, V0, RK[5])

#y_best = np.amax(RK[1])
#
#
#print("Landingpoint: ", RK[4], "m ")
#print("Time of fligth: ", RK[6], "s ")
#print("Best height: ", y_best , "m ")

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))



fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot3D(RK[0]/1000, RK[1]/1000, RK[2]/1000, label='path')
#ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='green')
ax.scatter3D(Xc/1000,Yc/1000,Zc/1000, color = "darkorange", marker = "o")
ax.scatter3D(Xp/1000,Yp/1000,Zp/1000, color = "red", marker = "o")
#ax.scatter3D(RK[6]/1000,RK[7]/1000,R/1000, color = "green", marker = "o")
#ax.axis([4100000,4150000, 250000,35000])
ax.legend()
plt.show()


##RK   
#plt.figure()
#plt.title("Projectile path of Paris Gun")
#plt.plot(RK[0]/1000, RK[1]/1000, color = "darkblue")
##plt.plot(AN[0], AN[1], color = "red", label = "Analytical path")
##plt.plot(RK[4], [0], color = "darkorange", marker = "o", markersize = 5)
#plt.legend()
#plt.xlabel(r"$x$ [km]")
#plt.ylabel(r"$z$ [km]")
#plt.axis([0,150,0,40])
#plt.grid()
#plt.savefig("pathBB.pdf")
#plt.show()
#
#plt.figure()
#plt.title("Projectile height as function of time")
#plt.plot(RK[5], RK[1]/1000, color = "darkblue")
##plt.plot(AN[0], AN[1], color = "red", label = "Analytical path")
#plt.legend()
#plt.xlabel(r"$time$ [s]")
#plt.ylabel(r"$z$ [km]")
#plt.axis([0,180,0,40])
#plt.grid()
#plt.savefig("heighttimeBB.pdf")
#plt.show()
#
##plt.figure()
##plt.title("Velocity")
##plt.plot(RK[5], RK[2], color = "darkblue", label = "X-velocity")
##plt.plot(RK[5], RK[3], color = "red", label = "Y-velocity")
###plt.plot([0], [0], color = "darkorange", marker = "o", markersize = 19, label = "Fixed sun")
##plt.legend(loc = 4)
##plt.xlabel(r"$x$ [s]")
##plt.ylabel(r"$y$ [m/s]")
###plt.axis("equal")
##plt.grid()
###plt.savefig("/Users/elveb/Documents/1_RK_pos.pdf")
##plt.show()
#
