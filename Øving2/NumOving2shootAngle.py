# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:09:59 2018

@author: lise
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

g = -9.81 #m/s^2
m1 = 50 #kg
m2 =  106 #kg
B2_m1 = 4*10**(-5) #m^-1
B2_m2 = (B2_m1 * m1 )/ m2 #m^-1

a = 6.5 * 10**(-3) #K/m
alpha = 2.5 #air 
T_0 = 288.15 #K (tilsvarer 15 grader celcius)


#Initial conditions
#Earth
R = 6371*10**3 #m 
omega = 7.29*10**(-5) #s^-1

def centralVec(Z, X, L):
    theta = np.arccos(Z/L)
    phi = np.arccos(X/(np.sin(theta)*L))
    CartCoor= np.array([ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return CartCoor

def lenVec(vec1, vec2):
    return np.sqrt(np.dot(vec1,vec2))

def lenCoor(X,Y,Z):
    return np.sqrt(X**2+Y**2+Z**2)
    
#Crépy
Nc = np.deg2rad(49.60500)
Ec = np.deg2rad(3.514722)
Crepy = np.array([R*np.cos(Nc)*np.cos(Ec), R*np.cos(Nc)*np.sin(Ec), R*np.sin(Nc)])

print("Crepy: ", Crepy[0]/1000, Crepy[1]/1000, Crepy[2]/1000)


#Paris
Np = np.deg2rad(48.8667)
Ep = np.deg2rad(2.35083)
Paris = np.array([R*np.cos(Np)*np.cos(Ep), R*np.cos(Np)*np.sin(Ep), R*np.sin(Np)])

print("Paris: ", Paris[0]/1000, Paris[1]/1000, Paris[2]/1000)

#Reims
Nr = np.deg2rad(49.2583)
Er = np.deg2rad(4.0317)
Reims = np.array([R*np.cos(Nr)*np.cos(Er), R*np.cos(Nr)*np.sin(Er), R*np.sin(Nr)])

#StQt
Ns = np.deg2rad(49.8471)
Es = np.deg2rad(3.2874)
StQt = np.array([R*np.cos(Ns)*np.cos(Es), R*np.cos(Ns)*np.sin(Es), R*np.sin(Ns)])

#Distance between Crepy and Paris 
beta = np.arccos(np.dot(Paris,Crepy)/(lenVec(Paris,Paris)*lenVec(Crepy,Crepy)))
d = beta * R
print("Distance between places: ", d/1000)

t_min = 0
t_max = 400
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
    L = lenCoor(X_,Y_,Z_)
    V = lenCoor(U_,V_,W_)
    h = L-R
    AD = (1 -((a*h)/T_0))**alpha #airdensity adiabatic
    if np.isnan(AD) or h<0: 
        AD = 0
    #print("air density stuff: ", h, L, R)
#    if(AD!=0):
#        print("AD and height", AD, h, gammaTry[i])
    central = centralVec(Z_,X_,L)
    G = g*central[0]
    if trigger:
        C = 2*V_*omega
    else: 
        C = 0
#    C = 0 #2*V_*omega
    return G -B2_m2*U_*V*AD+C

def I(X_,Y_,Z_,U_,V_,W_):          #dV/dt
    L = lenCoor(X_,Y_,Z_)
    V = lenCoor(U_,V_,W_)
    h = L-R
    AD = (1 - ((a*h)/T_0))**alpha #airdensity adiabatic
    if np.isnan(AD): 
        AD = 0
    central = centralVec(Z_,X_,L)
    G = g*central[1]
    if trigger:
        C = -2*U_*omega
    else: 
        C = 0
    #C= 0 #-2*U_*omega
    return G -B2_m2*V_*V*AD+C
    
def J(X_,Y_,Z_,U_,V_,W_):           #dW/dt
    L = lenCoor(X_,Y_,Z_)
    V = lenCoor(U_,V_,W_)
    h = L-R
    AD = (1 - ((a*h)/T_0))**alpha
    if np.isnan(AD): 
        AD = 0
    central = centralVec(Z_,X_,L)
    G = g*central[2]
    return G-B2_m2*W_*V*AD

def RKfunc(X0, Y0, Z0, U0, V0, W0, t_min, t_max, tau):  

    dt_RK = tau
    N_RK = int(t_max/dt_RK)
    t_RK = np.linspace(t_min, t_max, N_RK)
    x_l=0
    y_l=0
    z_l=0
    t_l=0
    index=0
    max_height = 0
    
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
        
        rho2 = lenCoor(X_RK[n+1],Y_RK[n+1],Z_RK[n+1])
        rho1 = lenCoor(X_RK[n],Y_RK[n],Z_RK[n])
                    
        if rho2 < R and rho1 > R:
            r = abs(- lenCoor(X_RK[n],Y_RK[n],Z_RK[n]) / lenCoor(X_RK[n+1],Y_RK[n+1],Z_RK[n+1]))
            #print("r ", r)
            x_l = (X_RK[n] + r*X_RK[n+1])/(r + 1)
            y_l = (Y_RK[n] + r*Y_RK[n+1])/(r + 1)
            z_l = (Z_RK[n] + r*Z_RK[n+1])/(r + 1)
            t_l = (t_RK[n] + t_RK[n+1])/2
            print("HIT GROUND at time ", t_l)
            index = n
            
        #calculating maximum height
        if ((rho1-R)>max_height):
            max_height = rho1-R
        
    
    return X_RK, Y_RK, Z_RK, U_RK, V_RK, W_RK, x_l, y_l, z_l, t_RK, t_l, rho1, index


###code to run

#gamma = 0.99 # 0.99 er bra
target = StQt

bvec = (target - Crepy) / lenVec(target-Crepy,target-Crepy)
rvec = centralVec(Crepy[2],Crepy[0], lenVec(Crepy,Crepy))
Ng = 10
gamma = np.linspace(0.01,0.1,Ng)
diffList = np.zeros(Ng)
trigger=False

for i in range(len(gamma)): 
    print("Gamma: ",gamma[i])
    direction=(bvec+rvec*gamma[i])
    shootvec= direction/lenVec(direction,direction)
    
    #print("length of shooting vector", lenVec(shootvec,shootvec))
    
    V_start = 1640
    V = V_start*shootvec
    
    #print("starting velocity: ", V[0], V[1], V[2], " velocity: ", lenVec(V,V))
    #print("Starting position: ", lenVec(Crepy,Crepy))

    RK = RKfunc(Crepy[0], Crepy[1], Crepy[2], V[0], V[1], V[2], t_min, t_max, dt) 
    

    if RK[-1] != 0:
        xlist = RK[0][0:RK[-1]]
        ylist = RK[1][0:RK[-1]]
        zlist = RK[2][0:RK[-1]]
        
    diffVec = np.array([RK[6]-target[0],RK[7]-target[1],RK[8]-target[2]])
    diffList[i] = lenVec(diffVec,diffVec)
    
    print("landing position",xlist[-1]/1000,ylist[-1]/1000,zlist[-1]/1000, "\ndifference: ", (RK[6]-target[0])/1000,(RK[7]-target[1])/1000,(RK[8]-target[2])/1000)
   
#Kjør denne med dt = 0.01 og et par steps. 
plt.figure()
plt.plot(np.rad2deg(np.arctan(gamma)), diffList)
plt.title("Distance from Paris")
plt.xlabel(r"Shooting angle $[\degree]$")
plt.ylabel(r"Distance $[m]$")
plt.grid()
plt.show()




#u = np.linspace(0, 2 * np.pi, 100)
#v = np.linspace(0, np.pi, 100)
#x = R * np.outer(np.cos(u), np.sin(v))
#y = R * np.outer(np.sin(u), np.sin(v))
#z = R * np.outer(np.ones(np.size(u)), np.cos(v))
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
##ax.plot3D(RK[0]/1000, RK[1]/1000, RK[2]/1000, label='path')
#ax.plot3D(xlist/1000, ylist/1000, zlist/1000, label='path')
##ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='green')
#ax.scatter3D(Crepy[0]/1000,Crepy[1]/1000,Crepy[2]/1000, color = "darkorange", marker = "o")
#ax.scatter3D(Paris[0]/1000,Paris[1]/1000,Paris[2]/1000, color = "red", marker = "o")
##ax.scatter3D(RK[6]/1000,RK[7]/1000,R/1000, color = "green", marker = "o")
#ax.legend()
#plt.show()
