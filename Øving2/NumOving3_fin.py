# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:32:15 2018

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

trigger = False


#Initial conditions
#Earth
R = 6371*10**3 #m 
omega = 7.29*10**(-5) #s^-1

def centralVec(Z, X, L):
    theta = np.arccos(Z/L)
    phi = np.arccos(X/(np.sin(theta)*L))
    CartCoor= np.array([ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return CartCoor

#onlu if vec1 and vec 2 is alike?
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
#Np = np.deg2rad(48.5667)
#Ep = np.deg2rad(2.35083)
#Paris = np.array([R*np.cos(Np)*np.cos(Ep), R*np.cos(Np)*np.sin(Ep), R*np.sin(Np)])
#
#print("Paris: ", Paris[0]/1000, Paris[1]/1000, Paris[2]/1000)

#Target

#(48°51ʹ24ʺN 2°21ʹ032''E) PARIS
#(49°36ʹ18ʺN 3°30ʹ53ʺE) CREPY


#N =np.array([48,51,24])
#E = np.array([2,21,32])

Nt = np.deg2rad(48.8667)#N[0] + N[1]/60 + N[2]/3600)
Et = np.deg2rad(2.35083)#E[0] + E[1]/60 + E[2]/3600)

Target = np.array([R*np.cos(Nt)*np.cos(Et), R*np.cos(Nt)*np.sin(Et), R*np.sin(Nt)])

print("Target: ", Target[0]/1000, Target[1]/1000, Target[2]/1000)


#Distance between Crepy and Target
#beta = np.arccos(np.dot(Target,Crepy)/(lenVec(Target,Target)*lenVec(Crepy,Crepy)))
#d = beta * R
#print("Distance between places: ", d/1000)
print("Distance between places: ", lenVec(Target-Crepy,Target-Crepy)/1000)



bvec = (Target - Crepy) / lenVec(Target-Crepy,Target-Crepy)

rvec = centralVec(Crepy[2],Crepy[0], lenVec(Crepy,Crepy))


t_min = 0
t_max = 200
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
    if h < 0 or AD < 0: 
        AD = 0
    #print("air density stuff: ", h, L, R)
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
    if h < 0 or AD < 0: 
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
    if h < 0 or AD < 0: 
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

#gamma = 0.706
gammaTry = np.linspace(0.1,1,50)
diffList = np.zeros(len(gammaTry))
best = 1000000
gamma = 0
V_start = 1640

for i in range(len(gammaTry)): 
    print("Gamma: ",gammaTry[i])
    directionTry=(bvec+rvec*gammaTry[i])
    shootvecTry= directionTry/lenVec(directionTry,directionTry)
    
    #print("length of shooting vector", lenVec(shootvec,shootvec))
    
    Vtry = V_start*shootvecTry
    
    #print("starting velocity: ", V[0], V[1], V[2], " velocity: ", lenVec(V,V))
    #print("Starting position: ", lenVec(Crepy,Crepy))

    RKtry = RKfunc(Crepy[0], Crepy[1], Crepy[2], Vtry[0], Vtry[1], Vtry[2], t_min, t_max, dt) 
    

    if RKtry[-1] != 0:
        xlist = RKtry[0][0:RKtry[-1]]
        ylist = RKtry[1][0:RKtry[-1]]
        zlist = RKtry[2][0:RKtry[-1]]
        
    diffVec = np.array([RKtry[6]-Target[0],RKtry[7]-Target[1],RKtry[8]-Target[2]])
    diffList[i] = lenVec(diffVec,diffVec)
    
    if diffList[i]<best: 
        best = diffList[i]
        gamma = gammaTry[i]
    
    
direction=(bvec+rvec*gamma)
shootvec= direction/lenVec(direction,direction)

#print("length of shooting vector", lenVec(shootvec,shootvec))

V = V_start*shootvec

#print("starting velocity: ", V[0], V[1], V[2], " velocity: ", lenVec(V,V))
#print("Starting position: ", lenVec(Crepy,Crepy))
trigger = False
RK = RKfunc(Crepy[0], Crepy[1], Crepy[2], V[0], V[1], V[2], t_min, t_max, dt) 
trigger = True
RKc = RKfunc(Crepy[0], Crepy[1], Crepy[2], V[0], V[1], V[2], t_min, t_max, dt) 


if RK[-1] != 0:
    xlist = RK[0][0:RK[-1]]
    ylist = RK[1][0:RK[-1]]
    zlist = RK[2][0:RK[-1]]
    
if RKc[-1] != 0:
    xlistc = RKc[0][0:RKc[-1]]
    ylistc = RKc[1][0:RKc[-1]]
    zlistc = RKc[2][0:RKc[-1]]
 
print("landing position",xlist[-1]/1000,ylist[-1]/1000,zlist[-1]/1000, "\nkm from Target: ", (RK[6]-Target[0])/1000,(RK[7]-Target[1])/1000,(RK[8]-Target[2])/1000)
lvec1 = Crepy - (RK[6:9])
lvec2 = Crepy - (RKc[6:9])
#print(lvec1, RK[6])
deltaL=[RK[6]-RKc[6],RK[7]-RKc[7],RK[8]-RKc[8]] #difference in landing point
b = lenVec(lvec1,lvec1)
c = lenVec(lvec2,lvec2)
a = lenVec(deltaL,deltaL)
#using law of cosines, finding deflection angle
deflAng = np.arccos((b**2+c**2-a**2)/(2*b*c))
print("Difference in landing position: ", deltaL,(lenVec(deltaL,deltaL)))
print("Direction: ", deltaL/(lenVec(deltaL,deltaL)))
print("Deflection angle: ", deflAng)

print("Maximum heights (km): ", RK[11]/1000,RKc[11]/1000, "Difference: ", (RK[11]-RKc[11])/1000)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot3D(RK[0]/1000, RK[1]/1000, RK[2]/1000, color = 'red', label='no coreolis')
#ax.plot3D(RKc[0]/1000, RKc[1]/1000, RKc[2]/1000, color = 'green', label='coreolis')

# Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
ax.text2D(0.05, 0.95, "Projectile path", transform=ax.transAxes)


ax.plot3D(xlist/1000, ylist/1000, zlist/1000, color= 'red', label='No coriolis force')
ax.plot3D(xlistc/1000,ylistc/1000,zlistc/1000, color= 'green', label='Coriolis force')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

#ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='green')
ax.scatter3D(Crepy[0]/1000,Crepy[1]/1000,Crepy[2]/1000, color = "darkorange", marker = "o")
ax.scatter3D(Target[0]/1000,Target[1]/1000,Target[2]/1000, color = "blue", marker = "o")
#ax.scatter3D(RK[6]/1000,RK[7]/1000,R/1000, color = "green", marker = "o")
#ax.axis((4100,4250,170,175))
ax.legend()
plt.show()

##example earth
#u = np.linspace(0, 2 * np.pi, 100)
#v = np.linspace(0, np.pi, 100)
#x = R * np.outer(np.cos(u), np.sin(v))/R
#y = R * np.outer(np.sin(u), np.sin(v))/R
#z = R * np.outer(np.ones(np.size(u)), np.cos(v))/R
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.text2D(0.05, 0.95, "Earth", transform=ax.transAxes)
#ax.set_xlabel('X axis')
#ax.set_ylabel('Y axis')
#ax.set_zlabel('Z axis')
#ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='lightblue')
#ax.axis("equal")
#ax.legend()
#plt.show()