# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:21:06 2018

@author: lise
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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
    if math.isnan(AD) or h<0: 
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
    if math.isnan(AD): 
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
    if math.isnan(AD): 
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
    
#Timesteps etc
t_min = 0
t_max = 300
dt = 0.1             #time step / tau
N = int(t_max/dt)


#Crépy
Nc = np.deg2rad(49.60500)
Ec = np.deg2rad(3.514722)
Crepy = np.array([R*np.cos(Nc)*np.cos(Ec), R*np.cos(Nc)*np.sin(Ec), R*np.sin(Nc)])

print("Crepy: ", Crepy[0]/1000, Crepy[1]/1000, Crepy[2]/1000)
    
#paris, reims, saint quentin
Nt = np.deg2rad(np.array([48.8667,49.2583,49.8471]))
Et = np.deg2rad(np.array([2.35083,4.0317,3.2874]))

rvec = centralVec(Crepy[2],Crepy[0], lenVec(Crepy,Crepy))

Target = np.zeros([3,3])
Target[0:] = np.array([R*np.cos(Nt)*np.cos(Et), R*np.cos(Nt)*np.sin(Et), R*np.sin(Nt)])
Target = np.transpose(Target)

gammaMin=np.array([0.6, 6, 10])
gammaMax=np.array([0.75, 6.5, 12])

gamma = np.zeros(3)

V_start = 1640
V = np.zeros([3,3])

for n in range(3): 
    print("\n \n \n")
    print("Target: ", Target[n][0]/1000, Target[n][1]/1000, Target[n][2]/1000)
    dist = lenVec(Target[n]-Crepy,Target[n]-Crepy)/1000
    print("Distance between places: ", dist)
    if dist > 120: 
        exit(0)
    
    bvec = np.zeros(3)
    bvec = (Target[n] - Crepy) / lenVec(Target[n]-Crepy,Target[n]-Crepy)
 
    gammaTry = np.linspace(gammaMin[n],gammaMax[n], 10)
    
    best = 100000
    
    for i in range(len(gammaTry)): 
        
        print("Gamma: ",gammaTry[i])
        
        directionTry=(bvec+rvec*gammaTry[i])
        shootvecTry= directionTry/lenVec(directionTry,directionTry)
           
        Vtry = V_start*shootvecTry
        
        RKtry = RKfunc(Crepy[0], Crepy[1], Crepy[2], Vtry[0], Vtry[1], Vtry[2], t_min, t_max, dt) 
        
        diffVec = np.zeros(3)
        diffVec = np.array([RKtry[6]-Target[n][0],RKtry[7]-Target[n][1],RKtry[8]-Target[n][2]])
        
        diff = lenVec(diffVec,diffVec)
        
        if diff<best: 
            best = diff
            gamma[n] = gammaTry[i]
        
    print("Best gamma: ", gamma[n])    
    direction=(bvec+rvec*gamma[n])
    shootvec= direction/lenVec(direction,direction)

    V[n] = V_start * shootvec
    
  
trigger = False
RK0 = RKfunc(Crepy[0], Crepy[1], Crepy[2], V[0][0], V[0][1], V[0][2], t_min, t_max, dt) 
trigger = True
RKc0 = RKfunc(Crepy[0], Crepy[1], Crepy[2], V[0][0], V[0][1], V[0][2], t_min, t_max, dt) 

if RK0[-1] != 0:
    xlist0 = RK0[0][0:RK0[-1]]
    ylist0 = RK0[1][0:RK0[-1]]
    zlist0 = RK0[2][0:RK0[-1]]
    
if RKc0[-1] != 0:
    xlistc0 = RKc0[0][0:RKc0[-1]]
    ylistc0 = RKc0[1][0:RKc0[-1]]
    zlistc0 = RKc0[2][0:RKc0[-1]]
    
trigger = False
RK1 = RKfunc(Crepy[0], Crepy[1], Crepy[2], V[1][0], V[1][1], V[1][2], t_min, t_max, dt) 
trigger = True
RKc1 = RKfunc(Crepy[0], Crepy[1], Crepy[2], V[1][0], V[1][1], V[1][2], t_min, t_max, dt) 

if RK1[-1] != 0:
    xlist1 = RK1[0][0:RK1[-1]]
    ylist1 = RK1[1][0:RK1[-1]]
    zlist1 = RK1[2][0:RK1[-1]]
    
if RKc1[-1] != 0:
    xlistc1 = RKc1[0][0:RKc1[-1]]
    ylistc1 = RKc1[1][0:RKc1[-1]]
    zlistc1 = RKc1[2][0:RKc1[-1]]

trigger = False
RK2 = RKfunc(Crepy[0], Crepy[1], Crepy[2], V[2][0], V[2][1], V[2][2], t_min, t_max, dt) 
trigger = True
RKc2 = RKfunc(Crepy[0], Crepy[1], Crepy[2], V[2][0], V[2][1], V[2][2], t_min, t_max, dt) 

if RK2[-1] != 0:
    xlist2 = RK2[0][0:RK2[-1]]
    ylist2 = RK2[1][0:RK2[-1]]
    zlist2 = RK2[2][0:RK2[-1]]
    
if RKc2[-1] != 0:
    xlistc2 = RKc2[0][0:RKc2[-1]]
    ylistc2 = RKc2[1][0:RKc2[-1]]
    zlistc2 = RKc2[2][0:RKc2[-1]]


        
#print("landing position uten coriolis",RK0[6]/1000,RK0[7]/1000,RK0[8]/1000, "\nkm from Target: ", (RK0[6]-Target[n][0])/1000,(RK0[7]-Target[n][1])/1000,(RK0[8]-Target[n][2])/1000)
#print("landing position med coriolis",RKc0[6]/1000,RKc0[7]/1000,RKc0[8]/1000, "\nkm from Target: ", (RKc0[6]-Target[n][0])/1000,(RKc0[7]-Target[n][1])/1000,(RKc0[8]-Target[n][2])/1000)

#lvec1 = Crepy - (RK0[6:9])
#lvec2 = Crepy - (RKc0[6:9])
#
#deltaL=np.array([RK0[6]-RKc0[6],RK0[7]-RKc0[7],RK0[8]-RKc0[8]]) #difference in landing point
#
#
#b = lenVec(lvec1,lvec1)
#c = lenVec(lvec2,lvec2)
#a = lenVec(deltaL,deltaL)
##using law of cosines, finding deflection angle
#deflAng = np.arccos((b**2+c**2-a**2)/(2*b*c))
#
#print("Difference in landing position: ", deltaL, lenVec(deltaL,deltaL))
#print("Direction: ", deltaL/(lenVec(deltaL,deltaL)))
#print("Deflection angle: ", deflAng)

#print("Maximum heights (km): ", RK[11]/1000,RKc[11]/1000, "Difference: ", (RK[11]-RKc[11])/1000)



#print("landing position",xlist[-1]/1000,ylist[-1]/1000,zlist[-1]/1000, "\nkm from Target: ", (RK[6]-Target[0])/1000,(RK[7]-Target[1])/1000,(RK[8]-Target[2])/1000)
#lvec1 = Crepy - (RK[6:9])
#lvec2 = Crepy - (RKc[6:9])
##print(lvec1, RK[6])
#deltaL=[RK[6]-RKc[6],RK[7]-RKc[7],RK[8]-RKc[8]] #difference in landing point
#b = lenVec(lvec1,lvec1)
#c = lenVec(lvec2,lvec2)
#a = lenVec(deltaL,deltaL)
##using law of cosines, finding deflection angle
#deflAng = np.arccos((b**2+c**2-a**2)/(2*b*c))
#print("Difference in landing position: ", deltaL,(lenVec(deltaL,deltaL)))
#print("Direction: ", deltaL/(lenVec(deltaL,deltaL)))
#print("Deflection angle: ", deflAng)
#
#print("Maximum heights (km): ", RK[11]/1000,RKc[11]/1000, "Difference: ", (RK[11]-RKc[11])/1000)
#
fig = plt.figure()
ax = fig.gca(projection='3d')
##ax.plot3D(RK[0]/1000, RK[1]/1000, RK[2]/1000, color = 'red', label='no coreolis')
##ax.plot3D(RKc[0]/1000, RKc[1]/1000, RKc[2]/1000, color = 'green', label='coreolis')
#
## Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
#ax.text2D(0.05, 0.95, "Projectile path", transform=ax.transAxes)
#
ax.plot3D(xlist0/1000, ylist0/1000, zlist0/1000, color= 'red', label='No coriolis force')
ax.plot3D(xlistc0/1000,ylistc0/1000,zlistc0/1000, color= 'green', label='Coriolis force')
ax.plot3D(xlist1/1000, ylist1/1000, zlist1/1000, color= 'red', label='No coriolis force')
ax.plot3D(xlistc1/1000,ylistc1/1000,zlistc1/1000, color= 'green', label='Coriolis force')
ax.plot3D(xlist2/1000, ylist2/1000, zlist2/1000, color= 'red', label='No coriolis force')
ax.plot3D(xlistc2/1000,ylistc2/1000,zlistc2/1000, color= 'green', label='Coriolis force')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
#
##ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='green')
#ax.scatter3D(Crepy[0]/1000,Crepy[1]/1000,Crepy[2]/1000, color = "darkorange", marker = "o")
#ax.scatter3D(Target[0]/1000,Target[1]/1000,Target[2]/1000, color = "blue", marker = "o")
##ax.scatter3D(RK[6]/1000,RK[7]/1000,R/1000, color = "green", marker = "o")
##ax.axis((4100,4250,170,175))
ax.legend()
plt.show()
#
###example earth
##u = np.linspace(0, 2 * np.pi, 100)
##v = np.linspace(0, np.pi, 100)
##x = R * np.outer(np.cos(u), np.sin(v))/R
##y = R * np.outer(np.sin(u), np.sin(v))/R
##z = R * np.outer(np.ones(np.size(u)), np.cos(v))/R
##
##fig = plt.figure()
##ax = fig.gca(projection='3d')
##ax.text2D(0.05, 0.95, "Earth", transform=ax.transAxes)
##ax.set_xlabel('X axis')
##ax.set_ylabel('Y axis')
##ax.set_zlabel('Z axis')
##ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='lightblue')
##ax.axis("equal")
##ax.legend()
##plt.show()