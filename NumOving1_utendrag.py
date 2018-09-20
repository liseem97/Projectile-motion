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
t_max = 102
dt = 0.1             #time step / tau
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
    i = 0
    for i in range(0, len(times)):
        Y_A[i] = -0.5*g*times[i]**2 + times[i]*V0 + Y0
        X_A[i] = V0*times[i] + X0
    return X_A, Y_A
   

###code to run
     
RK = RK(X0, Y0, U0, V0, t_min, t_max, dt) 
AN = analytical(X0, Y0, U0, V0, RK[-1])



#RK   
plt.figure()
plt.title("Position numerical")
plt.plot(RK[0], RK[1], color = "darkblue", label = "Projectile path RK")
plt.plot(AN[0], AN[1], color = "red", label = "Analytical path")
#plt.plot([0], [0], color = "darkorange", marker = "o", markersize = 19, label = "Fixed sun")
plt.legend(loc = 4)
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
#"plt.axis("equal")
plt.grid()
#plt.savefig("/Users/elveb/Documents/1_RK_pos.pdf")
plt.show()

#ENERGY
fig = plt.figure(figsize=(8, 2))
plt.title("Energy as a function of time")
plt.plot(RK[-1], RK[4]/1000, label= "Total energy", color = "crimson")
plt.plot(RK[-1], RK[5]/1000, label = "Kinetic energy", color = "blue")
plt.plot(RK[-1], RK[6]/1000, label= "Potential energy", color = "orange")
plt.xlabel(r"time [s]")
plt.ylabel(r"$Etot$ [kJ / kg]")
plt.legend(loc = 4)
plt.grid()
#plt.savefig("/Users/elveb/Documents/1_Ep.pdf")
plt.show()

#ENERGY
fig = plt.figure(figsize=(8, 2))
plt.title("Energy as a function of time")
plt.plot(RK[-1], RK[4]/1000, label= "Total energy", color = "crimson")
plt.xlabel(r"time [s]")
plt.ylabel(r"$Etot$ [kJ / kg]")
plt.legend(loc = 4)
plt.grid()
#plt.savefig("/Users/elveb/Documents/1_Ep.pdf")
plt.show()

#Velocity numerical
#plt.figure()
#plt.title("Velocity")
#plt.plot(RK[4], RK[2], color = "darkblue", label = "X-velocity")
#plt.plot(RK[4], RK[3], color = "red", label = "Y-velocity")
##plt.plot([0], [0], color = "darkorange", marker = "o", markersize = 19, label = "Fixed sun")
#plt.legend(loc = 4)
#plt.xlabel(r"$x$ [s]")
#plt.ylabel(r"$y$ [m/s]")
##plt.axis("equal")
#plt.grid()
##plt.savefig("/Users/elveb/Documents/1_RK_pos.pdf")
#plt.show()

##analytical
#plt.figure()
#plt.title("Position analytical")
##plt.plot(RK[0], RK[1], color = "darkblue", label = "Projectile path RK")
#plt.plot(AN[0], AN[1], color = "red", label = "Analytical path")
##plt.plot([0], [0], color = "darkorange", marker = "o", markersize = 19, label = "Fixed sun")
#plt.legend(loc = 4)
#plt.xlabel(r"$x$ [m]")
#plt.ylabel(r"$y$ [m]")
##"plt.axis("equal")
#plt.grid()
##plt.savefig("/Users/elveb/Documents/1_RK_pos.pdf")
#plt.show()