#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:04:52 2024

@author: JuliusSiebenaller
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpmath
from joblib import Parallel, delayed

# Parameters
v0 = 0.5  # nmol per mm in the simulation domain
o0 = 1.0  # nmol per mm      # S1 for 3D to 1D conversion
p0 = 40.  # cells per mm
# Continuous approx of Heaviside decreasing step function H_y(x-eps)=1.-(1/(1+exp(-2.*gamma(x-eps))))

# Density of glioma cells: p(x,t)
# Intrinsic diffusion rate of glioma cells
Dp = 2.73e-2  # mm^2 per day  # Range: 2.73e-3 to 2.73e-1 low- to high-grade gliomas
# Intrinsic proliferation rate of glioma cells:
bp = 2.73e-3  # mm^2 per day  # Range: 2.73e-4 to 2.73e-2 low- to high-grade gliomas
BCC = 100.  # Brain carrying capacity - N in the model     # cells per mm
l1 = 2.0  # nmol per mm        # Range: 10 to 80mmHg #Lambda_1
l2 = 1.0  # 0.0 for no proliferation   # Lambda_2

# Oxygen concentration: o(x,t)
Do = 1.51e2  # mm^2 per day
h1 = 3.37e-1  # per day     # Oxygen supply rate
h2 = 5.73e-3  # mm per cell per day # Range: 5.73e-3 to 1.14e-1 # Glioma cell oxygen consumption rate - mm per cell per day

# Vasculature density: v(x,t)
Dv = 5.0e-4 # mm^2 per day - Vasculature dispersal rate
g1 = 0.1  # Vasculature formation rate - per day
oa_star = 2.5e-1  # Oxygen concentration threshold for hypoxia - nmol per mm
K = 1.0  # Half maximal pro-angiogenic factor concentration  -nmol per mm
g2 = 5e-13  # Vaso-occlusion rate - cell^-n mm^n per day
vo_deg = 6.  # Dimensionless vaso-occlusion degree
theta = 10.  # Heaviside parameter vasculature


# Define the reaction terms
def reaction_terms(p):
    p_reaction = bp * o0/l1 * p * (1-p/BCC)

    return p_reaction

# Define the system of ODEs resulting from discretization
def odes(t, y, x, Dp, dx):
    N = len(x)
    p = y[:N]

    # Compute first and second derivatives using central difference
    dp_dx = np.gradient(p, dx)
    d2p_dx2 = np.gradient(dp_dx, dx)

    # Apply Neumann boundary conditions (zero flux)
    dp_dx[0] = 0
    dp_dx[-1] = 0
    d2p_dx2[0] = d2p_dx2[1]
    d2p_dx2[-1] = d2p_dx2[-2]

    # Compute reaction terms
    p_reaction = reaction_terms(p)

    # Combine diffusion and reaction terms
    Nabla_alpha_p = (l1-o0)/l1 * d2p_dx2
    dpdt = Dp*Nabla_alpha_p + p_reaction

    return np.concatenate([dpdt])


# Initial conditions
gamma = 1.e1
eps = 0.5
def initial_conditions(x):
    p_0 = p0 * (1. - 1. / (1+ np.exp(-2.*gamma*(x-eps))))  # Initial distribution for u
    return np.concatenate([p_0])


# Spatial domain and time span
x = np.linspace(0, 200, 1000)
t_span = (0, 1000)
dx = x[1] - x[0]
y0 = initial_conditions(x)

# Solve the system of ODEs
sol = solve_ivp(odes, t_span, y0, args=(x, Dp, dx), t_eval=np.linspace(0, 1000, 1000), method='RK45')
#sol = solve_ivp(odes, t_span, y0, args=(x, Dp, dx), t_eval=np.linspace(0, 1000, 1000), method='RK45')

# Extract solutions
p_sol = sol.y[:len(x), :]
np.save('FisherSolI_p.npy', p_sol)

# Plot results
X, T = np.meshgrid(x, sol.t)

fig, ax = plt.subplots(3, 1, figsize=(10, 8))

cfp = ax[0].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfp, ax=ax[0], label='p')
ax[0].set_title('Solution $p(x,t)$ of the Fisher-Kolmogoroff Model')
ax[0].set_xlabel('Position $x$ in $mm$')
ax[0].set_ylabel('Time $t$ in days')
#plt.title('Solution of Fisher-Kolmogoroff Model')

cfo = ax[1].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfo, ax=ax[1], label='p')
#ax[1].set_title('Solution p(x,t)')
ax[1].set_xlabel('Position $x$ in $mm$')
ax[1].set_ylabel('Time $t$ in days')
ax[1].set_xlim(0,10)
ax[1].set_ylim(0,300)

cfv = ax[2].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfv, ax=ax[2], label='p')
#ax[2].set_title('Solution p(x,t)')
ax[2].set_xlabel('Position $x$ in $mm$')
ax[2].set_ylabel('Time $t$ in days')
ax[2].set_xlim(0,2)
ax[2].set_ylim(0,150)

plt.tight_layout()
#plt.title('Solution of Fisher-Kolmogoroff Model')
plt.savefig('FK_Model_Contour.png')
plt.show()

# Plotting

plt.figure()
#plt.figure(figsize=(10, 6))
t_end = len(T)
for i in range(0, t_end, int(t_end/10)):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[i,:], p_sol[:,i], label=f"t = {i} days")
plt.plot(X[-1,:], p_sol[:,-1], label=f"t = {1000} days")
plt.title("Solution $p(x,t)$ of the Fisher-Kolmogoroff Model")
plt.xlim(0,30)
plt.ylim(0,40)
plt.xlabel("Position $x$ in mm")
plt.ylabel("Density $p(x, t)$ in cells per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('FK_Model_Dist.png')
plt.show()

plt.figure()
#plt.figure(figsize=(10, 6))
for i in range(0, t_end, int(t_end/10)):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[i,:], p_sol[:,i], label=f"t = {i} days")
plt.plot(X[-1,:], p_sol[:,-1], label=f"t = 1000 days")
plt.title("Solution $p(x,t)$ of the Fisher-Kolmogoroff Model")
plt.xlim(0,20)
plt.ylim(0,17.50)
plt.xlabel("Position $x$ in mm")
plt.ylabel("Density $p(x, t)$ in cells per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('FK_Model_Dist_Zoom.png')
plt.show()



##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
# Change the parameters 

# Density of glioma cells: p(x,t)
# Intrinsic diffusion rate of glioma cells
Dp = 2.73e-1  # mm^2 per day  # Range: 2.73e-3 to 2.73e-1 low- to high-grade gliomas
# Intrinsic proliferation rate of glioma cells:
bp = 2.73e-3  # mm^2 per day  # Range: 2.73e-4 to 2.73e-2 low- to high-grade gliomas
BCC = 100.  # Brain carrying capacity - N in the model     # cells per mm
l1 = 2.0  # nmol per mm        # Range: 10 to 80mmHg #Lambda_1
l2 = 1.0  # 0.0 for no proliferation   # Lambda_2

# Spatial domain and time span
x = np.linspace(0, 200, 1000)
t_span = (0, 1000)
dx = x[1] - x[0]
y0 = initial_conditions(x)

# Solve the system of ODEs
sol = solve_ivp(odes, t_span, y0, args=(x, Dp, dx), t_eval=np.linspace(0, 1000, 1000), method='RK45')
#sol = solve_ivp(odes, t_span, y0, args=(x, Dp, dx), t_eval=np.linspace(0, 1000, 1000), method='RK45')

# Extract solutions
p_sol = sol.y[:len(x), :]
np.save('FisherSol_II_p.npy', p_sol)

# Plot results
X, T = np.meshgrid(x, sol.t)

fig, ax = plt.subplots(3, 1, figsize=(10, 8))

cfp = ax[0].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfp, ax=ax[0], label='$p(x,t)$ in cells per $mm$')
ax[0].set_title('Solution $p(x,t)$ of the Fisher-Kolmogoroff Model')
ax[0].set_xlabel('Position $x$ in $mm$')
ax[0].set_ylabel('Time $t$ in days')
#plt.title('Solution of Fisher-Kolmogoroff Model')

cfo = ax[1].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfo, ax=ax[1], label='$p(x,t)$ in cells per $mm$')
#ax[1].set_title('Solution p(x,t)')
ax[1].set_xlabel('Position $x$ in $mm$')
ax[1].set_ylabel('Time $t$ in days')
ax[1].set_xlim(0,10)
ax[1].set_ylim(0,300)

cfv = ax[2].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfv, ax=ax[2], label='$p(x,t)$ in cells per $mm$')
#ax[2].set_title('Solution p(x,t)')
ax[2].set_xlabel('Position $x$ in $mm$')
ax[2].set_ylabel('Time $t$ in days')
ax[2].set_xlim(0,2)
ax[2].set_ylim(0,150)

plt.tight_layout()
#plt.title('Solution of Fisher-Kolmogoroff Model')
plt.savefig('FK_Model_Contour_II.png')
plt.show()

# Plotting

plt.figure()
#plt.figure(figsize=(10, 6))
t_end = len(T)
for i in range(0, t_end, int(t_end/10)):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[i,:], p_sol[:,i], label=f"t = {i} days")
plt.plot(X[-1,:], p_sol[:,-1], label=f"t = {1000} days")
plt.title("Solution $p(x,t)$ of the Fisher-Kolmogoroff Model")
#plt.xlim(0,30)
#plt.ylim(0,40)
plt.xlabel("Position $x$ in mm")
plt.ylabel("Density $p(x, t)$ in cells per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('FK_Model_Dist_II.png')
plt.show()

plt.figure()
#plt.figure(figsize=(10, 6))
for i in range(0, t_end, int(t_end/10)):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[i,:], p_sol[:,i], label=f"t = {i} days")
plt.plot(X[-1,:], p_sol[:,-1], label=f"t = 1000 days")
plt.title("Solution $p(x,t)$ of the Fisher-Kolmogoroff Model")
plt.xlim(0,40)
plt.ylim(0,25.0)
plt.xlabel("Position $x$ in mm")
plt.ylabel("Density $p(x, t)$ in cells per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('FK_Model_Dist_Zoom_II.png')
plt.show()

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
# Change the parameters 

# Density of glioma cells: p(x,t)
# Intrinsic diffusion rate of glioma cells
Dp = 2.73e-2  # mm^2 per day  # Range: 2.73e-3 to 2.73e-1 low- to high-grade gliomas
# Intrinsic proliferation rate of glioma cells:
bp = 2.73e-2  # mm^2 per day  # Range: 2.73e-4 to 2.73e-2 low- to high-grade gliomas
BCC = 100.  # Brain carrying capacity - N in the model     # cells per mm
l1 = 2.0  # nmol per mm        # Range: 10 to 80mmHg #Lambda_1
l2 = 1.0  # 0.0 for no proliferation   # Lambda_2

# Spatial domain and time span
x = np.linspace(0, 200, 1000)
t_span = (0, 1000)
dx = x[1] - x[0]
y0 = initial_conditions(x)

# Solve the system of ODEs
sol = solve_ivp(odes, t_span, y0, args=(x, Dp, dx), t_eval=np.linspace(0, 1000, 1000), method='RK45')
#sol = solve_ivp(odes, t_span, y0, args=(x, Dp, dx), t_eval=np.linspace(0, 1000, 1000), method='RK45')

# Extract solutions
p_sol = sol.y[:len(x), :]
np.save('FisherSol_III_p.npy', p_sol)

# Plot results
X, T = np.meshgrid(x, sol.t)

fig, ax = plt.subplots(3, 1, figsize=(10, 8))

cfp = ax[0].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfp, ax=ax[0], label='$p(x,t)$ in cells per $mm$')
ax[0].set_title('Solution $p(x,t)$ of the Fisher-Kolmogoroff Model')
ax[0].set_xlabel('Position $x$ in $mm$')
ax[0].set_ylabel('Time $t$ in days')
#plt.title('Solution of Fisher-Kolmogoroff Model')

cfo = ax[1].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfo, ax=ax[1], label='$p(x,t)$ in cells per $mm$')
#ax[1].set_title('Solution p(x,t)')
ax[1].set_xlabel('Position $x$ in $mm$')
ax[1].set_ylabel('Time $t$ in days')
ax[1].set_xlim(0,10)
ax[1].set_ylim(0,300)

cfv = ax[2].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfv, ax=ax[2], label='p')
#ax[2].set_title('Solution p(x,t)')
ax[2].set_xlabel('Position $x$ in $mm$')
ax[2].set_ylabel('Time $t$ in days')
ax[2].set_xlim(0,2)
ax[2].set_ylim(0,150)

plt.tight_layout()
#plt.title('Solution of Fisher-Kolmogoroff Model')
plt.savefig('FK_Model_Contour_III.png')
plt.show()

# Plotting

plt.figure()
#plt.figure(figsize=(10, 6))
t_end = len(T)
for i in range(0, t_end, int(t_end/10)):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[i,:], p_sol[:,i], label=f"t = {i} days")
plt.plot(X[-1,:], p_sol[:,-1], label=f"t = {1000} days")
plt.title("Solution $p(x,t)$ of the Fisher-Kolmogoroff Model")
#plt.xlim(0,30)
#plt.ylim(0,40)
plt.xlabel("Position $x$ in $mm$")
plt.ylabel("Density $p(x, t)$ in cells per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('FK_Model_Dist_III.png')
plt.show()

plt.figure()
#plt.figure(figsize=(10, 6))
for i in range(0, t_end, int(t_end/10)):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[i,:], p_sol[:,i], label=f"t = {i} days")
plt.plot(X[-1,:], p_sol[:,-1], label=f"t = 1000 days")
plt.title("Solution $p(x,t)$ of the Fisher-Kolmogoroff Model")
plt.xlim(0,50)
#plt.ylim(0,17.50)
plt.xlabel("Position $x$ in $mm$")
plt.ylabel("Density $p(x, t)$ in cells per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('FK_Model_Dist_Zoom_III.png')
plt.show()

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
# Change the parameters 

# Density of glioma cells: p(x,t)
# Intrinsic diffusion rate of glioma cells
Dp = 2.73e-1  # mm^2 per day  # Range: 2.73e-3 to 2.73e-1 low- to high-grade gliomas
# Intrinsic proliferation rate of glioma cells:
bp = 2.73e-2  # mm^2 per day  # Range: 2.73e-4 to 2.73e-2 low- to high-grade gliomas
BCC = 100.  # Brain carrying capacity - N in the model     # cells per mm
l1 = 2.0  # nmol per mm        # Range: 10 to 80mmHg #Lambda_1
l2 = 1.0  # 0.0 for no proliferation   # Lambda_2

# Spatial domain and time span
x = np.linspace(0, 200, 1000)
t_span = (0, 1000)
dx = x[1] - x[0]
y0 = initial_conditions(x)

# Solve the system of ODEs
sol = solve_ivp(odes, t_span, y0, args=(x, Dp, dx), t_eval=np.linspace(0, 1000, 1000), method='RK45')
#sol = solve_ivp(odes, t_span, y0, args=(x, Dp, dx), t_eval=np.linspace(0, 1000, 1000), method='RK45')

# Extract solutions
p_sol = sol.y[:len(x), :]
np.save('FisherSol_IV_p.npy', p_sol)

# Plot results
X, T = np.meshgrid(x, sol.t)

fig, ax = plt.subplots(3, 1, figsize=(10, 8))

cfp = ax[0].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfp, ax=ax[0], label='$p(x,t)$ in cells per $mm$')
ax[0].set_title('Solution $p(x,t)$ of the Fisher-Kolmogoroff Model')
ax[0].set_xlabel('Position $x$ in $mm$')
ax[0].set_ylabel('Time $t$ in days')
#plt.title('Solution of Fisher-Kolmogoroff Model')

cfo = ax[1].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfo, ax=ax[1], label='$p(x,t)$ in cells per $mm$')
#ax[1].set_title('Solution p(x,t)')
ax[1].set_xlabel('Position $x$ in $mm$')
ax[1].set_ylabel('Time $t$ in days')
ax[1].set_xlim(0,10)
ax[1].set_ylim(0,300)

cfv = ax[2].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfv, ax=ax[2], label='$p(x,t)$ in cells per $mm$')
#ax[2].set_title('Solution p(x,t)')
ax[2].set_xlabel('Position $x$ in $mm$')
ax[2].set_ylabel('Time $t$ in days')
ax[2].set_xlim(0,2)
ax[2].set_ylim(0,150)

plt.tight_layout()
#plt.title('Solution of Fisher-Kolmogoroff Model')
plt.savefig('FK_Model_Contour_IV.png')
plt.show()

# Plotting

plt.figure()
#plt.figure(figsize=(10, 6))
t_end = len(T)
for i in range(0, t_end, int(t_end/10)):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[i,:], p_sol[:,i], label=f"t = {i} days")
plt.plot(X[-1,:], p_sol[:,-1], label=f"t = {1000} days")
plt.title("Solution $p(x,t)$ of the Fisher-Kolmogoroff Model")
#plt.xlim(0,30)
#plt.ylim(0,40)
plt.xlabel("Position $x$ in $mm$")
plt.ylabel("Density $p(x, t)$ in cells per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('FK_Model_Dist_IV.png')
plt.show()

plt.figure()
#plt.figure(figsize=(10, 6))
for i in range(0, t_end, int(t_end/10)):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[i,:], p_sol[:,i], label=f"t = {i} days")
plt.plot(X[-1,:], p_sol[:,-1], label=f"t = 1000 days")
plt.title("Solution $p(x,t)$ of the Fisher-Kolmogoroff Model")
plt.xlim(0,100)
#plt.ylim(0,17.50)
plt.xlabel("Position $x$ in $mm$")
plt.ylabel("Density $p(x, t)$ in cells per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('FK_Model_Dist_Zoom_Iv.png')
plt.show()


