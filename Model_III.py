import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#import mpmath
#from joblib import Parallel, delayed

# Parameters
v0 = 0.5  # nmol per mm in the simulation domain
o0 = 1.0  # nmol per mm      # S1 for 3D to 1D conversion
p0 = 40.  # cells per mm
# Continuous approx of Heaviside decreasing step function H_y(x-eps)=1.-(1/(1+exp(-2.*gamma(x-eps))))

# Density of glioma cells: p(x,t)
Dp = 2.73e-2  # mm^2 per day  # Intrinsic diffusion rate of glioma cells # Range: 2.73e-3 to 2.73e-1 low- to high-grade gliomas
# Intrinsic proliferation rate of glioma cells:
bp = 2.73e-3  # mm^2 per day  # Intrinsic proliferation rate of glioma cells: # Range: 2.73e-4 to 2.73e-2 low- to high-grade gliomas
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


# Define the reaction terms with parallel computation
def reaction_terms(p, o, v):
    p_reaction = bp * o / l1 * p * (1 - p / BCC)
    o_star_diff = o - oa_star
    v_reaction = g1 * p * np.heaviside(o_star_diff,0) * (1 - v) / (K + p * np.heaviside(o_star_diff,0) / v) - g2 * v * p ** vo_deg

    o_reaction = h1 * v0 * (o0 - o) - h2 * p * o

    # Parallelize Heaviside function computation
    #large_values = -2 * theta * (o - oa_star)
    #H_theta = np.array(Parallel(n_jobs=-1)(delayed(mpmath_exp)(lv) for lv in large_values))

    return p_reaction, o_reaction, v_reaction

"""def mpmath_exp(value): 
    # Used this in case of modeling with large values 
    return float(1 - 1 / (1 + mpmath.exp(mpmath.mpf(value))))
"""

# Define the system of ODEs resulting from discretization
def odes(t, y, x, Dp, Do, Dv, dx):
    N = len(x) # solutions are stored in a 1-dim array
    p = y[:N]
    o = y[N:2*N]
    v = y[2*N:]

    # Compute first and second derivatives using central difference
    dp_dx = np.gradient(p, dx)
    do_dx = np.gradient(o, dx)
    dv_dx = np.gradient(v, dx)
    d2p_dx2 = np.gradient(dp_dx, dx)
    d2o_dx2 = np.gradient(do_dx, dx)
    d2v_dx2 = np.gradient(dv_dx, dx)

    # Apply Neumann boundary conditions (zero flux)
    dp_dx[0] = 0
    dp_dx[-1] = 0
    do_dx[0] = 0
    do_dx[-1] = 0
    dv_dx[0] = 0
    dv_dx[-1] = 0
    # And smoothness at the boundaries
    d2p_dx2[0] = d2p_dx2[1]
    d2p_dx2[-1] = d2p_dx2[-2]
    d2o_dx2[0] = d2o_dx2[1]
    d2o_dx2[-1] = d2o_dx2[-2]
    d2v_dx2[0] = d2v_dx2[1]
    d2v_dx2[-1] = d2v_dx2[-2]

    # Compute reaction terms
    p_reaction, o_reaction, v_reaction = reaction_terms(p, o, v)

    # Combine diffusion and reaction terms
    Nabla_alpha_p = (l1*d2p_dx2 - 2.*dp_dx*do_dx - o*d2p_dx2 - d2o_dx2*p) / l1
    dpdt = Dp*Nabla_alpha_p + p_reaction
    dvdt = Dv * d2v_dx2 + v_reaction
    dodt = Do * d2o_dx2 + o_reaction

    return np.concatenate([dpdt, dodt, dvdt])


# Initial conditions
gamma = 1.e1
eps = 0.5
def initial_conditions(x):
    # Here, we follow the model and provide a smooth initial distribution of the tumour cell density
    p_0 = p0 * (1. - 1. / (1+ np.exp(-2.*gamma*(x-eps))))  # Initial distribution for p
    o_0 = o0 * np.ones_like(x)
    v_0 = v0 * np.ones_like(x)
    return np.concatenate([p_0, o_0, v_0])

# Spatial domain and time span
x = np.linspace(0, 200, 800) # 200mm divided in 800 points
t_span = (0, 1000) # model for close to 3 years
dx = x[1] - x[0]
y0 = initial_conditions(x)

# Solve the system of ODEs
sol = solve_ivp(odes, t_span, y0, args=(x, Dp, Do, Dv, dx), t_eval=np.linspace(0, 1000, 80), method='RK45')
# sol = solve_ivp(odes, t_span, y0, args=(x, Dp, Do, Dv, dx), method='Radau')

# Extract solutions
p_sol = sol.y[:len(x), :]
o_sol = sol.y[len(x):2*len(x), :]
v_sol = sol.y[2*len(x):, :]

np.save('ModelIII_p.npy', p_sol)
np.save('ModelIII_o.npy', o_sol)
np.save('ModelIII_v.npy', v_sol)



# Plot results
X, T = np.meshgrid(x, sol.t)
np.save('ModelIII_X.npy', X)
np.save('ModelIII_T.npy', T)

fig, ax = plt.subplots(3, 1, figsize=(10, 8))

cfp = ax[0].contourf(X, T, p_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfp, ax=ax[0], label='p')
ax[0].set_title('Solution p(x,t)')
ax[0].set_xlabel('Distance x')
ax[0].set_ylabel('Time t')

cfo = ax[1].contourf(X, T, o_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfo, ax=ax[1], label='o')
ax[1].set_title('Solution o(x,t)')
ax[1].set_xlabel('Distance x')
ax[1].set_ylabel('Time t')

cfv = ax[2].contourf(X, T, v_sol.T, levels=50, cmap='viridis')
fig.colorbar(cfv, ax=ax[2], label='v')
ax[2].set_title('Solution v(x,t)')
ax[2].set_xlabel('Distance x')
ax[2].set_ylabel('Time t')

plt.tight_layout()
plt.savefig('Model_III_Overview.png')
plt.show()

# Plot zoomed-in version
plt.figure()
#plt.figure(figsize=(10, 6))
for i in range(0, 80, 8):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[-1,:], p_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
plt.plot(X[-1,:], p_sol.T[-1,:], label=f"t = {1000:.1f} days")
plt.title("Model III - Cell Density")
plt.xlim(0,10)
#plt.ylim(0,40)
plt.xlabel("Position $x$ in mm")
plt.ylabel("Density $p(x, t)$ in cells per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('Model_III_CellDens.png')
plt.show()


plt.figure()
#plt.figure(figsize=(10, 6))
for i in range(0, 80, 8):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[-1,:], o_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
plt.plot(X[-1,:], o_sol.T[-1,:], label=f"t = {1000:.1f} days")
plt.title("Model III - Oxygen Concentration")
plt.xlim(0,30)
#plt.ylim(0,40)
plt.xlabel("Position $x$ in mm")
plt.ylabel("Density $o(x, t)$ in $nmol$ per $mm$")
plt.legend()
plt.grid(True)
plt.savefig('Model_III_OxConc.png')
plt.show()

plt.figure()
#plt.figure(figsize=(10, 6))
for i in range(0, 80,8):
    #plt.plot(x, p_sol[i], label=f"t = {T[i]:.1f} days")
    plt.plot(X[-1,:], v_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
    #print(i)
plt.plot(X[-1,:], v_sol.T[-1,:], label=f"t = {1000:.1f} days")
plt.title("Model III - Vasculature Density")
plt.xlim(0,30)
#plt.ylim(0,40)
plt.xlabel("Position $x$ in mm")
plt.ylabel("Density $v(x, t)$ per mm")
plt.legend()
plt.grid(True)
plt.savefig('Model_III_VascDens.png')
plt.show()


# Creating Multiple Subplots for Line Plots
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
# Line Plot 1
for i in range(0, 80, 16):
    axes[0].plot(X[-1,:], p_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[0].plot(X[-1,:], p_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[0].set_title('Model III - Cell Density')
axes[0].set_xlim(0,20)
axes[0].set_xlabel("Position $x$ in mm")
axes[0].set_ylabel("Density $p(x, t)$ in cells per $mm$")
axes[0].grid()
axes[0].legend()

for i in range(0, 80, 16):
    axes[1].plot(X[-1,:], o_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[1].plot(X[-1,:], o_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[1].set_title('Model III - Oxygen Concentration')
#axes[1].set_xlim(0,30)
axes[1].set_xlabel("Position $x$ in mm")
axes[1].set_ylabel("Concentration $o(x, t)$ in $nmol$ per $mm$")
axes[1].grid()
axes[1].legend()

for i in range(0, 80, 16):
    axes[2].plot(X[-1,:], v_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[2].plot(X[-1,:], v_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[2].set_title('Model III - Vascular Density')
axes[2].set_xlim(0,30)
axes[2].set_xlabel("Position $x$ in mm")
axes[2].set_ylabel("Density $v(x, t)$ per $mm$")
axes[2].grid()
axes[2].legend()

# Adjusting layout
plt.tight_layout()
 
# Show the plots
plt.savefig('Model_III_Overview_Line.png')
plt.show()

# Creating Multiple Subplots for Line Plots - Cell Density
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
# Line Plot 1
for i in range(0, 80, 16):
    axes[0].plot(X[-1,:], p_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[0].plot(X[-1,:], p_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[0].set_title('Model III - Cell Density')
#axes[0].set_xlim(0,20)
axes[0].set_xlabel("Position $x$ in mm")
axes[0].set_ylabel("Density $p(x, t)$ in cells per $mm$")
axes[0].grid()
axes[0].legend()

for i in range(0, 80, 16):
    axes[1].plot(X[-1,:], p_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[1].plot(X[-1,:], p_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[1].set_title('Model III - Cell Density')
axes[1].set_xlim(0,20)
axes[1].set_xlabel("Position $x$ in mm")
axes[1].set_ylabel("Density $p(x, t)$ in cells per $mm$")
axes[1].grid()
axes[1].legend()

for i in range(0, 80, 16):
    axes[2].plot(X[-1,:], p_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[2].plot(X[-1,:], p_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[2].set_title('Model III - Cell Density')
axes[2].set_xlim(0,15)
axes[2].set_ylim(0,15)
axes[2].set_xlabel("Position $x$ in mm")
axes[2].set_ylabel("Density $p(x, t)$ in cells per $mm$")
axes[2].grid()
axes[2].legend()

# Adjusting layout
plt.tight_layout()
 
# Show the plots
plt.savefig('Model_III_Overview_Cell_Line.png')
plt.show()

# Creating Multiple Subplots for Line Plots - Oxygen Concentration
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
# Line Plot 1
for i in range(0, 80, 16):
    axes[0].plot(X[-1,:], o_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[0].plot(X[-1,:], o_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[0].set_title('Model III - Oxygen Concentration')
#axes[0].set_xlim(0,20)
axes[0].set_xlabel("Position $x$ in mm")
axes[0].set_ylabel("Concentration $o(x, t)$ in cells per $mm$")
axes[0].grid()
axes[0].legend()

for i in range(0, 80, 16):
    axes[1].plot(X[-1,:], o_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[1].plot(X[-1,:], o_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[1].set_title('Model III - Oxygen Concentration')
axes[1].set_xlim(0,100)
axes[1].set_xlabel("Position $x$ in mm")
axes[1].set_ylabel("Concentration $o(x, t)$ in $nmol$ per $mm$")
axes[1].grid()
axes[1].legend()

for i in range(0, 80, 16):
    axes[2].plot(X[-1,:], o_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[2].plot(X[-1,:], o_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[2].set_title('Model III - Oxygen Concentration')
axes[2].set_xlim(0,30)
#axes[2].set_ylim(0,15)
axes[2].set_xlabel("Position $x$ in mm")
axes[2].set_ylabel("Concentration $o(x, t)$ in $nmol$ per $mm$")
axes[2].grid()
axes[2].legend()

# Adjusting layout
plt.tight_layout()
 
# Show the plots
plt.savefig('Model_III_Overview_Oxy_Line.png')
plt.show()


# Creating Multiple Subplots for Line Plots - Vascular Density
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
# Line Plot 1
for i in range(0, 80, 16):
    axes[0].plot(X[-1,:], v_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[0].plot(X[-1,:], v_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[0].set_title('Model III - Vascular Density')
#axes[0].set_xlim(0,20)
axes[0].set_xlabel("Position $x$ in mm")
axes[0].set_ylabel("Density $v(x, t)$ per $mm$")
axes[0].grid()
axes[0].legend()

for i in range(0, 80, 16):
    axes[1].plot(X[-1,:], v_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[1].plot(X[-1,:], v_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[1].set_title('Model III - Vascular Density')
axes[1].set_xlim(0,50)
axes[1].set_xlabel("Position $x$ in mm")
axes[1].set_ylabel("Density $v(x, t)$ per $mm$")
axes[1].grid()
axes[1].legend()

for i in range(0, 80, 16):
    axes[2].plot(X[-1,:], v_sol.T[i,:], label=f"t = {i*1000/80:.1f} days")
axes[2].plot(X[-1,:], v_sol.T[-1,:], label=f"t = {1000:.1f} days")
axes[2].set_title('Model III - Vascular Density')
axes[2].set_xlim(0,25)
#axes[2].set_ylim(0,15)
axes[2].set_xlabel("Position $x$ in mm")
axes[2].set_ylabel("Density $v(x, t)$ per $mm$")
axes[2].grid()
axes[2].legend()

# Adjusting layout
plt.tight_layout()
 
# Show the plots
plt.savefig('Model_III_Overview_Vasc_Line.png')
plt.show()
