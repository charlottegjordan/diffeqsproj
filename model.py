#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Parameters
s = 10           # (day^-1 mm^-3)
r = 0.03         # (day^-1)
muT = 0.02       # (day^-1)
k1 = 2.4e-5      # (mm^3 day^-1)
k2 = 0.003       # (day^-1)
mub = 0.24       # (day^-1)
muv = 2.4        # (day^-1)
N = 100          # bursting number
b = 316          # (mm^-3)

# Initial Conditions
T0 = 600         # healthy T-cells
T10 = 0          # latently infected T-cells
T20 = 0          # actively infected T-cells
V0 = 1e-3        # free virus
Inits = [T0, T10, T20, V0]

# System of ODEs
def model(t, y):
    T, T1, T2, V = y
    Tmax = 1500  # optional carrying capacity (if needed)
    growth_constant = (N * mub * t**2) / (b**2 + t)
    
    dT_dt = s + r * T * (1 - (T + T1 + T2)/Tmax) - muT*T - k1*V*T
    dT1_dt = k1*V*T - muT*T1 - k2*T1
    dT2_dt = k2*T1 - mub*T2
    dV_dt = growth_constant * T2 - k1*V*T - muv*V

    return [dT_dt, dT1_dt, dT2_dt, dV_dt]

# Time span
t0 = 0
tf = 3650
n = 10000
times = np.linspace(t0, tf, n)

# Solve the system
sol = solve_ivp(model, [t0, tf], Inits, t_eval=times, method='RK45')

# Plotting
labels = ['Healthy T-cells', 'Latently Infected T-cells', 'Actively Infected T-cells', 'Virus']
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('HIV Infection Model with Stages')

# Shaded regions
ax.axvspan(0, 300, color='lightblue', alpha=0.3, label='Acute Infection')
ax.axvspan(300, 2500, color='lightgreen', alpha=0.3, label='Chronic Infection')
ax.axvspan(2500, 3650, color='lightcoral', alpha=0.3, label='AIDS Phase')

# Plot each solution curve
for i in range(len(sol.y)):
    ax.plot(sol.t, sol.y[i], label=labels[i])

# Legend handling
handles, lbls = ax.get_legend_handles_labels()
by_label = dict(zip(lbls, handles))
ax.legend(by_label.values(), by_label.keys())

plt.xlabel('Time (Days)')
plt.ylabel('Concentration (mm⁻³)')
plt.grid(True)
plt.xlim(0, 3650)
plt.tight_layout()
plt.show()