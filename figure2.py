#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3: An immunocompromised person (T0 = 400) with a higher virus
load and faster T-cell death-rate, meaning AIDS occurs in about 5 years (earlier than case 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def run_hiv_model(
    # inital conditions
    T0=600, # healthy T-cells
    T10=0, # latently infected T-cells
    T20=0, # actively infected T-cells
    V0=1e-3, # free virus
    
    #parameters
    s=10, # supply rate of T-cells from the thymus (day^-1mm^-3)
    r=0.03, # growth rate parameter of T-cells (day^-1)
    muT=0.02, # death rate parameter of T-cells (day^-1)
    k1=2.4e-5, # infection rate parameter of the virsu (mm^-3day^-1)
    k2=0.003, # rate of transforming from T1 to T2 (day^-1)
    mub=0.24, # death rate parameter of infected T-cells (day^-1)
    muv=2.4, # viral clearance rate parameter (day^-1)
    N=100, # bursting number parameter
    b=316, #scale of virus in the bloodstream at which saturation takes place (mm^-3)
    Tmax=1500, # max T-cell count

    # time period
    tf=3650, # time period
    n=10000 # distribution
):
    Inits = [T0, T10, T20, V0]

    def model(t, y):
        T, T1, T2, V = y
        growth_constant = (N * mub * t**2) / (b**2 + t)
        dT_dt = s + r * T * (1 - (T + T1 + T2)/Tmax) - muT*T - k1*V*T
        dT1_dt = k1*V*T - muT*T1 - k2*T1
        dT2_dt = k2*T1 - mub*T2
        dV_dt = growth_constant * T2 - k1*V*T - muv*V
        return [dT_dt, dT1_dt, dT2_dt, dV_dt]

    times = np.linspace(0, tf, n)
    sol = solve_ivp(model, [0, tf], Inits, t_eval=times, method='RK45')

    labels = ['Healthy T-cells', 'Latently Infected T-cells', 'Actively Infected T-cells', 'Virus']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Subject 2: Immunocompromised Person')

    ax.axvspan(0, 300, color='lightblue', alpha=0.3, label='Acute Infection')
    ax.axvspan(300, 1450, color='lightgreen', alpha=0.3, label='Chronic Infection')
    ax.axvspan(1450, tf, color='lightcoral', alpha=0.3, label='AIDS Phase')

    for i in range(len(sol.y)):
        ax.plot(sol.t, sol.y[i], label=labels[i])

    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.xlabel('Time (Days)')
    plt.ylabel('Concentration (mm⁻³)')
    plt.grid(True)
    plt.xlim(0, tf)
    plt.tight_layout()
    plt.show()

run_hiv_model(
    T0=400,
    V0=10,
    muT=0.03,
    muv=2.5,
    k1=5e-5,   # higher infection rate
    k2=0.005,  # quicker transition to active infection
)