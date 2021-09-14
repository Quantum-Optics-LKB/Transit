# -*-coding:utf-8 -*
"""
Created by Tangui Aladjidi on the 29/06/2021
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from bloch_time import temporal_bloch, N
import time
import sys
import scipy.constants as cst
from multiprocessing import Pool
import progressbar
# from numba import jit
from functools import partial


T = 150+273  # cell temp
puiss = 1e-3  # power in W
waist = 1e-3  # beam waist
detun = -3e9  # detuning
L = 10e-3  # cell length
N_grid = 128
N_v = 20
v0 = 40.0
v1 = 800.0
N_real = 10000
N_proc = 16

def main():
    solver = temporal_bloch(T, puiss, waist, detun, L, N_grid=N_grid, N_v=N_v,
                            N_real=N_real, N_proc=N_proc)
    solver1 = temporal_bloch(T, 1e-9, waist, detun, L, N_grid=N_grid, N_v=N_v,
                             N_real=N_real, N_proc=N_proc)
    renorm, counter = solver.do_V_span(v0, v1, N_v)
    plt.close('all')
    renorm1, counter_1 = solver1.do_V_span(v0, v1, N_v)
    # np.save(f'results/pols_long_highp_{N_v}_{N_real}_{time.ctime()}.npy', renorm)
    # np.save(f'results/pols_long_lowp_{N_v}_{N_real}_{time.ctime()}.npy', renorm)
    chi3 = (np.real(renorm) - np.real(renorm1))/solver.I
    n0 = np.sqrt(1 + np.real(renorm1))
    n2 = (3/(4*n0*cst.epsilon_0*cst.c))*chi3
    plt.imshow(n2, origin='lower', norm=colors.SymLogNorm(linthresh=1e-15, linscale=1,
                                              vmin=np.nanmin(n2), vmax=0, base=10))
    plt.title("n2")
    plt.colorbar()
    plt.show()
    # waists = np.linspace(0.25e-3, 2e-3, 5)
    # Ts = np.linspace(90, 150, 5)
    # n2_w_T = np.empty((len(waists), len(Ts), N_grid, N_grid), dtype=np.float32)
    # np.save(f'results/Ts_{time.ctime()}.npy', Ts)
    # np.save(f'results/waists_{time.ctime()}.npy', waists)
    # for counter_w, waist in enumerate(waists):
    #     for counter_T, T in enumerate(Ts):
    #         print(f"Waist {counter_w+1}/{len(waists)}   /// T {counter_T+1}/{len(Ts)}")
    #         solver = temporal_bloch(T, puiss, waist, detun, L, N_grid=N_grid,
    #                                 N_v=N_v, N_real=N_real, N_proc=N_proc)
    #         solver1 = temporal_bloch(T, 1e-9, waist, detun, L, N_grid=N_grid,
    #                                  N_v=N_v, N_real=N_real, N_proc=N_proc)
    #         renorm = run_sim(solver, plot=False)
    #         renorm1 = run_sim(solver1, plot=False)
    #         chi3 = (renorm - renorm1)/solver.I
    #         n0 = np.sqrt(1 + renorm1)
    #         n2 = (3/(4*n0*cst.epsilon_0*cst.c))*chi3
    #         n2_w_T[counter_w, counter_T, :, :] = n2
    #         np.save(f'results/n2_w_T_{time.ctime()}.npy', n2)



if __name__ == "__main__":
    main()