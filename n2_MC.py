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
puiss = 1.0  # power in W
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
    # solver = temporal_bloch(T, puiss, waist, detun, L, N_grid=N_grid, N_v=N_v,
    #                         N_real=N_real, N_proc=N_proc)
    # solver1 = temporal_bloch(T, 1e-9, waist, detun, L, N_grid=N_grid, N_v=N_v,
    #                          N_real=N_real, N_proc=N_proc)
    # solver.window = 8*waist
    # solver.r0 = solver.window/2
    # solver1.window = solver.window
    # solver1.r0 = solver.r0
    # print("High power run ...")
    # renorm, counter = solver.do_V_span(v0, v1, N_v)
    # print("Low power run ...")
    # renorm1, counter_1 = solver1.do_V_span(v0, v1, N_v)
    # # np.save(f'results/pols_long_highp_{N_v}_{N_real}_{time.ctime()}.npy', renorm)
    # # np.save(f'results/pols_long_lowp_{N_v}_{N_real}_{time.ctime()}.npy', renorm)
    # chi3 = (np.real(renorm) - np.real(renorm1))/solver.I
    # n0 = np.sqrt(1 + np.real(renorm1))
    # n2 = (3/(4*n0*cst.epsilon_0*cst.c))*chi3
    # avg_zone = 10
    # n2_center = np.mean(n2[N_grid//2-avg_zone:N_grid//2+avg_zone,
    #                        N_grid//2-avg_zone:N_grid//2+avg_zone])
    # print(f"Center n2 : {n2_center}")
    # fig, ax = plt.subplots(1, 3)
    # im0 = ax[0].imshow(np.real(renorm))
    # im1 = ax[1].imshow(np.real(renorm1))
    # im2 = ax[2].imshow(n2, origin='lower', norm=colors.SymLogNorm(linthresh=1e-15, linscale=1,
    #                                           vmin=np.nanmin(n2), vmax=0, base=10))

    # ax[0].set_title("$\\rho_{32}$ High Power run")
    # ax[1].set_title("$\\rho_{32}$ Low Power run")
    # ax[2].set_title("n2")
    # fig.colorbar(im0, ax=ax[0])
    # fig.colorbar(im1, ax=ax[1])
    # fig.colorbar(im2, ax=ax[2])

    # plt.show()
    waists = np.linspace(50e-6, 2e-3, 10)
    powers = np.linspace(50e-3, 1.5, 10)
    # Ts = np.linspace(90, 150, 5)
    # n2_w_T = np.empty((len(waists), N_grid, N_grid), dtype=np.float64)
    n2_center = np.empty((len(waists), len(powers)), dtype=np.float64)
    # np.save(f'results/Ts_{time.ctime()}.npy', Ts)
    start_time = time.ctime()
    np.save(f'results/waists_{start_time}.npy', waists)
    # fig, ax = plt.subplots(4, 5)
    # fig1, ax1 = plt.subplots()
    for counter_w, waist in enumerate(waists):
        for counter_p, power in enumerate(powers):
            print(f"Waist {counter_w+1}/{len(waists)} --- Waist {counter_p+1}/{len(powers)}")
            solver = temporal_bloch(T, power, waist, detun, L, N_grid=N_grid,
                                    N_v=N_v, N_real=N_real, N_proc=N_proc)
            solver1 = temporal_bloch(T, 1e-9, waist, detun, L, N_grid=N_grid,
                                        N_v=N_v, N_real=N_real, N_proc=N_proc)
            solver.window = 10*waist
            solver.r0 = solver.window/2
            solver1.window = solver.window
            solver1.r0 = solver.r0
            renorm, counter = solver.do_V_span(v0, v1, N_v)
            renorm1, counter_1 = solver1.do_V_span(v0, v1, N_v)
            chi3 = (np.real(renorm) - np.real(renorm1))/solver.I
            n0 = np.sqrt(1 + np.real(renorm1))
            n2 = (3/(4*n0*cst.epsilon_0*cst.c))*chi3
            avg_zone = 10
            n2_c = np.mean(n2[N_grid//2-avg_zone:N_grid//2+avg_zone,
                                N_grid//2-avg_zone:N_grid//2+avg_zone])
            # n2_w_T[counter_w, :, :] = n2
            np.save(f'results/n2_w{counter_w}_P{counter_p}_{start_time}.npy', n2)
            n2_center[counter_w, counter_p] = n2_c
    np.save(f"results/n2_center_w_P_{start_time}.npy", n2_center)
    #     im = ax[counter_w//5, counter_w%5].imshow(np.abs(n2))
    #     ax[counter_w//5, counter_w%5].set_title("$w_{0}$ = "+
    #                                             f"{np.round(waist*1e3, decimals=2)} mm")
    #     fig.colorbar(im, ax=ax[counter_w//5, counter_w%5])
    # ax1.plot(waists*1e3, np.abs(n2_center))
    # ax1.set_title("n2 vs waist")
    # ax1.set_xlabel("waist in mm")
    # ax1.set_ylabel("abs(n2) in $m^{2} / W$")
    # plt.show()
if __name__ == "__main__":
    main()