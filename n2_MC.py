# -*-coding:utf-8 -*
"""
Created by Tangui Aladjidi on the 29/06/2021
"""

import sys
import time
# from numba import jit
from functools import partial
from multiprocessing import Pool
from scipy.constants import Boltzmann
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy.constants as cst
from scipy.optimize import curve_fit

from bloch_time import N, temporal_bloch

T = 150+273  # cell temp
puiss = 560e-3  # power in W
waist = 1.85e-3  # beam waist
I = puiss/(np.pi*waist**2)
detun = -2.2e9  # detuning
L = 10e-3  # cell length
N_grid = 128
N_v = 20
v0 = 40.0
v1 = 800.0
N_real = 5000
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
    waists = np.linspace(50e-6, 2e-3, 20)
    powers = np.linspace(50e-3, 540e-3, 10)
    power_in_murad = np.asarray(
        [330e-3, 330e-3, 440e-3, 380e-3, 416e-3, 424e-3, 412e-3, 375e-3, 375e-3, 382e-3])
    power_out_murad = np.asarray(
        [240e-3, 240e-3, 328e-3, 265e-3, 270e-3, 274e-3, 250e-3, 255e-3, 255e-3, 233e-3])
    power_in_low_murad = np.asarray(
        [2.2e-3, 2.2e-3, 550e-6, 500e-6, 100e-6, 100e-6, 1e-3, 0.9e-3, 0.9e-3, 0.9e-3])
    power_out_low_murad = np.asarray(
        [1.3e-3, 1.3e-3, 298e-6, 250e-6, 60e-6, 60e-6, 600e-6, 0.6e-3, 0.6e-3, 0.6e-3])
    waists_murad = np.asarray([0.054e-3, 8.350e-4, 9.400e-05, 2.050e-04, 3.680e-04, 8.900e-04, 0.00130,
                               0.00186, 0.00390, 0.00250], dtype=np.float64)
    idx_sorted = np.argsort(waists_murad)
    n2_murad = np.asarray([1.466451787760415e-10, 6.146519158111747e-09, 4.648785173881415e-10,
                           1.296385036598109e-09, 2.742352966926649e-09, 4.018992194568556e-09,
                           6.691264052482064e-09, 9.751877517360527e-09, 1.445521229964598e-08,
                           1.544469999857501e-08], dtype=np.float64)
    I_sat_murad = np.asarray([5.602997069021627e+02, 11.770124366553920, 2.053285829870257e+02,
                              60.502837888731115, 30.902688270484745, 58.693562254310365,
                              11.395745635138198, 6.867638307604812, 5.469433881999493,
                              2.784375157084676], dtype=np.float64)*1e4
    n2_err = np.array([0.15, 4, 0.4, 2, 1.2, 3.1, 3, 3.5, 6, 9.3])*1e-10
    Isat_err = np.array([70, 1.42, 25, 9.3, 2, 10, 1.4, 1.32, 0.6, 0.4])*1e4
    waists_murad = waists_murad[idx_sorted]
    n2_murad = n2_murad[idx_sorted]
    I_sat_murad = I_sat_murad[idx_sorted]
    power_in_murad = power_in_murad[idx_sorted]
    power_out_murad = power_out_murad[idx_sorted]
    power_in_low_murad = power_in_low_murad[idx_sorted]
    power_out_low_murad = power_out_low_murad[idx_sorted]
    # Ts = np.linspace(90, 150, 5)
    # n2_w_T = np.empty((len(waists), N_grid, N_grid), dtype=np.float64)
    # n2_center = np.empty((len(waists), len(powers)), dtype=np.float64)
    # np.save(f'results/Ts_{time.ctime()}.npy', Ts)
    # start_time = time.ctime()
    # np.save(f'results/waists_{start_time}.npy', waists)
    # fig, ax = plt.subplots(2, 5)
    # fig1, ax1 = plt.subplots()
    # for counter_w, waist in enumerate(waists):
    #     for counter_p, power in enumerate(powers):
    #         print(f"Waist {counter_w+1}/{len(waists)} --- Power {counter_p+1}/{len(powers)}")
    #         solver = temporal_bloch(T, power, waist, detun, L, N_grid=N_grid,
    #                                 N_v=N_v, N_real=N_real, N_proc=N_proc)
    #         solver1 = temporal_bloch(T, 1e-9, waist, detun, L, N_grid=N_grid,
    #                                     N_v=N_v, N_real=N_real, N_proc=N_proc)
    #         solver.window = 10*waist
    #         solver.r0 = solver.window/2
    #         solver1.window = solver.window
    #         solver1.r0 = solver.r0
    #         renorm, counter = solver.do_V_span(v0, v1, N_v)
    #         renorm1, counter_1 = solver1.do_V_span(v0, v1, N_v)
    #         chi3 = (np.real(renorm) - np.real(renorm1))/solver.I
    #         n0 = np.sqrt(1 + np.real(renorm1))
    #         n2 = (3/(4*n0*cst.epsilon_0*cst.c))*chi3
    #         avg_zone = 10
    #         n2_c = np.mean(n2[N_grid//2-avg_zone:N_grid//2+avg_zone,
    #                             N_grid//2-avg_zone:N_grid//2+avg_zone])
    #         # n2_w_T[counter_w, :, :] = n2
    #         np.save(f'results/n2_w{counter_w}_P{counter_p}_{start_time}.npy', n2)
    #         n2_center[counter_w, counter_p] = n2_c
    # np.save(f"results/n2_center_w_P_{start_time}.npy", n2_center)
    #     im = ax[counter_w//5, counter_w%5].imshow(np.abs(n2))
    #     ax[counter_w//5, counter_w%5].set_title("$w_{0}$ = "+
    #                                             f"{np.round(waist*1e3, decimals=2)} mm")
    #     fig.colorbar(im, ax=ax[counter_w//5, counter_w%5])
    # ax1.plot(waists*1e3, np.abs(n2_center))
    # ax1.set_title("n2 vs waist")
    # ax1.set_xlabel("waist in mm")
    # ax1.set_ylabel("abs(n2) in $m^{2} / W$")
    # plt.show()
    # Waist run
    # fig, ax = plt.subplots(2, 5)
    # fig1, ax1 = plt.subplots()
    # indices = range(10)
    # n2_w_murad = np.empty(
    #     (len(waists_murad[indices]), N_grid, N_grid), dtype=np.float64)
    # n2_center = np.empty(len(waists_murad[indices]), dtype=np.float64)
    # for counter_w, waist in enumerate(waists_murad[indices]):
    #     print(f"Waist {counter_w+1}/{len(waists_murad[indices])}")
    #     solver = temporal_bloch(T, I*(np.pi*waist**2), waist, detun, L, N_grid=N_grid,
    #                             N_v=N_v, N_real=N_real, N_proc=N_proc)
    #     solver1 = temporal_bloch(T, 1e-9, waist, detun, L, N_grid=N_grid,
    #                              N_v=N_v, N_real=N_real, N_proc=N_proc)
    #     solver.window = 4*waist
    #     solver1.window = solver.window
    #     renorm, counter = solver.do_V_span(v0, v1, N_v)
    #     renorm1, counter_1 = solver1.do_V_span(v0, v1, N_v)
    #     # alpha = -np.log(0.6)/L
    #     # Itilde = solver.I*(1-np.exp(alpha*L))/(alpha*L)
    #     nlow = np.sqrt(1 + renorm1)
    #     nhigh = np.sqrt(1 + renorm)
    #     Dn = np.real(nhigh-nlow)
    #     n2 = Dn/solver.I
    #     avg_zone = 10
    #     n2_c = np.mean(n2[N_grid//2-avg_zone:N_grid//2+avg_zone,
    #                       N_grid//2-avg_zone:N_grid//2+avg_zone])
    #     # n2_c = np.mean(n2)
    #     n2_w_murad[counter_w, :, :] = n2
    #     np.save(f'results/n2_w{counter_w}_murad_{start_time}.npy', n2)
    #     n2_center[counter_w] = n2_c
    # np.save(f"results/n2_center_w_P_{start_time}.npy", n2_center)
    # for counter_w, waist in enumerate(waists_murad[indices]):
    #     im = ax[counter_w//5, counter_w % 5].imshow(np.abs(n2_w_murad[counter_w, :, :]),
    #                                                 vmin=np.nanmin(
    #         np.abs(n2_w_murad)),
    #         vmax=np.nanmax(np.abs(n2_w_murad)))
    #     ax[counter_w//5, counter_w % 5].set_title("$w_{0}$ = " +
    #                                               f"{np.round(waist*1e3, decimals=2)} mm")
    #     fig.colorbar(im, ax=ax[counter_w//5, counter_w % 5])
    # # n2_center = np.load("results/n2_center_w_P_Fri Jan  7 13:29:56 2022.npy")
    # ax1.scatter(waists_murad[indices]*1e3, np.abs(n2_center))
    # ax1.scatter(waists_murad[indices]*1e3, n2_murad[indices], marker='x')
    # ax1.legend(["Computed", "Data"])
    # ax1.set_title("$\Delta n /I$ vs waist")
    # ax1.set_xlabel("Beam waist in mm")
    # ax1.set_ylabel("$|\Delta n / I|$ in $m^{2} / W$")
    # # ax1.set_yscale("log")
    # # ax1.set_xscale("log")
    # fig1.tight_layout()
    # plt.show()
    # Power run
    indices = range(10)
    for idx in indices:
        start_time = time.ctime()
        fig, ax = plt.subplots(4, 5)
        fig1, ax1 = plt.subplots()
        Dn_P_murad = np.zeros((len(powers), N_grid, N_grid), dtype=np.float64)
        Dn_center = np.zeros(len(powers), dtype=np.float64)
        Dn_analytical = np.zeros(len(powers), dtype=np.float64)
        solver1 = temporal_bloch(T, 1e-9*np.pi*waists_murad[idx], waists_murad[idx], detun, L, N_grid=N_grid,
                                 N_v=N_v, N_real=N_real, N_proc=N_proc)
        solver1.window = 10*waists_murad[idx]
        Vs = np.linspace(v0, v1, N_v)
        pvs = np.sqrt(2.0/np.pi)*((solver1.m87/(Boltzmann*T))**(3.0/2.0)) * \
            Vs**2.0*np.exp(-solver1.m87*Vs**2.0/(2.0*Boltzmann*T))
        chi_analytical_low = solver1.chi_analytical(0)
        t0 = time.time()
        for counter_p, power in enumerate(powers):
            solver = temporal_bloch(T, power, waists_murad[idx], detun, L, N_grid=N_grid,
                                    N_v=N_v, N_real=N_real, N_proc=N_proc)
            N_steps_z = 10
            dz = solver.L/N_steps_z
            Dn_c = 0
            alpha = - \
                np.log(0.6)/solver.L
            renorm1, counter_1 = solver1.do_V_span(v0, v1, N_v)
            chi_analytical_low = np.sum(
                solver1.chi_analytical(Vs)*pvs*np.abs(Vs[1]-Vs[0]))
            nlow = np.sqrt(1 + renorm1)
            nlow_a = np.sqrt(1 + chi_analytical_low)
            solver.window = solver1.window
            for k in range(N_steps_z):
                print(
                    f"Power {counter_p+1}/{len(powers)} ---- z {k+1}/{N_steps_z}")
                renorm, counter = solver.do_V_span(v0, v1, N_v)
                chi_analytical_high = np.sum(
                    solver.chi_analytical(Vs)*pvs*np.abs(Vs[1]-Vs[0]))
                nhigh_a = np.sqrt(1 + chi_analytical_high)
                nhigh = np.sqrt(1 + renorm)
                Dn_a = np.real(nhigh_a - nlow_a)
                Dn = np.real(nhigh-nlow)
                # print(f"{n2_a=}")
                avg_zone = 5
                Dn_c = np.mean(Dn[N_grid//2-avg_zone:N_grid//2+avg_zone,
                                  N_grid//2-avg_zone:N_grid//2+avg_zone])
                Dn_P_murad[counter_p, :, :] += Dn/N_steps_z
                Dn_center[counter_p] += Dn_c/N_steps_z
                Dn_analytical[counter_p] += Dn_a/N_steps_z
                solver.puiss *= np.exp(-alpha*dz)
            np.save(f'results/Dn_w{counter_p}_murad_{start_time}.npy', Dn)
        np.save(f"results/Dn_center_w_P_{start_time}.npy", Dn_center)
        np.save(
            f"results/Dn_center_w_P_{start_time}_analytical.npy", Dn_analytical)
        print(f"Time spent to loop powers : {time.time()-t0} s")
        # Dn_center = np.load("results/Dn_center_w_P_Wed May  4 14:00:37 2022.npy")
        # Dn_analytical = np.load(
        #     "results/Dn_center_w_P_Wed May  4 14:00:37 2022_analytical.npy")
        # for counter_p, power in enumerate(powers):
        #     Dn_P_murad[counter_p, :, :] = np.load(
        #         f'results/Dn_w{counter_p}_murad_Wed May  4 14:00:37 2022.npy')
        # for counter_p, power in enumerate(powers):
        #     im = ax[counter_p//5, counter_p % 5].imshow(np.abs(Dn_P_murad[counter_p, :, :]),
        #                                                 vmin=np.nanmin(
        #         np.abs(Dn_P_murad)),
        #         vmax=np.nanmax(np.abs(Dn_P_murad)))
        #     ax[counter_p//5, counter_p % 5].set_title("$P_{0}$ = " +
        #                                               f"{np.round(power*1e3, decimals=2)} mW")
        #     fig.colorbar(im, ax=ax[counter_p//5, counter_p % 5])

        def fit_Isat(P, n2_0, Isat):
            I = P/(np.pi*waists_murad[idx]**2)
            alpha = -np.log(0.6)/L
            Itilde = I*(1-np.exp(-alpha*L))/(alpha*L)
            Itilde *= 1/(1+Itilde/Isat)
            return n2_0*Itilde
        popt, pcov = curve_fit(fit_Isat, powers, Dn_center,
                               p0=[Dn_center[0]/I, 1.0], maxfev=16000)
        popt1, pcov1 = curve_fit(fit_Isat, powers, Dn_analytical,
                                 p0=[Dn_center[0]/I, 1.0], maxfev=16000)
        print(popt)
        # ax1.plot(powers*1e3, np.abs(Dn_center))
        # ax1.plot(powers*1e3, np.abs(fit_Isat(powers, popt[0], popt[1])))
        # ax1.plot(powers*1e3, np.abs(fit_Isat(powers,
        #                                     n2_murad[idx], I_sat_murad[idx])))
        # ax1.plot(powers*1e3, np.abs(Dn_analytical))
        # leg1 = "Fit : $n_{2}$ = "+"{:.2e}".format(popt[0])+" $m^{2}/W$, " +\
        #     "$I_{sat}$ = "+f"{np.round(popt[1], decimals=2)} "+"$W/m^{2}$"
        # leg2 = "Data : $n_{2}$ = "+"{:.2e}".format(-n2_murad[idx])+" $m^{2}/W$, " +\
        #     "$I_{sat}$ = "+f"{np.round(I_sat_murad[idx], decimals=2)} "+"$W/m^{2}$"
        # leg3 = "Analytical : $n_{2}$ = "+"{:.2e}".format(popt1[0])+" $m^{2}/W$, " +\
        #     "$I_{sat}$ = "+f"{np.round(popt1[1], decimals=2)} "+"$W/m^{2}$"
        # ax1.legend(["Computed", leg1, leg2, leg3])
        # ax1.set_title("$\Delta n$ vs power, $w_{0}$ = " +
        #             f"{np.round(waists_murad[idx]*1e3, decimals=2)} mm")
        # fig.suptitle("$\Delta n$ vs power, $w_{0}$ = " +
        #             f"{np.round(waists_murad[idx]*1e3, decimals=2)} mm")
        # ax1.set_xlabel("Power in mW")
        # ax1.set_ylabel("$\Delta n$")
        # # ax1.set_yscale('log')
        # plt.show()
    # realizations run
    # reals = [1000, 2000, 5000, 10000, 50000, 100000]
    # n2_center = np.empty(len(reals), dtype=np.float64)
    # # chi_analytical_low = np.sum(solver1.chi_analytical(vs)*pvs)
    # t0 = time.time()
    # for counter_r, n_real in enumerate(reals):
    #     print(f"Real {counter_r+1}/{len(reals)}")
    #     solver = temporal_bloch(T, 1.0, 1e-3, detun, L, N_grid=N_grid,
    #                             N_v=N_v, N_real=n_real, N_proc=N_proc)
    #     solver1 = temporal_bloch(T, 1e-9, 1e-3, detun, L, N_grid=N_grid,
    #                             N_v=N_v, N_real=n_real, N_proc=N_proc)
    #     solver1.window = 8e-3
    #     solver1.r0 = solver1.window/2
    #     renorm1, counter_1 = solver1.do_V_span(v0, v1, N_v)
    #     solver.window = solver1.window
    #     solver.r0 = solver1.r0
    #     renorm, counter = solver.do_V_span(v0, v1, N_v)
    #     chi3 = (np.real(renorm) - np.real(renorm1))/solver.I
    #     n0 = np.sqrt(1 + np.real(renorm1))
    #     n2 = (3/(4*n0*cst.epsilon_0*cst.c))*chi3
    #     avg_zone = 5
    #     n2_c = np.mean(n2[N_grid//2-avg_zone:N_grid//2+avg_zone,
    #                         N_grid//2-avg_zone:N_grid//2+avg_zone])
    #     n2_center[counter_r] = n2_c
    # np.save(f"results/n2_center_real_{start_time}.npy", n2_center)
    # print(f"Time spent to loop reals : {time.time()-t0} s")
    # Waist run pops
    # w_number = 10
    # fig, ax = plt.subplots(2, 5)
    # fig1, ax1 = plt.subplots()
    # Pops = np.empty((len(waists_murad[0:w_number]), N_grid, N_grid), dtype=np.complex128)
    # Cohs = np.empty((len(waists_murad[0:w_number]), N_grid, N_grid), dtype=np.complex128)
    # pops_c = np.empty(len(waists_murad[0:w_number]), dtype=np.complex128)
    # cohs_c = np.empty(len(waists_murad[0:w_number]), dtype=np.complex128)
    # for counter_w, waist in enumerate(waists_murad[0:w_number]):
    #     print(f"Waist {counter_w+1}/{len(waists_murad[0:w_number])}")
    #     solver = temporal_bloch(T, I*(np.pi*waist**2), waist, detun, L, N_grid=N_grid,
    #                             N_v=N_v, N_real=N_real, N_proc=N_proc)
    #     solver.window =8*waist
    #     solver.r0 = solver.window/2
    #     Pops[counter_w, :, :], Cohs[counter_w, :, :], counter = solver.do_V_span_pop(v0, v1, N_v)
    #     avg_zone = 10
    #     pops_c[counter_w] = np.mean(Pops[counter_w, N_grid//2-avg_zone:N_grid//2+avg_zone,
    #                         N_grid//2-avg_zone:N_grid//2+avg_zone])
    #     cohs_c[counter_w] = np.mean(Cohs[counter_w, N_grid//2-avg_zone:N_grid//2+avg_zone,
    #     N_grid//2-avg_zone:N_grid//2+avg_zone])
    # np.save(f"results/pops_center_{start_time}.npy", pops_c)
    # np.save(f"results/cohs_center_{start_time}.npy", cohs_c)
    # for counter_w, waist in enumerate(waists[0:w_number]):
    #     im = ax[counter_w//5, counter_w%5].imshow(np.real(Pops[counter_w, :, :]),
    #                                               vmin=np.nanmin(np.real(Pops)),
    #                                               vmax=np.nanmax(np.real(Pops)))
    #     ax[counter_w//5, counter_w%5].set_title("$w_{0}$ = "+
    #                                             f"{np.round(waist*1e3, decimals=2)} mm")
    #     fig.colorbar(im, ax=ax[counter_w//5, counter_w%5])
    # # pops_c = np.load("results/pops_center_Mon Jan  3 17:39:47 2022.npy")
    # # cohs_c = np.load("results/cohs_center_Mon Jan  3 17:39:47 2022.npy")
    # ax1b = ax1.twinx()
    # ax1.scatter(waists_murad[0:w_number]*1e3, np.abs(pops_c), label="$|\\rho_{33}|$", color='#1f77b4')
    # ax1b.scatter(waists_murad[0:w_number]*1e3, np.real(cohs_c), label="$Re(\\rho_{23})$", color='#ff7f0e')
    # ax1.set_title("Center population and coherences vs waist")
    # ax1.set_xlabel("waist in mm")
    # ax1.set_ylabel("Density matrix element")
    # ax1.set_yscale('log')
    # ax1b.set_yscale('log')
    # ax1.legend(loc="upper left")
    # ax1b.legend(loc="upper right")
    # plt.show()
    # Waist run with propagation
    # indices = range(10)
    # start_time = time.ctime()
    # # indices = [0, 3, 4, 5, 6]
    # waists_subset = waists_murad[indices]
    # powers_subset = power_in_murad[indices]
    # Dn_w_murad = np.zeros(
    #     (len(waists_murad[indices]), N_grid, N_grid), dtype=np.float64)
    # Dn_center = np.zeros(len(waists_murad[indices]), dtype=np.float64)
    # for counter_w, waist in enumerate(waists_murad[indices]):
    #     # for counter_w in indices:
    #     solver = temporal_bloch(T, I*np.pi*waist**2, waist, detun, L, N_grid=N_grid,
    #                             N_v=N_v, N_real=N_real, N_proc=N_proc)
    #     solver1 = temporal_bloch(T, 1e-9*(np.pi*waist**2), waist, detun, L, N_grid=N_grid,
    #                              N_v=N_v, N_real=N_real, N_proc=N_proc)
    #     solver.window = 10*waist
    #     solver1.window = solver.window
    #     N_steps_z = 10
    #     dz = solver.L/N_steps_z
    #     Dn_c = 0
    #     alpha = - \
    #         np.log(power_out_low_murad[indices][counter_w] /
    #                power_in_low_murad[indices][counter_w])/solver.L
    #     renorm1, counter_1 = solver1.do_V_span(v0, v1, N_v)
    #     nlow = np.sqrt(1 + renorm1)
    #     for k in range(N_steps_z):
    #         print(
    #             f"Waist {counter_w+1}/{len(waists_murad[indices])} z {k+1}/{N_steps_z}")
    #         renorm, counter = solver.do_V_span(v0, v1, N_v)
    #         nhigh = np.sqrt(1 + renorm)
    #         Dn = np.real(nhigh-nlow)
    #         avg_zone = 2
    #         Dn_c += np.mean(Dn[N_grid//2-avg_zone:N_grid//2+avg_zone,
    #                            N_grid//2-avg_zone:N_grid//2+avg_zone])/N_steps_z
    #         Dn_w_murad[counter_w, :, :] += Dn/N_steps_z
    #         solver.puiss *= np.exp(-alpha*dz)
    #         np.save(f'results/Dn_w{counter_w}_z{k}_murad_{start_time}.npy', Dn)
    #     Dn_center[counter_w] = Dn_c
    # np.save(f"results/n2_center_w_P_{start_time}.npy", Dn_center)
    # fig, ax = plt.subplots(2, 5)
    # fig1, ax1 = plt.subplots()
    # for counter_w, waist in enumerate(waists_murad[indices]):
    #     im = ax[counter_w//5, counter_w % 5].imshow(np.abs(Dn_w_murad[counter_w, :, :]),
    #                                                 norm=colors.LogNorm(vmin=np.nanmin(np.abs(Dn_w_murad)),
    #                                                                     vmax=np.nanmax(np.abs(Dn_w_murad))))
    #     ax[counter_w//5, counter_w % 5].set_title("$w_{0}$ = " +
    #                                               f"{np.round(waist*1e3, decimals=2)} mm")
    #     fig.colorbar(im, ax=ax[counter_w//5, counter_w % 5])
    # n2_center = np.load("results/n2_center_w_P_Thu Jan  6 14:13:10 2022.npy")
    # alpha = -np.log(power_out_low_murad[counter_w] /
    #                 power_in_low_murad[counter_w])/L
    # Itilde = I*(1-np.exp(-alpha*L))/(alpha*L)
    # Itilde *= 1/(1+Itilde/I_sat_murad[indices])
    # ax1.scatter(waists_murad[indices]*1e3, np.abs(Dn_center))
    # ax1.scatter(waists_murad[indices]*1e3,
    #             n2_murad[indices]*Itilde, marker='x')
    # ax1.legend(["Computed", "Data"])
    # ax1.set_title("$\Delta n$ vs waist")
    # ax1.set_xlabel("Beam waist in mm")
    # ax1.set_ylabel("$\Delta n$")
    # # ax1.set_ylabel("$|n_{2}|$ in $m^{2} / W$")
    # # ax1.set_yscale("log")
    # # ax1.set_xscale("log")
    # fig1.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
