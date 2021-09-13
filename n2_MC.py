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
from numba import jit
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


# choose_points(plot=False)
# iinit, jinit, ifinal, jfinal = solver.choose_points(plot=False)
# xinit = jinit*solver.window/N_grid
# yinit = iinit*solver.window/N_grid
# xfinal = jfinal*solver.window/N_grid
# yfinal = ifinal*solver.window/N_grid
# # ts = np.arange(0, np.hypot(xfinal-xinit, yfinal-yinit)/200, 5e-9)
# ts = np.arange(0, 2e-6, 5e-10)
# t, y0 = solver.integrate_notransit(40, ts, xinit, yinit, xfinal, yfinal)
# v = 40
# vz = solver.draw_vz(v)
# # vz = -14.394064511361588
# # vz = -6.0
# v_perp = np.sqrt(v**2 - vz**2)
# print(f"{vz=} m/s, {v_perp=} m/s, {solver.k*vz*1e-9/(2*np.pi)} GHz")
# t, y1, path = solver.integrate_notransit(vz, v_perp, iinit, jinit, ifinal, jfinal, np.zeros(8, dtype=np.complex128))
# # t, y1 = solver.integrate_notransit(0, ts, solver.r0, solver.r0, solver.r0, solver.r0, np.zeros(8, dtype=np.complex128))
# # plt.plot(t, y0[:, -2],)
# indices = np.linspace(0, len(t)-1, 1000, dtype=np.uint32)
# # print(y1[indices, -2])
# plt.plot(t[indices]*1e6, np.abs(y1[indices, -2]))
# # plt.legend(["Short", "Long"])
# plt.xlabel("Time in $\\mu s$")
# plt.ylabel("$\\rho_{23}$")
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# fig, ax = plt.subplots()
# counter = np.zeros((N_grid, N_grid), dtype=np.uint16)
# for i in range(N_real):
#     iinit, jinit, ifinal, jfinal = choose_points()
#     path = bresenham(jinit, iinit, jfinal, ifinal)
#     # ax.plot([jinit, jfinal], [iinit, ifinal], color='blue')
#     # ax.scatter([_[0] for _ in path], [_[1] for _ in path], color='gray')
#     for coord in path:
#         counter[coord[0], coord[1]] += 1
# ax.imshow(counter)
# plt.title(f"{N_real} realizations")
# plt.show()


def thread_function(k, solver, counter, v, plot):
    np.random.seed(counter*N_real+k+1)
    iinit, jinit, ifinal, jfinal = solver.choose_points()
    ynext = np.empty(solver.x0.shape, dtype=np.complex128)
    vz = solver.draw_vz(v)
    v_perp = np.sqrt(v**2 - vz**2)
    a, b, path = solver.integrate_notransit(vz, v_perp, iinit, jinit, ifinal,
                                            jfinal, ynext)
    # a, b = solver.integrate_notransit_c(v, t, xinit, yinit, xfinal, yfinal)
    grid = np.zeros((solver.N_grid, solver.N_grid), dtype=np.float32)
    counter_grid = np.zeros((solver.N_grid, solver.N_grid), dtype=np.uint16)
    for coord in path:
        if coord[0] > N_grid-1:
            coord[0] = N_grid-1
        if coord[1] > N_grid-1:
            coord[1] = N_grid-1
        tpath = np.hypot(coord[1]-iinit, coord[0]-jinit)*solver.window/(v_perp*solver.N_grid)
        grid[coord[1], coord[0]] += np.real(b[np.argmin(np.abs(a-tpath)), -2]).astype(np.float32)
        counter_grid[coord[1], coord[0]] += 1
    if plot:
        # velocity unit vector
        xinit = jinit*solver.window/solver.N_grid
        yinit = iinit*solver.window/solver.N_grid
        xfinal = jfinal*solver.window/solver.N_grid
        yfinal = ifinal*solver.window/solver.N_grid
        u0 = xfinal-xinit
        u1 = yfinal-yinit
        norm = np.hypot(u0, u1)
        u0 /= norm
        u1 /= norm
        t_path = np.array([np.hypot(_[1]-iinit, _[0]-jinit)*solver.window/(solver.N_grid*v_perp) for _ in path])
        fig, ax = plt.subplots(1, 3)
        ax[0].plot([jinit, jfinal], [iinit, ifinal], color='red', ls='-')
        ax[0].scatter([_[0] for _ in path], [_[1] for _ in path], color='blue')
        ax[0].scatter((xinit+u0*v_perp*t_path)*(N_grid/solver.window),
                      (yinit+u1*v_perp*t_path)*(N_grid/solver.window), color='green')
        ax[0].scatter(solver.r0*(N_grid/solver.window), solver.r0*(N_grid/solver.window), marker='+')
        ax[0].set_xlim((0, N_grid-1))
        ax[0].set_ylim((0, N_grid-1))
        ax[1].imshow(grid, origin='lower')
        ax[2].imshow(counter_grid, origin='lower')
        fig.suptitle("Realization")
        plt.show()
    return grid, counter_grid


def run_sim(solver, plot=True):
    t0 = time.time()
    vs = np.linspace(40, 800, N_v, dtype=np.float64)
    dv = vs[1]-vs[0]
    pv = ((2/np.pi)**1/2)*((solver.m87/(cst.Boltzmann*solver.T))**(3/2))*vs**2*np.exp(-solver.m87*vs**2/(2*cst.Boltzmann*solver.T))
    pols = np.empty((len(vs), N_grid, N_grid), dtype=np.float32)
    counter_grids_final = np.empty((len(vs), N_grid, N_grid), dtype=np.int32)
    print("Computing the velocity classes ...")
    # computing asynchronously for speed
    with progressbar.ProgressBar(max_value=len(vs)*N_real) as bar:
    # with progressbar.ProgressBar(max_value=len(vs)) as bar:
        for counter, v in enumerate(vs):
            grids = []
            counter_grids = []
            # plot = False
            t_func = partial(thread_function, solver=solver, counter=counter,
                             v=v, plot=False)
            # for loop iterative for debugging
            # for k in range(N_real):
            #     grid, counter_grid = thread_function(k, solver, counter, v, plot=False)
            #     grids.append(grid)
            #     counter_grids.append(counter_grid)
            #     bar.update(counter*N_real + k)
            with Pool(N_proc) as executor:
                for i, _ in enumerate(executor.imap_unordered(t_func,
                                      range(N_real), N_real//N_proc)):
                    bar.update(counter*N_real + i)
                    grids.append(_[0])
                    counter_grids.append(_[1])
            # grid, counter_grid = solver.do_N_real(v)
            grids = np.asarray(grids)
            counter_grids = np.asarray(counter_grids)
            pols[counter, :, :] = np.sum(grids, axis=0)
            counter_grids_final[counter, :, :] = np.sum(counter_grids, axis=0)
            # pols[counter, :, :] = np.real(grid)
            # counter_grids_final[counter, :, :] = counter_grid
            # plt.imshow(np.abs(counter_grid))
            # plt.show()
            bar.update(counter)
    t1 = time.time()-t0
    print(f"\nTime elapsed {t1} s")
    # pols = np.load('pols_long_1000.npy')
    pols = np.divide(pols, counter_grids_final, out=np.zeros_like(pols),
                     where=counter_grids_final != 0)
    pols = np.swapaxes(pols, 0, 1)
    pols = np.swapaxes(pols, 1, 2)
    renorm = np.sum(dv*pols*pv, axis=2)  #/np.sum(counter_grids_final, axis=0)
    renorm *= 2*N(solver.T) * np.abs(solver.mu23) / (solver.E * cst.epsilon_0)
    if plot:
        fig, ax = plt.subplots(1, 3)
        im0 = ax[0].imshow(np.sum(pols*pv, axis=2), origin='lower')
        # im0 = ax[0].imshow(pols[:, :, 0], origin='lower')
        ax[0].set_title("Polarization")
        im1 = ax[1].imshow(np.sum(counter_grids_final, axis=0), origin='lower')
        # im1 = ax[1].imshow(counter_grids_final[0, :, :], origin='lower')
        ax[1].set_title("Counts")
        im2 = ax[2].imshow(renorm, origin='lower')
        ax[2].set_title("Renormalized")
        fig.colorbar(im0, ax=ax[0])
        fig.colorbar(im1, ax=ax[1])
        fig.colorbar(im2, ax=ax[2])
        plt.show()
    return renorm


if __name__ == "__main__":
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
