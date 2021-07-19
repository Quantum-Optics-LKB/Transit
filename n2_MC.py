# -*-coding:utf-8 -*
"""
Created by Tangui Aladjidi on the 29/06/2021
"""

import numpy as np
import matplotlib.pyplot as plt
from bloch_time import temporal_bloch
import time
import sys
import scipy.constants as cst
from multiprocessing import Pool
from bresenham import bresenham
import progressbar


T = 150+273  # cell temp
puiss = 1  # power in W
waist = 0.5e-3  # beam waist
detun = -1e-9  # detuning
L = 10e-3  # cell length
solver = temporal_bloch(T, puiss, waist, detun, L)
vs = np.linspace(40, 800, 20, dtype=np.float64)
dv = vs[1]-vs[0]
pv = ((2/np.pi)**1/2)*((solver.m87/(cst.Boltzmann*solver.T))**(3/2))*vs**2*np.exp(-solver.m87*vs**2/(2*cst.Boltzmann*solver.T))
# plt.plot(vs, pv)
# plt.show()
N = 128
N_real = 20
N_proc = 15

def choose_points():
    edges = []
    for i in range(N-1):
        edges.append((0, i))
        edges.append((N-1, i))
        edges.append((i, 0))
        edges.append((i, N-1))
    iinit, jinit = edges[np.random.randint(0, len(edges)-1)]
    ifinal, jfinal = iinit, jinit
    while (ifinal == iinit or jfinal == jinit):
        ifinal, jfinal = edges[np.random.randint(0, len(edges)-1)]
    return iinit, jinit, ifinal, jfinal

# iinit, jinit, ifinal, jfinal = choose_points()
# xinit = jinit*2*solver.waist/N
# yinit = iinit*2*solver.waist/N
# xfinal = jfinal*2*solver.waist/N
# yfinal = ifinal*2*solver.waist/N
# ts = np.arange(0, np.hypot(xfinal-xinit, yfinal-yinit)/vs[5], 2e-9)
# t, y0 = solver.integrate_short_notransit(vs[0], ts, xinit, yinit, xfinal, yfinal)
# t, y1 = solver.integrate_notransit(vs[0], ts, xinit, yinit, xfinal, yfinal)
# plt.plot(t, y0[:, -2],)
# plt.plot(t, y1[:, -2])
# plt.legend(["Short", "Long"])
# plt.xlabel("Time in s")
# plt.ylabel("$\\rho_{23}$")
# plt.show()

# fig, ax = plt.subplots()
# for i in range(N_real):
#     iinit, jinit, ifinal, jfinal = choose_points()
#     path = bresenham(jinit, iinit, jfinal, ifinal)
#     ax.plot([jinit, jfinal], [iinit, ifinal])
#     ax.scatter([_[0] for _ in path], [_[1] for _ in path])
# plt.title("One realization")
# plt.show()


t0 = time.time()
pols = np.empty((len(vs), N, N), dtype=np.float32)
print("Computing the velocity classes ...")
# computing asynchronously for speed
with progressbar.ProgressBar(max_value=len(vs)*N_real) as bar:
    for counter, v in enumerate(vs):
        grids = []
        plot = False
        
        def thread_function(k):
            np.random.seed(counter*N_real+k+1)
            path = []
            while len(path) < 2:
                iinit, jinit, ifinal, jfinal = choose_points()
                path = bresenham(jinit, iinit, jfinal, ifinal)
            t_path = [(np.hypot(_[1]-iinit, _[0]-jinit)*2*solver.waist/N)/v for _ in path]
            tfinal = np.float32(np.max(t_path))
            t = np.linspace(0, tfinal, N**2, dtype=np.float64)
            xinit = jinit*2*solver.waist/N
            yinit = iinit*2*solver.waist/N
            xfinal = jfinal*2*solver.waist/N
            yfinal = ifinal*2*solver.waist/N
            ynext = np.empty(solver.x0.shape, dtype=np.complex128)
            a, b = solver.integrate_notransit(v, t, xinit, yinit, xfinal, yfinal, ynext)
            # a, b = solver.integrate_notransit_c(v, t, xinit, yinit, xfinal, yfinal)
            grid = np.zeros((N, N), dtype=np.float32)
            for coord in path:
                if coord[0] > N-1:
                    coord[0] = N-1
                if coord[1] > N-1:
                    coord[1] = N-1
                tpath = np.hypot(coord[1]-iinit, coord[0]-jinit)*2*solver.waist/(v*N)
                grid[coord[1], coord[0]] += np.real(b[np.argmin(np.abs(t-tpath)), -2]).astype(np.float32)
            if plot:
                fig, ax = plt.subplots(1, 2)
                ax[0].plot([jinit, jfinal], [iinit, ifinal], color='red', ls='-')
                ax[0].scatter([_[0] for _ in path], [_[1] for _ in path], color='blue')
                ax[0].set_xlim((0, N-1))
                ax[0].set_ylim((0, N-1))
                ax[1].imshow(grid, origin='lower')
                fig.suptitle("Realization")
                plt.show()
            return grid
        # for loop iterative for debugging
        # for k in range(N_real):
        #     grids.append(thread_function(k))
        with Pool(N_proc) as executor:
            for i, _ in enumerate(executor.imap_unordered(thread_function,
                                  range(N_real), N_real//N_proc)):
                bar.update(counter*N_real + i)
                grids.append(_)
        grids = np.asarray(grids)
        pols[counter, :, :] = np.mean(grids, axis=0)
t1 = time.time()-t0
print(f"\nTime elapsed {t1} s")
np.save(f'results/pols_long_{len(vs)}_{N_real}_{time.ctime()}.npy', pols)
# pols = np.load('pols_long_1000.npy')
pols = np.swapaxes(pols, 0, 1)
pols = np.swapaxes(pols, 1, 2)
plt.imshow(np.sum(dv*pols*pv, axis=2), origin='lower')
plt.show()
