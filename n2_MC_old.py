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

T = 150+273  # cell temp
puiss = 1  # power in W
waist = 0.5e-3  # beam waist
detun = -1e-9  # detuning
L = 10e-3  # cell length
solver = temporal_bloch(T, puiss, waist, detun, L)
vs = np.linspace(1, 1000, 40)
dv = vs[1]-vs[0]
pv = ((2/np.pi)**1/2)*((solver.m87/(cst.Boltzmann*solver.T))**(3/2))*vs**2*np.exp(-solver.m87*vs**2/(2*cst.Boltzmann*solver.T))
rs = np.linspace(0, 2*solver.waist, len(vs))
Ts = np.empty((len(vs), len(vs)), dtype=np.float32)
for i in range(Ts.shape[0]):
    for j in range(Ts.shape[1]):
        Ts[i, j] = rs[i]/vs[j]
# plt.imshow(Ts, extent=[np.min(vs), np.max(vs), np.max(rs)/solver.waist, np.min(rs)/solver.waist], aspect='auto')
# plt.xlabel("Speed in m/s")
# plt.ylabel("Position in $w_0$")
# plt.show()
# plt.plot(vs, pv)
# plt.xlabel("Speed in m/s")
# plt.ylabel("Probability")
# plt.show()
t = np.arange(0, np.max(Ts), 5e-9)
# ts = np.sort(Ts.ravel())
# y = np.empty((len(ts), 8, len(vs)), dtype=np.complex64)
def thread_function(k):
    a, b = solver.integrate_short_notransit(vs[k], t)
    return (b, k)
t0 = time.time()
# for i, v in enumerate(vs):
#     sys.stdout.write(f"\rVelocity {i+1}/{len(vs)}")
#     t, y[:, :, i], infodict = solver.integrate_notransit(v, ts)
    # print(infodict['hu'])
# with Pool(14) as executor:
#     y = executor.map(thread_function, range(len(vs)))
y = []
print("Computing the velocity classes ...")
# computing asynchronously for speed
with Pool(14) as executor:
    for i, _ in enumerate(executor.imap_unordered(thread_function,
                                        range(len(vs))), len(vs)//14):
        sys.stdout.write('\rdone {0:%}'.format(i/len(vs)))
        y.append(_)
print("\n")
# reordering the results
indices = [_[1] for _ in y]
y = [_[0] for _ in y]
indices_sorted = list(np.argsort(indices))
y = np.asarray(y)
y = y[indices_sorted, :]
y = np.swapaxes(y, 0, 2)
y = np.swapaxes(y, 0, 1)
# y /= np.nanmax(y)
t1 = time.time()-t0
print(f"\nTime elapsed {t1} s")
# indices = np.linspace(0, len(t)-1, 200, dtype=int)
# plt.plot(t[indices], np.sum(y[indices, 2, :]*pv, axis=1))
plt.plot(t, y[:, 2, 0])
plt.plot(t, y[:, 2, -1])
plt.legend(["Zero velocity", "Max velocity"])
plt.show()
np.save('t_short.npy', t.astype(np.float32))
np.save('y_short.npy', y.astype(np.complex64))
# t = np.load('t_short.npy')
# y = np.load('y_short.npy')
polarizations_r = []
len(Ts[Ts < np.max(t)])
for i in range(Ts.shape[0]):
    pol = []
    for j in range(Ts.shape[1]):
        pol.append(y[np.argmin(np.abs(t-Ts[i, j])), 2, j]*dv*pv[j])
    polarizations_r.append(np.sum(pol))
plt.plot(rs, np.real(polarizations_r))
plt.show()
