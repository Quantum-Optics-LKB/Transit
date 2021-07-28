# -*-coding:utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, hbar, c, mu_0, Boltzmann,  m_n,  e
from odeintw import odeintw
from scipy.integrate import solve_ivp
from numba import jit, cfunc, types, float64, complex128, void
import sys
from bresenham import bresenham
from julia import Main, Julia
jl = Julia(sysimage="sys.so")
from diffeqpy import de


def N(T):
    P_v = 10**(15.88253 - 4529.635/T + 0.00058663*T - 2.99138*np.log10(T))
    return 133.323*P_v/(Boltzmann*T)


def beta_n(T):
    return 2*np.pi*1.03e-13*N(T=T)


def fit(t, a, b, c):
    return a*np.exp(-t/b) + c


@jit(complex128[::1](float64, complex128[::1], complex128[::1], complex128[::1]),
     nopython=True, nogil=True, fastmath=True, cache=True)
def deriv_notransit(t, x, p, dy):
    """Computes the differential term of MBE in the case of a 3 level system
    with one coupling field. The x vector represents the density matrix
    elements rho_11, 22, 21, 12, 31, 13, 32, 23. This function is
    optimized for speed through compilation, pre-allocation and the use
    of fast maths.

    :param np.ndarray x: Density matrix coefficients vector
    :param float t: Time
    p = [v, u0, u1, xinit, yinit, Gamma, Omega13, Omega23,
         gamma21tilde, gamma31tilde, gamma32tilde, waist, r0]
        :param float v: Atom initial velocity
        :param float u0: unit vector for velocity along x
        :param float u1: unit vector for velocity along y
        :param float xinit: initial position x
        :param float yinit: initial position y
    :param np.ndarray dy: vector to store next iteration
    :return: The updated density matrix elements
    :rtype: np.ndarray

    """
    v, u0, u1, xinit, yinit = p[0], p[1], p[2], p[3], p[4]
    Gamma, Omega13, Omega23 = p[5], p[6], p[7]
    gamma21tilde, gamma31tilde, gamma32tilde = p[8], p[9], p[10]
    waist, r0 = p[11], p[12]
    r_sq = (xinit+u0*v*t - r0)*(xinit+u0*v*t - r0) +\
           (yinit+u1*v*t - r0)*(yinit+u1*v*t - r0)
    Om23 = Omega23 * np.exp(-r_sq/(2*waist*waist))
    Om13 = Omega13 * np.exp(-r_sq/(2*waist*waist))
    b = np.array([Gamma/2, Gamma/2, 0, 0, -1j*Om13/2,
                  1j*np.conj(Om13)/2, -1j*Om23/2, 1j*np.conj(Om23)/2],
                 dtype=np.complex128)
    dy[0] = (-Gamma/2)*x[0]-(Gamma/2)*x[1]+(1j*np.conj(Om13)/2)*x[4]-(1j*Om13/2)*x[5]
    dy[1] = (-Gamma/2)*x[0]-(Gamma/2)*x[1]+(1j*np.conj(Om23)/2)*x[6]-(1j*Om23/2)*x[7]
    dy[2] = -gamma21tilde*x[2]+(1j*np.conj(Om23)/2)*x[4]-(1j*Om13/2)*x[7]
    dy[3] = -np.conj(gamma21tilde)*x[3] - (1j*Om23/2)*x[5] + (1j*np.conj(Om13)/2)*x[6]
    dy[4] = 1j*Om13*x[0] + (1j*Om13/2)*x[1] + (1j*Om23/2)*x[2] - gamma31tilde*x[4]
    dy[5] = -1j*np.conj(Om13)*x[0]-1j*(np.conj(Om13)/2)*x[1]-(1j*np.conj(Om23)/2)*x[3]-np.conj(gamma31tilde)*x[5]
    dy[6] = (1j*Om23/2)*x[0]+1j*Om23*x[1]+(1j*Om13/2)*x[3]-gamma32tilde*x[6]
    dy[7] = (-1j*np.conj(Om23)/2)*x[0]-1j*np.conj(Om23)*x[1]-(1j*np.conj(Om13)/2)*x[2]-np.conj(gamma32tilde)*x[7]
    dy += b
    return dy

@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def deriv_notransit_mut(dy, x, p, t):
    """Computes the differential term of MBE in the case of a 3 level system
    with one coupling field. The x vector represents the density matrix
    elements rho_11, 22, 21, 12, 31, 13, 32, 23. This function is
    optimized for speed through compilation, pre-allocation and the use
    of fast maths.

    :param np.ndarray x: Density matrix coefficients vector
    :param float t: Time
    p = [v, u0, u1, xinit, yinit, Gamma, Omega13, Omega23,
         gamma21tilde, gamma31tilde, gamma32tilde, waist, r0]
        :param float v: Atom initial velocity
        :param float u0: unit vector for velocity along x
        :param float u1: unit vector for velocity along y
        :param float xinit: initial position x
        :param float yinit: initial position y
    :param np.ndarray dy: vector to store next iteration
    :return: The updated density matrix elements
    :rtype: np.ndarray

    """
    v, u0, u1, xinit, yinit = p[0], p[1], p[2], p[3], p[4]
    Gamma, Omega13, Omega23 = p[5], p[6], p[7]
    gamma21tilde, gamma31tilde, gamma32tilde = p[8], p[9], p[10]
    waist, r0 = p[11], p[12]
    r_sq = (xinit+u0*v*t - r0)*(xinit+u0*v*t - r0) +\
           (yinit+u1*v*t - r0)*(yinit+u1*v*t - r0)
    Om23 = Omega23 * np.exp(-r_sq/(2*waist*waist))
    Om13 = Omega13 * np.exp(-r_sq/(2*waist*waist))
    b = np.array([Gamma/2, Gamma/2, 0, 0, -1j*Om13/2,
                  1j*np.conj(Om13)/2, -1j*Om23/2, 1j*np.conj(Om23)/2],
                 dtype=np.complex128)
    dy[0] = (-Gamma/2)*x[0]-(Gamma/2)*x[1]+(1j*np.conj(Om13)/2)*x[4]-(1j*Om13/2)*x[5]
    dy[1] = (-Gamma/2)*x[0]-(Gamma/2)*x[1]+(1j*np.conj(Om23)/2)*x[6]-(1j*Om23/2)*x[7]
    dy[2] = -gamma21tilde*x[2]+(1j*np.conj(Om23)/2)*x[4]-(1j*Om13/2)*x[7]
    dy[3] = -np.conj(gamma21tilde)*x[3] - (1j*Om23/2)*x[5] + (1j*np.conj(Om13)/2)*x[6]
    dy[4] = 1j*Om13*x[0] + (1j*Om13/2)*x[1] + (1j*Om23/2)*x[2] - gamma31tilde*x[4]
    dy[5] = -1j*np.conj(Om13)*x[0]-1j*(np.conj(Om13)/2)*x[1]-(1j*np.conj(Om23)/2)*x[3]-np.conj(gamma31tilde)*x[5]
    dy[6] = (1j*Om23/2)*x[0]+1j*Om23*x[1]+(1j*Om13/2)*x[3]-gamma32tilde*x[6]
    dy[7] = (-1j*np.conj(Om23)/2)*x[0]-1j*np.conj(Om23)*x[1]-(1j*np.conj(Om13)/2)*x[2]-np.conj(gamma32tilde)*x[7]
    dy += b


deriv_notransit_jul = Main.eval("""
function f(dy, x, p, t)
    v, u0, u1, xinit, yinit = p[1], p[2], p[3], p[4], p[5]
    Gamma, Omega13, Omega23 = p[6], p[7], p[8]
    gamma21tilde, gamma31tilde, gamma32tilde = p[9], p[10], p[11]
    waist, r0 = p[12], p[13]
    r_sq = (xinit+u0*v*t - r0)*(xinit+u0*v*t - r0) +\
           (yinit+u1*v*t - r0)*(yinit+u1*v*t - r0)
    Om23 = Omega23 * exp(-r_sq/(2*waist*waist))
    Om13 = Omega13 * exp(-r_sq/(2*waist*waist))
    dy[1] = (-Gamma/2)*x[1]-(Gamma/2)*x[2]+(im*conj(Om13)/2)*x[5]-(im*Om13/2)*x[6]+Gamma/2
    dy[2] = (-Gamma/2)*x[1]-(Gamma/2)*x[2]+(im*conj(Om23)/2)*x[7]-(im*Om23/2)*x[8]+Gamma/2
    dy[3] = -gamma21tilde*x[3]+(im*conj(Om23)/2)*x[5]-(im*Om13/2)*x[8]
    dy[4] = -conj(gamma21tilde)*x[4] - (im*Om23/2)*x[6] + (im*conj(Om13)/2)*x[7]
    dy[5] = im*Om13*x[1] + (im*Om13/2)*x[2] + (im*Om23/2)*x[3] - gamma31tilde*x[5]-im*Om13/2
    dy[6] = -im*conj(Om13)*x[1]-im*(conj(Om13)/2)*x[2]-(im*conj(Om23)/2)*x[4]-conj(gamma31tilde)*x[6]+im*conj(Om13)/2
    dy[7] = (im*Om23/2)*x[1]+im*Om23*x[2]+(im*Om13/2)*x[4]-gamma32tilde*x[7] - im*Om23/2
    dy[8] = (-im*conj(Om23)/2)*x[1]-im*conj(Om23)*x[2]-(im*conj(Om13)/2)*x[3]-conj(gamma32tilde)*x[8]+im*conj(Om23)/2
end""")

@jit(complex128[:, ::1](float64, complex128[::1], complex128[::1], complex128[::1]),
     nopython=True, nogil=True, fastmath=True, cache=True)
def deriv_notransit_jac(t, x, p, dy):
    """Computes the differential term of MBE in the case of a 3 level system
    with one coupling field. The x vector represents the density matrix
    elements rho_11, 22, 21, 12, 31, 13, 32, 23. This function is
    optimized for speed through compilation, pre-allocation and the use
    of fast maths.

    :param np.ndarray x: Density matrix coefficients vector
    :param float t: Time
    :param float v: Atom initial velocity
    :param float u0: unit vector for velocity along x
    :param float u1: unit vector for velocity along y
    :param float xinit: initial position x
    :param float yinit: initial position y5000
    :param np.ndarray dy: vector to store next iteration
    :return: The updated density matrix elements
    :rtype: np.ndarray

    """
    v, u0, u1, xinit, yinit = p[0], p[1], p[2], p[3], p[4]
    Gamma, Omega13, Omega23 = p[5], p[6], p[7]
    gamma21tilde, gamma31tilde, gamma32tilde = p[8], p[9], p[10]
    waist, r0 = p[11], p[12]
    r_sq = (xinit+u0*v*t - r0)*(xinit+u0*v*t - r0) +\
           (yinit+u1*v*t - r0)*(yinit+u1*v*t - r0)
    Om23 = Omega23 * np.exp(-r_sq/(2*waist*waist))
    Om13 = Omega13 * np.exp(-r_sq/(2*waist*waist))
    A = np.array([[-Gamma/2, -Gamma/2, 0, 0, 1j*np.conj(Om13)/2, -1j*Om13/2, 0, 0],
         [-Gamma/2, -Gamma/2, 0, 0, 0, 0, 1j*np.conj(Om23)/2, -1j*Om23/2],
         [0, 0, -gamma21tilde, 0, 1j*np.conj(Om23)/2, 0, 0, -1j*Om13/2],
         [0, 0, 0, -np.conj(gamma21tilde), 0, -1j*Om23/2, 1j*np.conj(Om13)/2, 0],
         [1j*Om13, 1j*Om13/2, 1j*Om23/2, 0, -gamma31tilde, 0, 0, 0],
         [-1j*np.conj(Om13), -1j*np.conj(Om13)/2, 0, -1j*np.conj(Om23)/2, 0, -np.conj(gamma31tilde), 0, 0],
         [1j*Om23/2, 1j*Om23, 0, 1j*Om13/2, 0, 0, -gamma32tilde, 0],
         [-1j*np.conj(Om23)/2, -1j*np.conj(Om23), -1j*np.conj(Om13)/2, 0, 0, 0, 0, -np.conj(gamma32tilde)]],
         dtype=np.complex128)
    return A


class temporal_bloch:

    def __init__(self, T, puiss, waist, detun, L, N_grid=128, N_v=20,
                 N_real=20, N_proc=15):
        # grid params for MC
        self.N_grid = N_grid
        self.N_v = N_v
        self.N_real = 20

        self.T = T
        self.L = L
        self.waist = waist
        self.window = 2.5*self.waist
        self.r0 = self.window/2
        self.detun = 2*np.pi*detun
        self.puiss = puiss
        self.frac = 0.995

        self.wl = 780.241e-9
        self.k = 2*np.pi/self.wl
        self.Gamma = 2*np.pi * 6.065e6
        self.m87 = 1.44316060e-25
        self.m85 = 1.44316060e-25 - 2*m_n
        self.u = np.sqrt(2*Boltzmann*T/(self.frac*self.m87+(1-self.frac)*self.m85))
        self.d = np.sqrt(9*epsilon_0*hbar*self.Gamma* self.wl**3/(8*np.pi**2))
        self.gamma_t = self.u/self.waist *2/np.sqrt(np.pi) #formule à compléter!
        self.gamma = self.Gamma/2 + self.gamma_t + beta_n(self.T)/2
        self.delta0 = 2*np.pi * 6.834e9
        self.gamma32tilde = self.gamma - 1j*self.detun
        self.gamma31tilde = self.gamma - 1j*(self.detun-self.delta0)
        self.gamma21tilde = self.gamma_t + 1j*self.delta0

        self.I = 2*self.puiss/(np.pi*self.waist**2)
        self.E = np.sqrt(2*self.I/(c*epsilon_0))


        self.mu23 = (1/np.sqrt(5))*np.sqrt(1/18 + 5/18 + 7/9)*self.d + 1j*0
        self.mu13 = (1/np.sqrt(3))*np.sqrt(1/9 + 5/18 + 5/18)*self.d + 1j*0
        self.G1 = 3/8
        self.G2 = 5/8
        self.Omega23 = self.E*self.mu23/hbar
        self.Omega13 = self.E*self.mu13/hbar

        self.x0 = np.array([self.G1 + 1j*0, self.G2 + 1j*0, 0, 0, 0, 0, 0, 0],
                           dtype=np.complex128)
        self.x0_short = np.array([self.G1 + 1j*0, self.G2 + 1j*0, 0, 0],
                                 dtype=np.complex128)

        print(f"Omega23 = {np.real(self.Omega23)*1e-9/(2*np.pi)} GHz")

    def choose_points(self, plot=False):
        edges = []
        for i in range(self.N_grid):
            edges.append((0, i))
            edges.append((self.N_grid-1, i))
            edges.append((i, 0))
            edges.append((i, self.N_grid-1))
        iinit, jinit = edges[np.random.randint(0, len(edges))]
        ifinal, jfinal = iinit, jinit
        cdtn = (ifinal == iinit) or (jfinal == jinit)
        while cdtn:
            ifinal, jfinal = edges[np.random.randint(0, len(edges))]
            cdtn = (ifinal == iinit) or (jfinal == jinit)
        if plot:
            fig, ax = plt.subplots()
            ax.plot([jinit, jfinal], [iinit, ifinal], ls='--', marker='o',
                    scalex=False, scaley=False)
            ax.set_xlim((0, self.N_grid-1))
            ax.set_ylim((0, self.N_grid-1))
            plt.show()
        return iinit, jinit, ifinal, jfinal

    def draw_vz(self, v):
        vz = np.abs(2*v)
        while np.abs(vz) > np.abs(v):
            vz = np.random.normal(0, np.sqrt(Boltzmann*self.T/self.m87))
        return vz

    def integrate_notransit(self, vz, v_perp, iinit, jinit, ifinal, jfinal, ynext):
        path = bresenham(jinit, iinit, jfinal, ifinal)
        xinit = jinit*self.window/self.N_grid
        yinit = iinit*self.window/self.N_grid
        xfinal = jfinal*self.window/self.N_grid
        yfinal = ifinal*self.window/self.N_grid
        # velocity unit vector
        if v_perp != 0:
            u0 = xfinal-xinit
            u1 = yfinal-yinit
            norm = np.hypot(u0, u1)
            u0 /= norm
            u1 /= norm
            t_path = np.array([np.hypot(_[1]-iinit, _[0]-jinit)*self.window/(self.N_grid*v_perp) for _ in path])
        else:
            u0 = u1 = 0
            t_path = np.array([np.hypot(_[1]-iinit, _[0]-jinit)*self.window/(self.N_grid*np.abs(vz)) for _ in path])
        tfinal = t_path[-1]
        # print(f'tfinal = {tfinal*1e6} us')
        # ts = np.arange(0, tfinal, 1e-10, dtype=np.float64)
        ts = np.linspace(0, tfinal, 1000)
        p = np.array([v_perp, u0, u1, xinit, yinit, self.Gamma, self.Omega13,
                      self.Omega23, self.gamma21tilde,
                      self.gamma31tilde - 1j*self.k*vz,
                      self.gamma32tilde - 1j*self.k*vz, self.waist, self.r0],
                     dtype=np.complex128)
        # ys, infodict = odeintw(deriv_notransit, self.x0, ts,
        #                        args=(p, ynext),
        #                        Dfun=deriv_notransit_jac,
        #                        full_output=True, hmax=1e-2, hmin=1e-36,
        #                        h0=1e-12, tfirst=True)
        # return ts, ys, path
        tspan = (0, tfinal)
        prob = de.ODEProblem(deriv_notransit_jul, self.x0, tspan, p,
                             maxiters=1e8)
        sol = de.solve(prob, de.BS3(), saveat=ts)
        return np.array(sol.t, dtype=np.float64), \
            np.array(sol.u, dtype=np.complex128), path


    def integrate_short_notransit(self, v, ts, xinit, yinit, xfinal, yfinal):
        # y, infodict = odeintw(self.deriv_notransit, self.x0, ts, args=(v,),
        #             full_output=True, hmax=1e-6, hmin=1e-14, h0=1e-14)
        # return ts, y, infodict
        Gamma = self.Gamma
        gamma32tilde = self.gamma32tilde + self.k*v
        Omega23 = self.Omega23
        waist = self.waist
        r0 = self.r0
        # velocity unit vector
        u0 = xfinal-xinit
        u1 = yfinal-yinit
        norm = np.hypot(u0, u1)
        u0 /= norm
        u1 /= norm

        @jit(nopython=True)
        def deriv_short_notransit(x, t, v):
            r = np.hypot(xinit+u0*v*t - r0, yinit+u1*v*t - r0)
            Om = Omega23 * np.exp(-r**2/(2*waist**2))
            A = np.array(
                [[-Gamma/2, -Gamma/2, 0, 0],
                 [-Gamma/2, -Gamma/2, 1j/2*Om, -1j/2*Om],
                 [1j/2*Om, 1j*Om, -1*gamma32tilde, 0],
                 [-1j/2*Om, -1j*Om, 0, -1*np.conj(gamma32tilde)]])

            b = np.array([Gamma/2, Gamma/2, -1j/2*Om, 1j/2*Om])

            return np.matmul(A, x) + b*np.heaviside(t, 1)
        y = odeintw(self.deriv_short_notransit, self.x0_short, ts, args=(v,),
                    full_output=False, hmax=1e-4, hmin=2e-14, h0=2e-14)
        return ts, y
