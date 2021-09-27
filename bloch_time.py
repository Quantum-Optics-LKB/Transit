# -*-coding:utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, hbar, c, mu_0, Boltzmann,  m_n,  e
from odeintw import odeintw
from scipy.integrate import solve_ivp
from numba import jit, cfunc, types, float64, complex128, void
import sys
from julia import Main

@jit(nopython=True)
def bresenham(x1, y1, x2, y2):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else [x, y]
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

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
        self.N_real = N_real
        self.T = T
        self.L = L
        self.waist = waist
        self.window = 4*self.waist
        self.r0 = self.window/2
        self.detun = 2*np.pi*detun
        self.puiss = puiss
        self.I = 2*self.puiss/(np.pi*self.waist**2)
        self.E = np.sqrt(2*self.I/(c*epsilon_0))
        self.frac = 0.995
        self.wl = 780.241e-9
        self.k = 2*np.pi/self.wl
        self.Gamma = 2*np.pi * 6.065e6
        self.m87 = 1.44316060e-25
        self.m85 = 1.44316060e-25 - 2*m_n
        self.u = np.sqrt(2*Boltzmann*T/(self.frac*self.m87+(1-self.frac)*self.m85))
        self.d = np.sqrt(9*epsilon_0*hbar*self.Gamma* self.wl**3/(8*np.pi**2))
        self.gamma_t_analytical = self.u/self.waist *2/np.sqrt(np.pi) #formule à compléter!
        self.gamma_t = 0.0
        self.gamma = self.Gamma/2 + self.gamma_t + beta_n(self.T)/2
        self.delta0 = 2*np.pi * 6.834e9
        self.gamma32tilde = self.gamma - 1j*self.detun
        self.gamma31tilde = self.gamma - 1j*(self.detun-self.delta0)
        self.gamma21tilde = self.gamma_t + 1j*self.delta0
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
        # deriv_notransit_jul = Main.eval("""
        # function f(dy, x, p, t)
        #     v, u0, u1, xinit, yinit = p[1], p[2], p[3], p[4], p[5]
        #     Gamma, Omega13, Omega23 = p[6], p[7], p[8]
        #     gamma21tilde, gamma31tilde, gamma32tilde = p[9], p[10], p[11]
        #     waist, r0 = p[12], p[13]
        #     r_sq = (xinit+u0*v*t - r0)*(xinit+u0*v*t - r0) +\
        #            (yinit+u1*v*t - r0)*(yinit+u1*v*t - r0)
        #     Om23 = Omega23 * exp(-r_sq/(2*waist*waist))
        #     Om13 = Omega13 * exp(-r_sq/(2*waist*waist))
        #     dy[1] = (-Gamma/2)*x[1]-(Gamma/2)*x[2]+(im*conj(Om13)/2)*x[5]-(im*Om13/2)*x[6]+Gamma/2
        #     dy[2] = (-Gamma/2)*x[1]-(Gamma/2)*x[2]+(im*conj(Om23)/2)*x[7]-(im*Om23/2)*x[8]+Gamma/2
        #     dy[3] = -gamma21tilde*x[3]+(im*conj(Om23)/2)*x[5]-(im*Om13/2)*x[8]
        #     dy[4] = -conj(gamma21tilde)*x[4] - (im*Om23/2)*x[6] + (im*conj(Om13)/2)*x[7]
        #     dy[5] = im*Om13*x[1] + (im*Om13/2)*x[2] + (im*Om23/2)*x[3] - gamma31tilde*x[5]-im*Om13/2
        #     dy[6] = -im*conj(Om13)*x[1]-im*(conj(Om13)/2)*x[2]-(im*conj(Om23)/2)*x[4]-conj(gamma31tilde)*x[6]+im*conj(Om13)/2
        #     dy[7] = (im*Om23/2)*x[1]+im*Om23*x[2]+(im*Om13/2)*x[4]-gamma32tilde*x[7] - im*Om23/2
        #     dy[8] = (-im*conj(Om23)/2)*x[1]-im*conj(Om23)*x[2]-(im*conj(Om13)/2)*x[3]-conj(gamma32tilde)*x[8]+im*conj(Om23)/2
        # end""")
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
        ys, infodict = odeintw(deriv_notransit, self.x0, ts,
                               args=(p, ynext),
                               Dfun=deriv_notransit_jac,
                               full_output=True, hmax=1e-2, hmin=1e-36,
                               h0=1e-12, tfirst=True)
        return ts, ys, path
        # tspan = (0, tfinal)
        # prob = de.ODEProblem(deriv_notransit_jul, self.x0, tspan, p,
        #                      maxiters=1e8)
        # sol = de.solve(prob, de.BS3(), saveat=ts)
        # return np.array(sol.t, dtype=np.float64), \
        #     np.array(sol.u, dtype=np.complex128), path

    def do_N_real(self, v: float):
        Main.eval(f"const N_real = {self.N_real}")
        Main.eval(f"const N_grid = {self.N_grid}")
        Main.eval(f"const T = {self.T}")
        Main.eval(f"const window = {self.window}")
        Main.eval(f"const Gamma = {np.real(self.Gamma)}+{np.imag(self.Gamma)}*im")
        Main.eval(f"const Omega13 = {np.real(self.Omega13)}+{np.imag(self.Omega13)}*im")
        Main.eval(f"const Omega23 = {np.real(self.Omega23)}+{np.imag(self.Omega23)}*im")
        Main.eval(f"const gamma21tilde = {np.real(self.gamma21tilde)}+{np.imag(self.gamma21tilde)}*im")
        Main.eval(f"const gamma31tilde = {np.real(self.gamma31tilde)}+{np.imag(self.gamma31tilde)}*im")
        Main.eval(f"const gamma32tilde = {np.real(self.gamma32tilde)}+{np.imag(self.gamma32tilde)}*im")
        Main.eval(f"const waist = {self.waist} + 0*im")
        Main.eval(f"const r0 = {self.r0} + 0*im")
        Main.eval(f"const x0 = [{self.G1} + 0*im, {self.G2} + im*0, 0.0*im, 0.0*im, 0.0*im, 0.0*im, 0.0*im, 0.0*im]")
        Main.eval(f"const k = {self.k}")
        Main.eval(f"const v = {v}")
        grid, counter_grid = Main.eval("""
        using DifferentialEquations
        using PyPlot

        # global variables to be set from python side through Julia "Main" namespace
        const m87 = 1.44316060e-25
        const k_B = 1.38064852e-23
        # function for line generation
        function bresenham(x1::Int32, y1::Int32, x2::Int32, y2::Int32)

            dx = x2 - x1
            dy = y2 - y1

            # Determine how steep the line is
            is_steep = abs(dy) > abs(dx)

            # Rotate line
            if is_steep
                x1, y1 = y1, x1
                x2, y2 = y2, x2
            end
            # Swap start and end points if necessary and store swap state
            swapped = false
            if x1 > x2
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                swapped = true
            end
            # Recalculate differentials
            dx = x2 - x1
            dy = y2 - y1

            # Calculate error
            error = dx / 2
            if y1 < y2
                ystep = 1
            else
                ystep = -1
            end

            # Iterate over bounding box generating points between start and end
            y = y1
            points = []
            for x in x1:x2+1
                if is_steep
                    coord = [y, x]
                else
                    coord = [x, y]
                end
                append!(points, [coord])
                error -= abs(dy)
                if error < 0
                    y += ystep
                    error += dx
                end
            end
            # Reverse the list if the coordinates were swapped
            if swapped
                reverse!(points)
            end
            return points
        end

        function choose_points()
            edges = Array{Tuple{Int32, Int32}, 1}(undef, 4*N_grid)
            for i in 1:N_grid
                edges[i] = (1, i)
                edges[i+N_grid] = (N_grid, i)
                edges[i+2*N_grid] = (i, 1)
                edges[i+3*N_grid] = (i, N_grid)
            end
            iinit, jinit = edges[rand(1:length(edges), 1)][1]
            ifinal, jfinal = iinit, jinit
            cdtn = (ifinal == iinit) || (jfinal == jinit)
            while cdtn
                ifinal, jfinal = edges[rand(1:length(edges), 1)][1]
                cdtn = (ifinal == iinit) || (jfinal == jinit)
            end
            return (iinit, jinit, ifinal, jfinal)
        end

        function draw_vz(v::Float64)::Float64
            vz = abs(2*v)
            while abs(vz) > abs(v)
                vz = randn()*sqrt(k_B*T/m87)
            end
            return vz
        end

        function f!(dy, x, p, t)
            v, u0, u1, xinit, yinit = p[1], p[2], p[3], p[4], p[5]
            Gamma, Omega13, Omega23 = p[6], p[7], p[8]
            gamma21tilde, gamma31tilde, gamma32tilde = p[9], p[10], p[11]
            waist, r0 = p[12], p[13]
            r_sq = (xinit+u0*v*t - r0)^2 + (yinit+u1*v*t - r0)^2
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
        end

        paths = [[] for i=1:N_real]
        xs = [[] for i=1:N_real]
        ts = [[] for i=1:N_real]
        v_perps = zeros(Float64, N_real)
        function prob_func(prob, i, repeat)
            # change seed of random number generation as the solver uses multithreading
            iinit, jinit, ifinal, jfinal = choose_points()
            vz = draw_vz(v)
            v_perp = sqrt(v^2 - vz^2)
            xinit = jinit*window/N_grid
            yinit = iinit*window/N_grid
            xfinal = jfinal*window/N_grid
            yfinal = ifinal*window/N_grid
            # velocity unit vector
            if v_perp != 0
                u0 = xfinal-xinit
                u1 = yfinal-yinit
                norm = hypot(u0, u1)
                u0 /= norm
                u1 /= norm
                new_tfinal = hypot((xfinal-xinit), (yfinal-yinit))/v_perp
            else
                u0 = u1 = 0
                new_tfinal = hypot((xfinal-xinit), (yfinal-yinit))/abs(vz)
            end
            new_p = [v_perp + 0*im, u0 + 0*im, u1 + 0*im, xinit + 0*im, yinit + 0*im,
                     Gamma, Omega13, Omega23, gamma21tilde, gamma31tilde - im*k*vz,
                     gamma32tilde - im*k*vz, waist, r0]
            new_tspan = (0.0, new_tfinal)
            tsave = collect(LinRange(0.0, new_tfinal, 1000))
            remake(prob, p=new_p, tspan=new_tspan, saveat=tsave)
        end
        # instantiate a problem
        p = [1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im, 1.0 + 0*im,
             Gamma, Omega13,
             Omega23, gamma21tilde,
             gamma31tilde - im*k*0.0,
             gamma32tilde - im*k*0.0, waist, r0]
        tspan = (0.0, 1.0)
        tsave = collect(LinRange(0.0, 1.0, 1000))
        prob = ODEProblem{true}(f!, x0, tspan, p, saveat=tsave)
        ensembleprob = EnsembleProblem(prob, prob_func=prob_func)
        sol = solve(ensembleprob, rodas4(autodiff=false), EnsembleThreads(), trajectories=N_real, maxiters=Int(1e8))
        for i in 1:N_real
            xs[i] = sol[i].u
            ts[i] = sol[i].t
            v_perps[i] = sol[i].prob.p[1]
            local u0 = sol[i].prob.p[2]
            local u1 = sol[i].prob.p[3]
            local xinit = sol[i].prob.p[4]
            local yinit = sol[i].prob.p[5]
            local tfinal = sol[i].prob.tspan[2]
            local xfinal = xinit+u0*v_perps[i]*tfinal
            local yfinal = yinit+u1*v_perps[i]*tfinal
            local iinit = Int32(round(real(yinit)*(N_grid/window), digits=1))
            local jinit = Int32(round(real(xinit)*(N_grid/window), digits=1))
            local ifinal = Int32(round(real(yfinal)*(N_grid/window), digits=1))
            local jfinal = Int32(round(real(xfinal)*(N_grid/window), digits=1))
            local path = bresenham(jinit, iinit, jfinal, ifinal)
            paths[i] = path
        end
        # Reformat xs as a proper multidimensional Array for easier indexing
        Xs = zeros(ComplexF64, (N_real, length(ts[1]), 8))
        for i = 1:N_real
            for j = 1:length(ts[i])
                for k in 1:8
                    Xs[i, j , k] = xs[i][j][k]
                end
            end
        end
        xs = nothing
        # define empty grid to accumulate the trajectories of the atoms and count them
        grid = zeros(ComplexF64, (N_grid, N_grid))
        counter_grid = zeros(Int32, (N_grid, N_grid))
        for i = 1:N_real
            local iinit = paths[i][1][1]
            local jinit = paths[i][1][2]
            for coord in paths[i]
                if coord[1] > N_grid-1
                    coord[1] = N_grid-1
                end
                if coord[2] > N_grid-1
                    coord[2] = N_grid-1
                end
                if coord[1] < 1
                    coord[1] = 1
                end
                if coord[2] < 1
                    coord[2] = 1
                end
                tpath = hypot(coord[2]-iinit, coord[1]-jinit)*window/(v_perps[i]*N_grid)
                grid[coord[2], coord[1]] += real(Xs[i, argmin(abs.([ts[i][k]-tpath for k=1:length(ts[i])])), 7])
                counter_grid[coord[2], coord[1]] += 1
            end
        end
        [grid, counter_grid]
        """)
        return grid, counter_grid

    def chi_analytical(self):
        a = (self.Gamma+self.gamma_t_analytical)/2*self.gamma
        b = self.gamma_t_analytical/self.gamma
        Es = np.sqrt((2*b*(1+a))/(1+b))*hbar*self.gamma/self.mu23
        pref = self.G2*N(self.T)/(epsilon_0*hbar) * abs(self.mu23)**2/self.gamma
        chi = pref * (1j-self.detun/self.gamma)/(1+(self.detun/self.gamma)**2 + (self.E/Es)**2)
        return chi

    def do_V_span(self, v0: float, v1: float, N_v: int):
        Main.eval(f"global N_v = {N_v}")
        Main.eval(f"global v0 = {v0}")
        Main.eval(f"global v1 = {v1}")
        Main.eval(f"global N_real = {self.N_real}")
        Main.eval(f"global N_grid = {self.N_grid}")
        Main.eval(f"global T = {self.T}")
        Main.eval(f"global window = {self.window}")
        Main.eval(f"global Gamma = {np.real(self.Gamma)}+{np.imag(self.Gamma)}*im")
        Main.eval(f"global Omega13 = {np.real(self.Omega13)}+{np.imag(self.Omega13)}*im")
        Main.eval(f"global Omega23 = {np.real(self.Omega23)}+{np.imag(self.Omega23)}*im")
        Main.eval(f"global gamma21tilde = {np.real(self.gamma21tilde)}+{np.imag(self.gamma21tilde)}*im")
        Main.eval(f"global gamma31tilde = {np.real(self.gamma31tilde)}+{np.imag(self.gamma31tilde)}*im")
        Main.eval(f"global gamma32tilde = {np.real(self.gamma32tilde)}+{np.imag(self.gamma32tilde)}*im")
        Main.eval(f"global waist = {self.waist} + 0*im")
        Main.eval(f"global r0 = {self.r0} + 0*im")
        Main.eval(f"global x0 = [{self.G1} + 0*im, {self.G2} + im*0, 0.0*im, 0.0*im, 0.0*im, 0.0*im, 0.0*im, 0.0*im]")
        Main.eval(f"global k = {self.k}")
        grid_weighted, counter_grid = Main.eval("""
        using OrdinaryDiffEq
        using ProgressBars

        const m87 = 1.44316060e-25
        const k_B = 1.38064852e-23
        const N_t = 2000


        function bresenham(x1::Int32, y1::Int32, x2::Int32, y2::Int32)::Array{Array{Int32, 1}, 1}

            dx = x2 - x1
            dy = y2 - y1

            # Determine how steep the line is
            is_steep = abs(dy) > abs(dx)

            # Rotate line
            if is_steep
                x1, y1 = y1, x1
                x2, y2 = y2, x2
            end
            # Swap start and end points if necessary and store swap state
            swapped = false
            if x1 > x2
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                swapped = true
            end
            # Recalculate differentials
            dx = x2 - x1
            dy = y2 - y1

            # Calculate error
            error = dx / 2
            if y1 < y2
                ystep = 1
            else
                ystep = -1
            end

            # Iterate over bounding box generating points between start and end
            y = y1
            points = []
            for x in x1:x2
                if is_steep
                    coord = [y, x]
                else
                    coord = [x, y]
                end
                append!(points, [coord])
                error -= abs(dy)
                if error < 0
                    y += ystep
                    error += dx
                end
            end
            # Reverse the list if the coordinates were swapped
            if swapped
                reverse!(points)
            end
            return points
        end

        function choose_points()
            edges = Array{Tuple{Int32, Int32}, 1}(undef, 4*N_grid)
            for i in 1:N_grid
                edges[i] = (1, i)
                edges[i+N_grid] = (N_grid, i)
                edges[i+2*N_grid] = (i, 1)
                edges[i+3*N_grid] = (i, N_grid)
            end
            iinit, jinit = edges[rand(1:length(edges), 1)][1]
            ifinal, jfinal = iinit, jinit
            cdtn = (ifinal == iinit) || (jfinal == jinit)
            while cdtn
                ifinal, jfinal = edges[rand(1:length(edges), 1)][1]
                cdtn = (ifinal == iinit) || (jfinal == jinit)
            end
            return (iinit, jinit, ifinal, jfinal)
        end

        function draw_vz(v::Float64)::Float64
            vz = abs(2*v)
            while abs(vz) > abs(v)
                vz = randn()*sqrt(k_B*T/m87)
            end
            return vz
        end

        @inbounds begin
        @fastmath begin
        function f!(dy::Array{ComplexF64, 1}, x::Array{ComplexF64, 1},
                    p::Array{ComplexF64, 1}, t::Float64)
            r_sq = (p[4]+p[2]*p[1]*t - p[13])*(p[4]+p[2]*p[1]*t - p[13]) + (p[5]+p[3]*p[1]*t - p[13])*(p[5]+p[3]*p[1]*t - p[13])
            Om23 = p[8] * exp(-r_sq/(2.0*p[12]*p[12]))
            Om13 = p[7] * exp(-r_sq/(2.0*p[12]*p[12]))
            dy[1] = (-p[6]/2.0)*x[1]-(p[6]/2.0)*x[2]+(im*conj(Om13)/2.0)*x[5]-(im*Om13/2)*x[6]+p[6]/2.0
            dy[2] = (-p[6]/2.0)*x[1]-(p[6]/2.0)*x[2]+(im*conj(Om23)/2.0)*x[7]-(im*Om23/2)*x[8]+p[6]/2.0
            dy[3] = -p[9]*x[3]+(im*conj(Om23)/2.0)*x[5]-(im*Om13/2.0)*x[8]
            dy[4] = -conj(p[9])*x[4] - (im*Om23/2.0)*x[6] + (im*conj(Om13)/2.0)*x[7]
            dy[5] = im*Om13*x[1] + (im*Om13/2.0)*x[2] + (im*Om23/2.0)*x[3] - p[10]*x[5]-im*Om13/2.0
            dy[6] = -im*conj(Om13)*x[1]-im*(conj(Om13)/2.0)*x[2]-(im*conj(Om23)/2.0)*x[4]-conj(p[10])*x[6]+im*conj(Om13)/2.0
            dy[7] = (im*Om23/2.0)*x[1]+im*Om23*x[2]+(im*Om13/2.0)*x[4]-p[11]*x[7] - im*Om23/2.0
            dy[8] = (-im*conj(Om23)/2.0)*x[1]-im*conj(Om23)*x[2]-(im*conj(Om13)/2.0)*x[3]-conj(p[11])*x[8]+im*conj(Om23)/2.0
        end
        end
        end


        @inbounds begin
        @fastmath begin
        function f_jac!(J::Array{ComplexF64, 2}, x::Array{ComplexF64, 1},
                    p::Array{ComplexF64, 1}, t::Float64)
            r_sq = (p[4]+p[2]*p[1]*t - p[13])^2.0 + (p[5]+p[3]*p[1]*t - p[13])^2.0
            Om23 = p[8] * exp(-r_sq/(2*p[12]*p[12]))
            Om13 = p[7] * exp(-r_sq/(2*p[12]*p[12]))
            J[1, 1] = (-p[6]/2.0)
            J[1, 2] = -(p[6]/2.0)
            J[1, 3] = 0.0 * im
            J[1, 4] = 0.0 * im
            J[1, 5] = (im*conj(Om13)/2.0)
            J[1, 6] = -(im*Om13/2.0)
            J[1, 7] = 0.0 * im
            J[1, 8] = 0.0 * im
            J[2, 1] = (-p[6]/2.0)
            J[2, 2] = -(p[6]/2.0)
            J[2, 3] = 0.0 * im
            J[2, 4] = 0.0 * im
            J[2, 5] = 0.0 * im
            J[2, 6] = 0.0 * im
            J[2, 7] = (im*conj(Om23)/2.0)
            J[2, 8] = -(im*Om23/2.0)
            J[3 ,1] = 0.0 * im
            J[3 ,2] = 0.0 * im
            J[3, 3] = -p[9]
            J[3 ,4] = 0.0 * im
            J[3, 5] = (im*conj(Om23)/2.0)
            J[3 ,6] = 0.0 * im
            J[3 ,7] = 0.0 * im
            J[3, 8] = -(im*Om13/2.0)
            J[4, 1] = 0.0 * im
            J[4, 2] = 0.0 * im
            J[4, 3] = 0.0 * im
            J[4, 4] = -conj(p[9])
            J[4, 5] = 0.0 * im
            J[4, 6] = -(im*Om23/2.0)
            J[4, 7] = (im*conj(Om13)/2.0)
            J[4, 8] = 0.0 * im
            J[5, 1] = im*Om13
            J[5, 2] = (im*Om13/2.0)
            J[5, 3] = (im*Om23/2.0)
            J[5, 4] = 0.0 * im
            J[5, 5] = - p[10]
            J[5, 6] = 0.0 * im
            J[5, 7] = 0.0 * im
            J[5, 8] = 0.0 * im
            J[6, 1] = -im*conj(Om13)
            J[6, 2] = -im*(conj(Om13)/2.0)
            J[6, 3] = 0.0 * im
            J[6, 4] = -(im*conj(Om23)/2.0)
            J[6, 5] = 0.0 * im
            J[6, 6] = -conj(p[10])
            J[6, 7] = 0.0 * im
            J[6, 8] = 0.0 * im
            J[7, 1] = (im*Om23/2.0)
            J[7, 2] = +im*Om23*x[2]
            J[7, 3] = 0.0 * im
            J[7, 4] = +(im*Om13/2.0)
            J[7, 5] = 0.0 * im
            J[7, 6] = 0.0 * im
            J[7, 7] = -p[11]
            J[7, 8] = 0.0 * im
            J[8, 1] = (-im*conj(Om23)/2.0)
            J[8, 2] = -im*conj(Om23)
            J[8, 3] = -(im*conj(Om13)/2.0)
            J[8, 4] = 0.0 * im
            J[8, 5] = 0.0 * im
            J[8, 6] = 0.0 * im
            J[8, 7] = 0.0 * im
            J[8, 8] = -conj(p[11])
        end
        end
        end

        grid = zeros(ComplexF64, (N_grid, N_grid))
        counter_grid = zeros(Int32, (N_grid, N_grid))
        counter_grid_total = zeros(Int32, (N_grid, N_grid))
        grid_weighted = zeros(ComplexF64, (N_grid, N_grid))
        normalized = zeros(Float64, (N_grid, N_grid))
        Vs = collect(LinRange{Float64}(v0, v1, N_v))
        pv = sqrt(2.0/pi)*((m87/(k_B*T))^(3.0/2.0)).*Vs.^2.0 .*exp.(-m87*Vs.^2.0/(2.0*k_B*T))
        v_perps = zeros(Float64, N_real)
        global coords = Array{Tuple{Int32, Int32, Int32, Int32}}(undef, N_real)

        for (index_v, v) in ProgressBar(enumerate(Vs))
            paths = [[] for i=1:N_real]
            tpaths = [[] for i=1:N_real]
            # Choose points in advance to save only the relevant points during solving
            Threads.@threads for i = 1:N_real
                iinit, jinit, ifinal, jfinal = choose_points()
                coords[i] = (iinit, jinit, ifinal, jfinal)
                paths[i] = bresenham(jinit, iinit, jfinal, ifinal)
                v_perps[i] = sqrt(v^2.0 - draw_vz(v)^2.0)
                tpaths[i] = Float64[hypot(coord[2]-iinit, coord[1]-jinit)*abs(window)/(v_perps[i]*N_grid) for coord in paths[i]]
            end
            function prob_func(prob, i, repeat)
                iinit, jinit, ifinal, jfinal = coords[i]
                v_perp = v_perps[i]
                vz = sqrt(v^2.0 - v_perp^2.0)
                xinit = jinit*window/N_grid
                yinit = iinit*window/N_grid
                xfinal = jfinal*window/N_grid
                yfinal = ifinal*window/N_grid
                # velocity unit vector
                if v_perp != 0
                    u0 = xfinal-xinit
                    u1 = yfinal-yinit
                    norm = hypot(u0, u1)
                    u0 /= norm
                    u1 /= norm
                else
                    u0 = u1 = 0
                end
                new_p = ComplexF64[v_perp + 0.0*im, u0 + 0.0*im, u1 + 0.0*im,
                        xinit + 0.0*im, yinit + 0.0*im,
                        Gamma, Omega13, Omega23, gamma21tilde, gamma31tilde - im*k*vz,
                        gamma32tilde - im*k*vz, waist, r0]
                remake(prob, p=new_p, tspan=(0.0, maximum(tpaths[i])), saveat=tpaths[i])

            end

            # instantiate a problem
            p = ComplexF64[1.0 + 0.0*im, 1.0 + 0.0*im, 1.0 + 0.0*im, 1.0 + 0.0*im, 1.0 + 0.0*im,
                Gamma, Omega13,
                Omega23, gamma21tilde,
                gamma31tilde - im*k*0.0,
                gamma32tilde - im*k*0.0, waist, r0]
            tspan = (0.0, 1.0)
            tsave = collect(LinRange{Float64}(tspan[1], tspan[2], 2))
            prob = ODEProblem{true}(f!, x0, tspan, p, jac=f_jac!, saveat=tsave)
            ensembleprob = EnsembleProblem(prob, prob_func=prob_func)
            # select appropriate solver
            if abs(waist) > 0.75e-3
                alg = Rodas4P(autodiff=false)
                atol = 1e-6
                rtol = 1e-5
            else
                alg = TRBDF2(autodiff=false)
                atol = 1e-6
                rtol = 1e-3
            end
            # run once on small system to try and speed up compile time
            if index_v==1
                sol = solve(ensembleprob, alg, EnsembleThreads(),
                                trajectories=2, save_idxs=7, abstol=atol, reltol=rtol)
            end
            sol = solve(ensembleprob, alg, EnsembleThreads(),
                            trajectories=N_real, save_idxs=7, abstol=atol, reltol=rtol)
            global grid .= zeros(ComplexF64, (N_grid, N_grid))
            global counter_grid .= zeros(Int32, (N_grid, N_grid))
            @inbounds begin
            Threads.@threads for i = 1:N_real
                for (j, coord) in enumerate(paths[i])
                    grid[coord[2], coord[1]] += sol[i].u[j]
                    counter_grid[coord[2], coord[1]] += 1
                end
            end
            end
            grid_weighted .+= (grid./counter_grid) * pv[index_v] #* (v1-v0/N_v)
            counter_grid_total .+= counter_grid
            
        end
        [grid_weighted, counter_grid_total]
        """)
        grid_weighted *= 2*N(self.T) * np.abs(self.mu23) / (self.E * epsilon_0)
        return grid_weighted, counter_grid

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