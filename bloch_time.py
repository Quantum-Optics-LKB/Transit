# -*-coding:utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import sys
from julia import Main
from odeintw import odeintw
from scipy.integrate import solve_ivp
from scipy.constants import (Boltzmann, c, e, elementary_charge, epsilon_0,
                             hbar, m_n, mu_0)
from numba import cfunc, complex128, float64, jit, types, void
from julia import Julia
# Compiler flags and options
jl = Julia(["--optimize=3", "--compile=all"])


@jit(nopython=True)
def bresenham(x1: int, y1: int, x2: int, y2: int) -> list:
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


def N(T: float) -> float:
    """Atomic density as a function of temperature. 
    From Siddons et al. 2008 
    "Absolute absorption on rubidium D lines: comparison between theory and experiment"

    Args:
        T (float): Temperature in K

    Returns:
        float: Atomic density in m^-3
    """
    P_v = 10**(15.88253 - 4529.635/T + 0.00058663*T - 2.99138*np.log10(T))
    return 133.323*P_v/(Boltzmann*T)


def fit(t: float, a: float, b: float, c: float) -> float:
    """Exponential fit for response time analysis

    Args:
        t (float): Time in seconds
        a (float): amplitude 
        b (float): response time
        c (float): offset

    Returns:
        float: The objective function
    """
    return a*np.exp(-t/b) + c


@jit(complex128[::1](float64, complex128[::1], complex128[::1], complex128[::1]),
     nopython=True, nogil=True, fastmath=True, cache=True)
def deriv(t, x, p, dy):
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
    dy[0] = (-Gamma/2)*x[0]-(Gamma/2)*x[1] + \
        (1j*np.conj(Om13)/2)*x[4]-(1j*Om13/2)*x[5]
    dy[1] = (-Gamma/2)*x[0]-(Gamma/2)*x[1] + \
        (1j*np.conj(Om23)/2)*x[6]-(1j*Om23/2)*x[7]
    dy[2] = -gamma21tilde*x[2]+(1j*np.conj(Om23)/2)*x[4]-(1j*Om13/2)*x[7]
    dy[3] = -np.conj(gamma21tilde)*x[3] - (1j*Om23/2) * \
        x[5] + (1j*np.conj(Om13)/2)*x[6]
    dy[4] = 1j*Om13*x[0] + (1j*Om13/2)*x[1] + \
        (1j*Om23/2)*x[2] - gamma31tilde*x[4]
    dy[5] = -1j*np.conj(Om13)*x[0]-1j*(np.conj(Om13)/2)*x[1] - \
        (1j*np.conj(Om23)/2)*x[3]-np.conj(gamma31tilde)*x[5]
    dy[6] = (1j*Om23/2)*x[0]+1j*Om23*x[1]+(1j*Om13/2)*x[3]-gamma32tilde*x[6]
    dy[7] = (-1j*np.conj(Om23)/2)*x[0]-1j*np.conj(Om23)*x[1] - \
        (1j*np.conj(Om13)/2)*x[2]-np.conj(gamma32tilde)*x[7]
    dy += b
    return dy


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def deriv_mut(dy, x, p, t):
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
    dy[0] = (-Gamma/2)*x[0]-(Gamma/2)*x[1] + \
        (1j*np.conj(Om13)/2)*x[4]-(1j*Om13/2)*x[5]
    dy[1] = (-Gamma/2)*x[0]-(Gamma/2)*x[1] + \
        (1j*np.conj(Om23)/2)*x[6]-(1j*Om23/2)*x[7]
    dy[2] = -gamma21tilde*x[2]+(1j*np.conj(Om23)/2)*x[4]-(1j*Om13/2)*x[7]
    dy[3] = -np.conj(gamma21tilde)*x[3] - (1j*Om23/2) * \
        x[5] + (1j*np.conj(Om13)/2)*x[6]
    dy[4] = 1j*Om13*x[0] + (1j*Om13/2)*x[1] + \
        (1j*Om23/2)*x[2] - gamma31tilde*x[4]
    dy[5] = -1j*np.conj(Om13)*x[0]-1j*(np.conj(Om13)/2)*x[1] - \
        (1j*np.conj(Om23)/2)*x[3]-np.conj(gamma31tilde)*x[5]
    dy[6] = (1j*Om23/2)*x[0]+1j*Om23*x[1]+(1j*Om13/2)*x[3]-gamma32tilde*x[6]
    dy[7] = (-1j*np.conj(Om23)/2)*x[0]-1j*np.conj(Om23)*x[1] - \
        (1j*np.conj(Om13)/2)*x[2]-np.conj(gamma32tilde)*x[7]
    dy += b


@jit(complex128[:, ::1](float64, complex128[::1], complex128[::1], complex128[::1]),
     nopython=True, nogil=True, fastmath=True, cache=True)
def deriv_jac(t, x, p, dy):
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
                  [-Gamma/2, -Gamma/2, 0, 0, 0, 0,
                      1j*np.conj(Om23)/2, -1j*Om23/2],
                  [0, 0, -gamma21tilde, 0, 1j *
                      np.conj(Om23)/2, 0, 0, -1j*Om13/2],
                  [0, 0, 0, -np.conj(gamma21tilde), 0, -1j*Om23 /
                   2, 1j*np.conj(Om13)/2, 0],
                  [1j*Om13, 1j*Om13/2, 1j*Om23/2, 0, -gamma31tilde, 0, 0, 0],
                  [-1j*np.conj(Om13), -1j*np.conj(Om13)/2, 0, -1j *
                   np.conj(Om23)/2, 0, -np.conj(gamma31tilde), 0, 0],
                  [1j*Om23/2, 1j*Om23, 0, 1j*Om13/2, 0, 0, -gamma32tilde, 0],
                  [-1j*np.conj(Om23)/2, -1j*np.conj(Om23), -1j*np.conj(Om13)/2, 0, 0, 0, 0, -np.conj(gamma32tilde)]],
                 dtype=np.complex128)
    return A


class temporal_bloch:

    def __init__(self, T: float, puiss: float, waist: float, detun: float, L: float,
                 N_grid: int = 128, N_v: int = 20, N_real: int = 5000, N_proc: int = 15):
        """Initializes all the parameters of the simulation and all hardtyped values.
        Almost all attributes are defined as properties such that when changing core attributes 
        of the experiment (power, waist, window size ...), all subsequent calls to properties 
        will return the updated values.

        Args:
            T (float): Cell temperature in C
            puiss (float): Input power in W
            waist (float): Beam waist in m
            detun (float): Laser detuning in Hz
            L (float): Cell length in m
            N_grid (int, optional): Number of grid points of the computational window. Defaults to 128.
            N_v (int, optional): Number of velocity classes. Defaults to 20.
            N_real (int, optional): Number of Monte-Carlo realizations. Defaults to 5000.
            N_proc (int, optional): Number of CPU cores used. Defaults to 15.
        """
        # grid params for MC
        self.N_grid = N_grid
        self.N_v = N_v
        self.N_real = N_real
        self.T = T
        self.L = L
        self.waist = waist
        self.window = 4*self.waist
        self.detun = 2*np.pi*detun
        self.puiss = puiss
        self.frac = 0.995
        self.wl = 780.241e-9
        self.k = 2*np.pi/self.wl
        self.Gamma = 2*np.pi * 6.0666e6
        self.m87 = 1.44316060e-25
        self.m85 = 1.44316060e-25 - 2*m_n
        self.u = np.sqrt(
            2*Boltzmann*T/(self.frac*self.m87+(1-self.frac)*self.m85))
        self.gamma_t = 0.0
        self.G1 = 3/8
        self.G2 = 5/8
        self.delta0 = 2*np.pi * 6.834e9
        self.x0 = np.array([self.G1 + 1j*0, self.G2 + 1j*0, 0, 0, 0, 0, 0, 0],
                           dtype=np.complex128)
        self.x0_short = np.array([self.G1 + 1j*0, self.G2 + 1j*0, 0, 0],
                                 dtype=np.complex128)

    @property
    def r0(self):
        return self.window/2

    @property
    def I(self):
        return 2*self.puiss/(np.pi*self.waist**2)

    @property
    def E(self):
        return np.sqrt(2*self.I/(c*epsilon_0))

    @property
    def X(self):
        return np.meshgrid(np.linspace(0, self.window, self.N_grid), np.linspace(0, self.window, self.N_grid))[0]

    @property
    def Y(self):
        return np.meshgrid(np.linspace(0, self.window, self.N_grid), np.linspace(0, self.window, self.N_grid))[1]

    @property
    def Kx(self):
        kx = 2 * np.pi * np.fft.fftfreq(self.N_grid, d=self.window/self.N_grid)
        ky = 2 * np.pi * np.fft.fftfreq(self.N_grid, d=self.window/self.N_grid)
        return np.meshgrid(kx, ky)[0]

    @property
    def Ky(self):
        kx = 2 * np.pi * np.fft.fftfreq(self.N_grid, d=self.window/self.N_grid)
        ky = 2 * np.pi * np.fft.fftfreq(self.N_grid, d=self.window/self.N_grid)
        return np.meshgrid(kx, ky)[1]

    @property
    def E_map(self):
        return (self.E + 1j*0)*np.exp(-((self.X-self.r0)**2 + (self.Y-self.r0)**2)/(2*self.waist**2))

    @property
    def I_map(self):
        return self.I*np.exp(-((self.X-self.r0)**2 + (self.Y-self.r0)**2)/(2*self.waist**2))

    @property
    def d(self):
        return np.sqrt(9*epsilon_0*hbar*self.Gamma * self.wl**3/(8*np.pi**2))

    @property
    def gamma_t_analytical(self):
        # return np.sqrt(2*Boltzmann*self.T/(self.m87*np.log(2)*np.pi*self.waist**2))
        return self.u/(self.waist*np.sqrt(np.pi))

    @property
    def gamma(self):
        return self.Gamma/2 + self.gamma_t

    @property
    def gamma_analytical(self):
        return self.Gamma/2 + self.gamma_t_analytical

    @property
    def gamma32tilde(self):
        return self.gamma - 1j*self.detun

    @property
    def gamma31tilde(self):
        return self.gamma - 1j*(self.detun-self.delta0)

    @property
    def gamma21tilde(self):
        return self.gamma_t + 1j*self.delta0

    @property
    def mu23(self):
        return (1/np.sqrt(5))*np.sqrt(1/18 + 5/18 + 7/9)*self.d + 1j*0

    @property
    def mu13(self):
        return (1/np.sqrt(3))*np.sqrt(1/9 + 5/18 + 5/18)*self.d + 1j*0

    @property
    def Omega23(self):
        return self.E*self.mu23/hbar

    @property
    def Omega13(self):
        return self.E*self.mu13/hbar

    def propagate_field(self, chi: np.ndarray, dz: float):
        """Propagates the field one dz step given a certain medium susceptibility chi.
        Needs a high resolution of the computational window to work.

        Args:
            chi (np.ndarray): Susceptibility
            dz (float): Propagation distance in m

        Returns:
            np.ndarray: Propagated field
        """
        propag = np.exp(-1j * (self.Kx**2 + self.Ky**2) * dz/(2*self.k))
        lin_phase = np.exp(-1j*self.k/2 * chi)
        E_map = self.E_map.copy()
        E_map *= lin_phase
        E_map = np.fft.fft2(E_map)
        E_map *= propag
        E_map = np.fft.ifft2(E_map)
        self.puiss = c/2 * epsilon_0 * \
            np.abs(E_map[self.N_grid//2, self.N_grid//2])**2 * \
            np.pi*self.waist**2
        return self.E_map

    def choose_points(self, plot=False) -> tuple:
        """Chooses random start and end points.

        Args:
            plot (bool, optional): Plots the grid with generated poitns.
            For debugging purpopses. Defaults to False.

        Returns:
            tuple(int): The 4 coordinates of the start and end points.
        """
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

    def draw_vz(self, v: float) -> float:
        """Draws the random velocity along z assuming a total velocity v.
        Follow the 1D Maxwell-Boltzmann distribution.

        Args:
            v (float): Value of the total velocity in m/s

        Returns:
            float: The chosen velocity along z.
        """
        vz = np.abs(2*v)
        while np.abs(vz) > np.abs(v):
            vz = np.random.normal(0, np.sqrt(Boltzmann*self.T/self.m87))
        return vz

    def integrate_short(self, v: float, ts: np.ndarray, xinit: float, yinit: float,
                        xfinal: float, yfinal: float) -> tuple:
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

    def integrate(self, vz: float, v_perp: float, iinit: int, jinit: int, ifinal: int,
                  jfinal: int, ynext: np.ndarray) -> tuple:
        """Integrates the MBE over one trajectory.

        Args:
            vz (float): Velocity of the atom over z
            v_perp (float): Velocity of the atom in the xy plane
            iinit (int): Initial position row
            jinit (int): Initial position column
            ifinal (int): Final position row
            jfinal (int): Final position column
            ynext (np.ndarray): Pre-allocated vector of size 8 when using the mutating 
            function deriv_mut. Allows better speed.

        Returns:
            tuple(float, float, int): [description]
        """
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
            t_path = np.array([np.hypot(_[1]-iinit, _[0]-jinit)
                               * self.window/(self.N_grid*v_perp) for _ in path])
        else:
            u0 = u1 = 0
            t_path = np.array([np.hypot(_[1]-iinit, _[0]-jinit)
                               * self.window/(self.N_grid*np.abs(vz)) for _ in path])
        tfinal = t_path[-1]
        # print(f'tfinal = {tfinal*1e6} us')
        # ts = np.arange(0, tfinal, 1e-10, dtype=np.float64)
        p = np.array([v_perp, u0, u1, xinit, yinit, self.Gamma, self.Omega13,
                      self.Omega23, self.gamma21tilde,
                      self.gamma31tilde - 1j*self.k*vz,
                      self.gamma32tilde - 1j*self.k*vz, self.waist, self.r0],
                     dtype=np.complex128)
        ys, infodict = odeintw(deriv, self.x0, t_path,
                               args=(p, ynext),
                               Dfun=deriv_jac,
                               full_output=True, hmax=1e-7, hmin=1e-16,
                               h0=1e-14, tfirst=True)
        return t_path, ys, path

    def chi_analytical(self, v: float) -> np.ndarray:
        """Steady state susceptibility over the 2->3 transition in the far 
        detuned approximation

        Args:
            v (float): Velocity class in m/s

        Returns:
            np.ndarray: The susceptibility map
        """
        a = (self.Gamma/2)/self.gamma_analytical
        b = self.gamma_t_analytical/self.gamma_analytical
        fac = np.sqrt(2*b*(1+a)/(1+b))
        # Es0 = fac*hbar*self.gamma_analytical/self.mu23
        gamma_analytical = self.gamma/2
        Es0 = hbar*gamma_analytical/self.mu23
        Is0 = 0.5*epsilon_0*c*Es0**2
        Delta = self.detun + self.k*v
        Is = Is0*(1 + (Delta/gamma_analytical)**2)
        # print(f"{Is=}")
        Es = np.sqrt(2*Is/(epsilon_0*c))
        pref = self.G2*N(self.T)/(epsilon_0*hbar) * \
            abs(self.mu23)**2/gamma_analytical
        chi = pref * (1j-Delta/gamma_analytical) / \
            (1+(Delta/gamma_analytical)**2 + (self.E_map/Es0)**2)
        return chi

    def chi_analytical_2(self, v: float) -> np.ndarray:
        """Steady state susceptibility over the 2->3 / 1->3 transitions 
        in the far detuned approximation

        Args:
            v (float): Velocity class in m/s

        Returns:
            np.ndarray: The susceptibility map
        """
        a = (self.Gamma/2)/self.gamma_analytical
        b = self.gamma_t_analytical/self.gamma_analytical
        fac = np.sqrt(2*b*(1+a)/(1+b))
        Es0 = fac*hbar*self.gamma_analytical/self.mu23
        Es1 = fac*hbar*self.gamma_analytical/self.mu13
        Is0 = 0.5*epsilon_0*c*Es0**2
        Is01 = 0.5*epsilon_0*c*Es1**2
        Delta = self.detun + self.k*v
        Is = Is0*(1 + (Delta/self.gamma_analytical)**2)
        Is1 = Is01*(1 + (Delta-self.delta0/self.gamma_analytical)**2)
        # print(f"{Is=}")
        Es = np.sqrt(2*Is/(epsilon_0*c))
        pref = self.G2*N(self.T)/(epsilon_0*hbar) * \
            abs(self.mu23)**2/self.gamma_analytical
        pref1 = self.G1*N(self.T)/(epsilon_0*hbar) * \
            abs(self.mu13)**2/self.gamma_analytical
        chi = pref * (1j-Delta/self.gamma_analytical) / \
            (1+(Delta/self.gamma_analytical)**2 + (self.E_map/Es0)**2)
        chi += pref1 * ((1j-Delta - self.delta0)/self.gamma_analytical) / \
            (1+((Delta-self.delta0)/self.gamma_analytical)**2 + (self.E_map/Es1)**2)
        return chi

    def do_V_span(self, v0: float, v1: float, N_v: int) -> tuple:
        """Averages the susceptibility over N_real realizations and N_v velocity 
        classes. Computes the suscpeptibility on both transitions 2->3 and 1->3.
        Uses the julia 'code do_V_span.jl'.
        Args:
            v0 (float): Starting velocity. Should not be too low to avoid stalling the 
            solver with very long evolution times.
            v1 (float): End velocity
            N_v (int): Number of velocity classes

        Returns:
            tuple(np.ndarray, np.ndarray): Averaged susceptibility map and number of counts.
        """
        Main.eval(f"global N_v = {N_v}")
        Main.eval(f"global v0 = {v0}")
        Main.eval(f"global v1 = {v1}")
        Main.eval(f"global N_real = {self.N_real}")
        Main.eval(f"global N_grid = {self.N_grid}")
        Main.eval(f"global T = {self.T}")
        Main.eval(f"global window = {self.window}")
        Main.eval(
            f"global Gamma = {np.real(self.Gamma)}+{np.imag(self.Gamma)}*im")
        Main.eval(
            f"global Omega13 = {np.real(self.Omega13)}+{np.imag(self.Omega13)}*im")
        Main.eval(
            f"global Omega23 = {np.real(self.Omega23)}+{np.imag(self.Omega23)}*im")
        Main.eval(
            f"global gamma21tilde = {np.real(self.gamma21tilde)}+{np.imag(self.gamma21tilde)}*im")
        Main.eval(
            f"global gamma31tilde = {np.real(self.gamma31tilde)}+{np.imag(self.gamma31tilde)}*im")
        Main.eval(
            f"global gamma32tilde = {np.real(self.gamma32tilde)}+{np.imag(self.gamma32tilde)}*im")
        Main.eval(f"global waist = {self.waist} + 0*im")
        Main.eval(f"global r0 = {self.r0} + 0*im")
        Main.eval(
            f"global x0 = [{self.G1} + 0*im, {self.G2} + im*0, 0.0*im, 0.0*im, 0.0*im, 0.0*im, 0.0*im, 0.0*im]")
        Main.eval(f"global k = {self.k}")
        with open("do_V_span.jl", mode='r') as f:
            grid_weighted_13, grid_weighted_23, counter_grid = Main.eval(
                f.read())
        grid_weighted = 2*N(self.T) * (np.abs(self.mu13) * grid_weighted_13 +
                                       np.abs(self.mu23) * grid_weighted_23) / (self.E_map * epsilon_0)
        return grid_weighted, counter_grid

    def do_V_span_pop(self, v0: float, v1: float, N_v: int) -> tuple:
        """Averages the populations over N_real realizations and N_v velocity 
        classes.
        Uses the julia 'code do_V_span_pop.jl'.

        Args:
            v0 (float): Starting velocity. Should not be too low to avoid stalling the 
            solver with very long evolution times.
            v1 (float): End velocity
            N_v (int): Number of velocity classes

        Returns:
            tuple(np.ndarray, np.ndarray): Averaged population map and number of counts.
        """
        Main.eval(f"global N_v = {N_v}")
        Main.eval(f"global v0 = {v0}")
        Main.eval(f"global v1 = {v1}")
        Main.eval(f"global N_real = {self.N_real}")
        Main.eval(f"global N_grid = {self.N_grid}")
        Main.eval(f"global T = {self.T}")
        Main.eval(f"global window = {self.window}")
        Main.eval(
            f"global Gamma = {np.real(self.Gamma)}+{np.imag(self.Gamma)}*im")
        Main.eval(
            f"global Omega13 = {np.real(self.Omega13)}+{np.imag(self.Omega13)}*im")
        Main.eval(
            f"global Omega23 = {np.real(self.Omega23)}+{np.imag(self.Omega23)}*im")
        Main.eval(
            f"global gamma21tilde = {np.real(self.gamma21tilde)}+{np.imag(self.gamma21tilde)}*im")
        Main.eval(
            f"global gamma31tilde = {np.real(self.gamma31tilde)}+{np.imag(self.gamma31tilde)}*im")
        Main.eval(
            f"global gamma32tilde = {np.real(self.gamma32tilde)}+{np.imag(self.gamma32tilde)}*im")
        Main.eval(f"global waist = {self.waist} + 0*im")
        Main.eval(f"global r0 = {self.r0} + 0*im")
        Main.eval(
            f"global x0 = [{self.G1} + 0*im, {self.G2} + im*0, 0.0*im, 0.0*im, 0.0*im, 0.0*im, 0.0*im, 0.0*im]")
        Main.eval(f"global k = {self.k}")
        with open("do_V_span_pop.jl") as f:
            grid_pop_weighted, grid_coh_weighted, counter_grid = Main.eval(
                f.read())
        return grid_pop_weighted, grid_coh_weighted, counter_grid
