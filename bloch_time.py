# -*-coding:utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, hbar, c, mu_0, Boltzmann,  m_n,  e
from odeintw import odeintw
from scipy.signal import savgol_filter
from numba import jit
import sys


def N(T):
    P_v = 10**(15.88253 - 4529.635/T + 0.00058663*T - 2.99138*np.log10(T))
    return 133.323*P_v/(Boltzmann*T)


def beta_n(T):
    return 2*np.pi*1.03e-13*N(T=T)


def fit(t, a, b, c):
    return a*np.exp(-t/b) + c


class temporal_bloch:

    def __init__(self, T, puiss, waist, detun, L):

        self.T = T
        self.L = L
        self.waist = waist
        self.r0 = self.waist
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

        self.x0 = np.array([self.G1 + 1j*0, self.G2 + 1j*0, 0, 0, 0, 0, 0, 0])
        self.x0_short = np.array([self.G1 + 1j*0, self.G2 + 1j*0, 0, 0])

        print(f"Omega23 = {np.real(self.Omega23)*1e-9/(2*np.pi)} GHz")

    def deriv(self, x, t, vp=False):
        """Computes the differential term of MBE in the case of a 3 level system
        with one coupling field. The x vector represents the density matrix
        elements rho_11, 22, 21, 12, 31, 13, 32, 23

        :param np.ndarray x: Density matrix coefficients vector
        :param float t: Time
        :param bool vp: Returns or not the eigenvalues
        :return: The updated density matrix elements
        :rtype: np.ndarray

        """

        A = np.array([[-self.gamma_t-self.Gamma/2, -self.Gamma/2, 0, 0, 1j*np.conj(self.Omega13)/2, -1j*self.Omega13/2, 0, 0],
             [-self.Gamma/2, -self.gamma_t-self.Gamma/2, 0, 0, 0, 0, 1j*np.conj(self.Omega23)/2, -1j*self.Omega23/2],
             [0, 0, -self.gamma21tilde, 0, 1j*np.conj(self.Omega23)/2, 0, 0, -1j*self.Omega13/2],
             [0, 0, 0, -np.conj(self.gamma21tilde), 0, -1j*self.Omega23/2, 1j*np.conj(self.Omega13)/2, 0],
             [1j*self.Omega13, 1j*self.Omega13/2, 1j*self.Omega23/2, 0, -self.gamma31tilde, 0, 0, 0],
             [-1j*np.conj(self.Omega13), -1j*np.conj(self.Omega13)/2, 0, -1j*np.conj(self.Omega23)/2, 0, -np.conj(self.gamma31tilde), 0, 0],
             [1j*self.Omega23/2, 1j*self.Omega23, 0, 1j*self.Omega13/2, 0, 0, -self.gamma32tilde, 0],
             [-1j*np.conj(self.Omega23)/2, -1j*np.conj(self.Omega23), -1j*np.conj(self.Omega13)/2, 0, 0, 0, 0, -np.conj(self.gamma32tilde)]])

        b = np.array([self.Gamma/2 + self.G1*self.gamma_t, +self.Gamma/2 + self.G2*self.gamma_t, 0, 0, -1j*self.Omega13/2, 1j*np.conj(self.Omega13)/2, -1j*self.Omega23/2, 1j*np.conj(self.Omega23)/2])

        if vp:
            return np.sort(np.abs(np.real(np.linalg.eig(A)[0])))[0]

        return np.matmul(A, x) + b*np.heaviside(t, 1)

    def deriv_notransit(self, x, t, v):
        """Computes the differential term of MBE in the case of a 3 level system
        with one coupling field. The x vector represents the density matrix
        elements rho_11, 22, 21, 12, 31, 13, 32, 23

        :param np.ndarray x: Density matrix coefficients vector
        :param float t: Time
        :param float v: Atom initial velocity
        :return: The updated density matrix elements
        :rtype: np.ndarray

        """

        Gamma = self.Gamma
        gamma21tilde = self.gamma21tilde
        gamma31tilde = self.gamma31tilde + self.k*v
        gamma32tilde = self.gamma32tilde + self.k*v
        Omega13 = self.Omega13*np.exp(-(v*t-self.r0)**2/(2*self.waist))
        Omega23 = self.Omega23*np.exp(-(v*t-self.r0)**2/(2*self.waist))
        A = np.array([[-Gamma/2, -Gamma/2, 0, 0, 1j*np.conj(Omega13)/2, -1j*Omega13/2, 0, 0],
             [-Gamma/2, -Gamma/2, 0, 0, 0, 0, 1j*np.conj(Omega23)/2, -1j*Omega23/2],
             [0, 0, -gamma21tilde, 0, 1j*np.conj(Omega23)/2, 0, 0, -1j*Omega13/2],
             [0, 0, 0, -np.conj(gamma21tilde), 0, -1j*Omega23/2, 1j*np.conj(Omega13)/2, 0],
             [1j*Omega13, 1j*Omega13/2, 1j*Omega23/2, 0, -gamma31tilde, 0, 0, 0],
             [-1j*np.conj(Omega13), -1j*np.conj(Omega13)/2, 0, -1j*np.conj(Omega23)/2, 0, -np.conj(gamma31tilde), 0, 0],
             [1j*Omega23/2, 1j*Omega23, 0, 1j*Omega13/2, 0, 0, -gamma32tilde, 0],
             [-1j*np.conj(Omega23)/2, -1j*np.conj(Omega23), -1j*np.conj(Omega13)/2, 0, 0, 0, 0, -np.conj(gamma32tilde)]])

        b = np.array([Gamma/2, Gamma/2, 0, 0, -1j*Omega13/2, 1j*np.conj(Omega13)/2, -1j*Omega23/2, 1j*np.conj(Omega23)/2])
        return np.matmul(A, x) + b*np.heaviside(t, 1)

    def deriv_short_notransit(self, x, t, v):
        Gamma = self.Gamma
        gamma32tilde = self.gamma32tilde + self.k*v
        Omega23 = self.Omega23*np.exp(-(v*t-self.r0)**2/(2*self.waist))
        A = np.array(
            [[-Gamma/2, -Gamma/2, 0, 0],
             [-Gamma/2, -Gamma/2, 1j/2*Omega23, -1j/2*Omega23],
             [1j/2*Omega23, 1j*Omega23, -1*gamma32tilde, 0],
             [-1j/2*Omega23, -1j*Omega23, 0, -1*np.conj(gamma32tilde)]])

        b = np.array([Gamma/2, Gamma/2, -1j/2*Omega23, 1j/2*Omega23])

        return np.matmul(A, x) + b*np.heaviside(t, 1)

    def deriv_short(self, x, t, vp=False):

        A = np.array([[-self.gamma_t-self.Gamma/2, -self.Gamma/2, 0, 0],
             [-self.Gamma/2, -self.gamma_t-self.Gamma/2, 1j/2*self.Omega23, -1j/2*self.Omega23],
             [1j/2*self.Omega23, 1j*self.Omega23, -1*self.gamma32tilde, 0],
             [-1j/2*self.Omega23, -1j*self.Omega23, 0, -1*np.conj(self.gamma32tilde)]])

        b = np.array([self.Gamma/2 + self.G1*self.gamma_t, +self.Gamma/2 + self.G2*self.gamma_t, -1j/2*self.Omega23, 1j/2*self.Omega23])

        if vp:
            return np.sort(np.abs(np.real(np.linalg.eig(A)[0])))[0]

        return np.matmul(A, x) + b*np.heaviside(t, 1)

    def integrate_short(self):
        t = np.arange(0, 10e-6, 5e-13)
        y = odeintw(self.deriv_short, self.x0_short, t, full_output=False)

        return t, y

    def integrate(self):
        t = np.arange(0, 2e-6, 2e-13)
        y = odeintw(self.deriv, self.x0, t, full_output=False)

        return t, y

    def integrate_notransit(self, v, ts, xinit, yinit, xfinal, yfinal):
        # y, infodict = odeintw(self.deriv_notransit, self.x0, ts, args=(v,),
        #             full_output=True, hmax=1e-6, hmin=1e-14, h0=1e-14)
        # return ts, y, infodict
        Gamma = self.Gamma
        gamma21tilde = self.gamma21tilde
        gamma31tilde = self.gamma31tilde + self.k*v
        gamma32tilde = self.gamma32tilde + self.k*v
        Omega13 = self.Omega13
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
        def deriv_notransit(self, x, t, v):
            """Computes the differential term of MBE in the case of a 3 level system
            with one coupling field. The x vector represents the density matrix
            elements rho_11, 22, 21, 12, 31, 13, 32, 23

            :param np.ndarray x: Density matrix coefficients vector
            :param float t: Time
            :param float v: Atom initial velocity
            :return: The updated density matrix elements
            :rtype: np.ndarray

            """
            r = np.hypot(xinit+u0*v*t - r0, yinit+u1*v*t - r0)
            Om23 = Omega23 * np.exp(-r**2/(2*waist**2))
            Om13 = Omega13 * np.exp(-r**2/(2*waist**2))
            A = np.array([[-Gamma/2, -Gamma/2, 0, 0, 1j*np.conj(Om13)/2, -1j*Om13/2, 0, 0],
                 [-Gamma/2, -Gamma/2, 0, 0, 0, 0, 1j*np.conj(Om23)/2, -1j*Om23/2],
                 [0, 0, -gamma21tilde, 0, 1j*np.conj(Om23)/2, 0, 0, -1j*Om13/2],
                 [0, 0, 0, -np.conj(gamma21tilde), 0, -1j*Om23/2, 1j*np.conj(Om13)/2, 0],
                 [1j*Om13, 1j*Om13/2, 1j*Om23/2, 0, -gamma31tilde, 0, 0, 0],
                 [-1j*np.conj(Om13), -1j*np.conj(Om13)/2, 0, -1j*np.conj(Om23)/2, 0, -np.conj(gamma31tilde), 0, 0],
                 [1j*Om23/2, 1j*Om23, 0, 1j*Om13/2, 0, 0, -gamma32tilde, 0],
                 [-1j*np.conj(Om23)/2, -1j*np.conj(Om23), -1j*np.conj(Om13)/2, 0, 0, 0, 0, -np.conj(gamma32tilde)]])

            b = np.array([Gamma/2, Gamma/2, 0, 0, -1j*Om13/2, 1j*np.conj(Om13)/2, -1j*Om23/2, 1j*np.conj(Om23)/2])
            return np.matmul(A, x) + b*np.heaviside(t, 1)
        y = odeintw(self.deriv_notransit, self.x0, ts, args=(v,),
                    full_output=False, hmax=1e-4, hmin=2e-14, h0=2e-14)
        return ts, y

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


if __name__=="__main__":

    detun = -1e9
    waist = 1e-3
    T     = 415
    puiss = 0.5


    high = temporal_bloch(T=T, puiss=puiss, waist=waist, detun=detun, L=0.1)
    """
    print(1e6/(high.deriv(x=np.zeros(4), t=0, vp=True)))

    t, y   = high.integrate()
    t_plot = np.linspace(0, len(t)-1, 50000, dtype=np.int32)
    t_plot_2 = np.logspace(np.log(t_plot[1]), np.log(t_plot[-1]), 5000, base=np.exp(1), dtype=np.int32)
    #n_high = np.sqrt(1+2*N(high.T)/(epsilon_0*high.E)*high.mu23*np.real(y[t_plot, 2]))
    n_high = np.sqrt(1+2*N(high.T)/(epsilon_0*high.E)*(high.mu23*np.real(y[t_plot, 6])+high.mu13*np.real(y[t_plot, 4])))

    plt.plot(t[t_plot_2]*1e6, np.abs(y[t_plot_2, 0]), color='b')
    plt.plot(t[t_plot_2]*1e6, np.abs(y[t_plot_2, 1]), color='r')
    plt.plot(t[t_plot_2]*1e6, 1-np.abs(y[t_plot_2, 1])-np.abs(y[t_plot_2, 0]), color='orange')
    #np.savetxt('rho_11.txt', np.column_stack(([t[t_plot_2]*1e6, np.abs(y[t_plot_2, 0])])))
    #np.savetxt('rho_22.txt', np.column_stack(([t[t_plot_2]*1e6, np.abs(y[t_plot_2, 1])])))
    #np.savetxt('rho_33.txt', np.column_stack(([t[t_plot_2]*1e6,1-np.abs(y[t_plot_2, 1])-np.abs(y[t_plot_2, 0])])))

    low   = temporal_bloch(T=T, puiss=1e-8, waist=waist, detun=detun, L=0.1)
    t, y  = low.integrate()
    #n_low = np.sqrt(1+2*N(low.T)/(epsilon_0*low.E)*low.mu23*np.real(y[t_plot, 2]))
    n_low = np.sqrt(1+2*N(low.T)/(epsilon_0*low.E)*(low.mu23*np.real(y[t_plot, 6])+low.mu13*np.real(y[t_plot, 4])))

    #plt.plot(t[t_plot]*1e6, np.abs(y[t_plot, 0]), '--', color='b')
    #plt.plot(t[t_plot]*1e6, np.abs(y[t_plot, 1]), '--', color='r')
    #plt.plot(t[t_plot]*1e6, 1-np.abs(y[t_plot, 1])-np.abs(y[t_plot, 0]), '--', color='orange')

    plt.xlabel(r"Time in $\mu$ s")
    plt.ylabel("Populations")
    plt.xscale('log')
    #plt.legend(["$\\rho_{11}$", "$\\rho_{22}$", "$\\rho_{33}$"])
    #tikz.save('population_2.tex')

    plt.figure(2)

    index = np.real(n_high - n_low)

    np.save('index', index)
    """
    index = np.load('index.npy')
    t = np.arange(0, 2e-6, 2e-13)
    t_plot = np.linspace(0, len(t)-1, 50000, dtype=np.int32)

    index = savgol_filter(index, 10001, 3)
    ptot, pcov = curve_fit(fit, t[t_plot], np.abs(index) , p0=[index[0]-index[-1], 1e-6 , index[-1]])

    a, b = high.Gamma/(2*high.gamma), high.gamma_t/high.gamma
    f    = np.real(1 + (high.detun/high.gamma)**2 + (1+b)/(2*b*(1+a))*(high.Omega23/high.gamma)**2)
    #print((high.detun/high.gamma)**2, (1+b)/(2*b*(1+a))*(high.Omega23/high.gamma)**2)
    # un essai plutot concluant : 4*np.pi/high.Gamma*(high.gamma/np.real(high.Omega23))*(1+np.abs(high.detun/high.gamma))

    #plt.plot(t[t_plot]*1e6, index, label=r'Simulation: {:.3} $\mu$s'.format(1e6*1/high.deriv_short(x=np.zeros(4), t=0, vp=True)))
    plt.plot(t[t_plot][::100]*1e6, np.abs(index)[::100], label=r'Simulation: {:.3} $\mu$s'.format(1e6*1/high.deriv(x=np.zeros(8), t=0, vp=True)))
    plt.plot(t[t_plot][::100]*1e6, fit(t[t_plot][::100], *ptot), '--', label=r'Fit: $\tau = ${:.3} $\mu$s'.format(ptot[1]*1e6))
    plt.xlabel(r"Time in $\mu$ s")
    plt.ylabel(r"$\Delta$ n")
    plt.legend()
    #plt.xscale('log')
    # tikz.save('nl_index.tex')
    plt.show()

    print(1e6/(high.deriv_short(x=np.zeros(4), t=0, vp=True)))
