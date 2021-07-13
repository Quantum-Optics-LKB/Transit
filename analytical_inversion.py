# -*- coding: utf-8 -*-
"""
Created by Tangui Aladjidi on the 03/03/2021
"""

import numpy as np
from sympy import *

gamma_t = symbols('gamma_t')
Gamma = symbols('Gamma')
Omega13 = symbols('Omega13')
Omega23 = symbols('Omega23')
gamma21tilde = symbols('gamma21tilde')
gamma31tilde = symbols('gamma31tilde')
gamma32tilde = symbols('gamma32tilde')
M = SparseMatrix([[-gamma_t-Gamma/2, -Gamma/2, 0, 0, 1j*conjugate(Omega13)/2, -1j*Omega13/2, 0, 0],
            [Gamma/2, gamma_t+Gamma/2, 0, 0, 0, 0, 1j*conjugate(Omega23)/2, -1j*Omega23/2],
            [0, 0, -gamma21tilde, 0, 1j*conjugate(Omega23)/2, 0, 0, -1j*Omega13/2],
            [0, 0, 0, -conjugate(gamma21tilde), 0, -1j*Omega23/2, 1j*conjugate(Omega13)/2, 0],
            [1j*Omega13, 1j*Omega13/2, 1j*Omega23/2, 0, -gamma31tilde, 0, 0, 0],
            [-1j*conjugate(Omega13), -1j*conjugate(Omega13)/2, 0, -1j*conjugate(Omega23)/2, 0, -conjugate(gamma31tilde), 0, 0],
            [1j*Omega23/2, 1j*Omega23, 0, 1j*Omega13/2, 0, 0, -gamma32tilde, 0],
            [-1j*conjugate(Omega23)/2, -1j*conjugate(Omega23), -1j*conjugate(Omega13)/2, 0, 0, 0, 0, -conjugate(gamma32tilde)]])
invM = M.inv(try_block_diag=True)
