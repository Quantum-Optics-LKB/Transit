import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.optimize import curve_fit

def fit_Isat(I, Isat, n2_0):
    return n2_0/(1+I/Isat)
# popt, pcov = curve_fit(fit_Isat, powers, n2_P)
waists_murad = np.asarray([0.054e-3, 8.350e-4, 9.400e-05, 2.050e-04, 3.680e-04, 8.900e-04, 0.00130,
                               0.00186, 0.00390, 0.00250], dtype=np.float64)
n2_murad = np.asarray([1.466451787760415e-10, 6.146519158111747e-09, 4.648785173881415e-10,
                        1.296385036598109e-09, 2.742352966926649e-09, 4.018992194568556e-09,
                        6.691264052482064e-09, 9.751877517360527e-09, 1.445521229964598e-08,
                        1.544469999857501e-08], dtype=np.float64)
I_sat_murad = np.asarray([5.602997069021627e+02, 11.770124366553920, 2.053285829870257e+02,
                            60.502837888731115, 30.902688270484745, 58.693562254310365,
                            11.395745635138198, 6.867638307604812, 5.469433881999493,
                            2.784375157084676], dtype=np.float64)


path = "/home/tangui/Documents/LKB/Transit/results"
filename = "n2_center_murad_w_P_Mon Sep 20 11:31:01 2021"
n2 = np.load(f"{path}/{filename}.npy")
n2 = np.asarray(n2)
p = np.load(f"{path}/Ps_Mon Sep 20 11:31:01 2021.npy")
# plt.imshow(n2, origin='lower', norm=colors.SymLogNorm(linthresh=1e-15, linscale=1,
#                                               vmin=np.nanmin(n2), vmax=0, base=10))
fig, ax = plt.subplots(2, 5)
for idx in range(10):
    ax[idx//5, idx%5].plot(p, np.abs(n2[idx, :]))
    ax[idx//5, idx%5].set_xlabel("Power in W")
    ax[idx//5, idx%5].set_ylabel("|$n_{2}$|")
    ax[idx//5, idx%5].set_title(f"n2, w = {np.round(waists_murad[idx]*1e3, decimals=2)} mm")
plt.show()