import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

path = "/home/tangui/Documents/Transit/results"
filename = "n2_w_Wed Sep 15 01:03:40 2021"
n2 = np.load(f"{path}/{filename}.npy")
n2 = np.asarray(np.real(n2))
# plt.imshow(n2, origin='lower', norm=colors.SymLogNorm(linthresh=1e-15, linscale=1,
#                                               vmin=np.nanmin(n2), vmax=0, base=10))
plt.imshow(n2, origin='lower', vmin=np.nanmin(n2), vmax=0)
plt.title("n2")
plt.colorbar()
plt.show()