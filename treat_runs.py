import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.optimize import curve_fit
from bloch_time import temporal_bloch
from scipy.constants import Boltzmann
from julia import Main

T = 150+273  # cell temp
puiss = 560e-3  # power in W
waist = 1.85e-3  # beam waist
I = puiss/(np.pi*waist**2)
I_low = 1e-9/(np.pi*waist**2)
detun = -2.2e9  # detuning
L = 10e-3  # cell length
N_grid = 128
N_v = 20
v0 = 40.0
v1 = 800.0
N_real = 5000
N_proc = 16


def power_fit(x, a, b):
    return a*x**b


def lin_fit(x, a, b):
    return a*x + b


waists_murad = np.asarray([0.054e-3, 8.350e-4, 9.400e-05, 2.050e-04, 3.680e-04, 8.900e-04, 0.00130,
                           0.00186, 0.00390, 0.00250], dtype=np.float64)
idx_sorted = np.argsort(waists_murad)
# n2_murad = np.asarray([1.466451787760415e-10, 6.146519158111747e-09, 4.648785173881415e-10,
#                        1.296385036598109e-09, 2.742352966926649e-09, 4.018992194568556e-09,
#                        6.691264052482064e-09, 9.751877517360527e-09, 1.445521229964598e-08,
#                        1.544469999857501e-08], dtype=np.float64)
# I_sat_murad = np.asarray([5.602997069021627e+02, 11.770124366553920, 2.053285829870257e+02,
#                           60.502837888731115, 30.902688270484745, 58.693562254310365,
#                           11.395745635138198, 6.867638307604812, 5.469433881999493,
#                           2.784375157084676], dtype=np.float64)*1e4
n2_murad = np.asarray([1.5, 61.4, 4.75, 14.6, 26.3, 32.4,
                       62, 81, 131, 150], dtype=np.float64)*1e-10
I_sat_murad = np.asarray(
    [570, 12, 185, 62, 32.4, 37, 13.2, 9.63, 3.4, 2.75], dtype=np.float64)*1e4
n2_murad = np.asarray([1.466451787760415e-10, 6.146519158111747e-09, 4.648785173881415e-10,
                       1.296385036598109e-09, 2.742352966926649e-09, 4.018992194568556e-09,
                       6.691264052482064e-09, 9.751877517360527e-09, 1.445521229964598e-08,
                       1.544469999857501e-08], dtype=np.float64)
I_sat_murad = np.asarray([5.602997069021627e+02, 11.770124366553920, 2.053285829870257e+02,
                          60.502837888731115, 30.902688270484745, 58.693562254310365,
                          11.395745635138198, 6.867638307604812, 5.469433881999493,
                          2.784375157084676], dtype=np.float64)*1e4
waists_err = np.array([2e-6, 8e-6, 1e-6, 5e-6, 1e-6,
                       10e-6, 0.05e-3, 0.03e-3, 0.03e-3, 0.03e-3])
n2_err = np.array([0.15, 4, 0.4, 2, 1.2, 3.1, 3, 3.5, 6, 9.3])*1e-10
Isat_err = np.array([70, 1.42, 25, 9.3, 2, 10, 1.4, 1.32, 0.6, 0.4])*1e4
waists_murad = waists_murad[idx_sorted]
n2_murad = n2_murad[idx_sorted]
I_sat_murad = I_sat_murad[idx_sorted]
n2_err = n2_err[idx_sorted]
Isat_err = Isat_err[idx_sorted]
waists_err = waists_err[idx_sorted]
powers = np.linspace(50e-3, 540e-3, 10)
Dn_P_murad = np.zeros((len(powers), N_grid, N_grid), dtype=np.float64)
path = "/home/tangui/Documents/Transit/results"
# filename = "n2_w9_murad_Mon Jan 31 14:56:05 2022"
# n2 = np.load(f"{path}/{filename}.npy")
# n2 = np.asarray(n2)
# fig, ax = plt.subplots(2, 5)
# fig1, ax1 = plt.subplots()
# n2_c = np.zeros(10)
# avg_zone = 10
# for i in range(10):
#     solver1 = temporal_bloch(T, 1e-9, waists_murad[i], detun, L, N_grid=N_grid,
#                                  N_v=N_v, N_real=N_real, N_proc=N_proc)
#     n2 = np.load(f"{path}/n2_w{i}_murad_Wed Feb  2 09:20:05 2022.npy")
#     n2[n2>0] = -1e-20
#     n2_c[i] = np.mean(n2[n2.shape[0]//2-avg_zone:n2.shape[0]//2+avg_zone,
#                                n2.shape[1]//2-avg_zone:n2.shape[1]//2+avg_zone])
#     ax[i//5, i%5].imshow(np.log10(np.abs(n2)*solver1.I_map))
#     ax[i//5, i%5].set_title(f"$w_0$ = {'{:.2e}'.format(waists_murad[i])}")
# popt, cov = curve_fit(power_fit, waists_murad, np.abs(n2_c))
# perr = np.sqrt(np.diag(cov))
# print(popt)
# popt1, cov1 = curve_fit(power_fit, waists_murad, n2_murad, maxfev=3200)
# perr1 = np.sqrt(np.diag(cov1))
# print(popt1)
# ax1.scatter(waists_murad, np.abs(n2_c), label="Computed")
# ax1.scatter(waists_murad, n2_murad, label="Exp")
# ax1.plot(waists_murad, power_fit(waists_murad, popt[0], popt[1], popt[2]),
#          label="Power law fit comp")
# ax1.plot(waists_murad, power_fit(waists_murad, popt1[0], popt1[1], popt1[2]),
#          label="Power law fit exp")
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# plt.show()
fig, ax = plt.subplots(2, 5)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots(2, 5)
indices = range(10)
Dn_c = np.zeros(len(indices))
Dn_analytical_c = np.zeros(len(indices))
Dn_analytical_c2 = np.zeros(len(indices))
avg_zone = 1
N_steps_z = 10
Vs = np.linspace(v0, v1, N_v)
dv = np.abs(Vs[0]-Vs[1])
for i in indices:
    Dn = np.zeros((N_grid, N_grid))
    Dn_analytical = np.zeros((N_grid, N_grid))
    Dn_analytical2 = np.zeros((N_grid, N_grid))
    solver = temporal_bloch(T, I_low*(np.pi*waists_murad[i]**2), waists_murad[i], detun, L, N_grid=N_grid,
                            N_v=N_v, N_real=N_real, N_proc=N_proc)
    solver1 = temporal_bloch(T, I*(np.pi*waists_murad[i]**2), waists_murad[i], detun, L, N_grid=N_grid,
                             N_v=N_v, N_real=N_real, N_proc=N_proc)
    pv = np.sqrt(2.0/np.pi)*((solver.m87/(Boltzmann*T))**(3.0/2.0)) * \
        Vs**2.0*np.exp(-solver.m87*Vs**2.0/(2.0*Boltzmann*T))
    dz = solver1.L/N_steps_z
    alpha = -np.log(0.6)/solver1.L
    for j in range(10):
        chihigh = np.sum(np.array(list(map(solver1.chi_analytical, Vs))).transpose(
            1, 2, 0)*dv*pv, axis=-1)
        chilow = np.sum(np.array(list(map(solver.chi_analytical, Vs))).transpose(
            1, 2, 0)*dv*pv, axis=-1)
        nhigh = np.sqrt(1+chihigh)
        nlow = np.sqrt(1+chilow)
        tab_analytical = np.real(nhigh-nlow)
        chihigh2 = np.sum(np.array(list(map(solver1.chi_analytical_2, Vs))).transpose(
            1, 2, 0)*dv*pv, axis=-1)
        chilow2 = np.sum(np.array(list(map(solver.chi_analytical_2, Vs))).transpose(
            1, 2, 0)*dv*pv, axis=-1)
        nhigh2 = np.sqrt(1+chihigh2)
        nlow2 = np.sqrt(1+chilow2)
        tab_analytical2 = np.real(nhigh2-nlow2)
        Dn_analytical2 += tab_analytical2/N_steps_z
        tab = np.load(
            f"{path}/Dn_w{i}_z{j}_murad_Thu Apr 28 10:07:55 2022.npy")  # gg paper
        # tab = np.load(
        #     f"{path}/Dn_w{i}_z{j}_murad_Thu Feb 10 19:13:04 2022.npy") # with colls
        Dn += tab/N_steps_z
        Dn[Dn > 0] = -1e-20
        Dn_c[i] += np.mean(tab[tab.shape[0]//2-avg_zone:tab.shape[0]//2+avg_zone,
                               tab.shape[1]//2-avg_zone:tab.shape[1]//2+avg_zone])/N_steps_z
        Dn_analytical_c[i] += np.mean(tab_analytical[tab_analytical.shape[0]//2-avg_zone:tab_analytical.shape[0]//2+avg_zone,
                                                     tab_analytical.shape[1]//2-avg_zone:tab_analytical.shape[1]//2+avg_zone])/N_steps_z
        Dn_analytical_c2[i] += np.mean(tab_analytical2[tab_analytical2.shape[0]//2-avg_zone:tab_analytical2.shape[0]//2+avg_zone,
                                                       tab_analytical2.shape[1]//2-avg_zone:tab_analytical2.shape[1]//2+avg_zone])/N_steps_z
        solver1.puiss *= np.exp(-alpha*dz)

    im = ax[i//5, i % 5].imshow(np.abs(Dn_analytical2), norm=colors.LogNorm(vmin=np.nanmin(np.abs(Dn_analytical2)),
                                                                            vmax=np.nanmax(np.abs(Dn_analytical2))))
    ax[i//5, i % 5].set_title(f"$w_0$ = {'{:.2e}'.format(waists_murad[i])} m")
    divider = make_axes_locatable(ax[i//5, i % 5])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    im2 = ax2[i//5, i % 5].imshow(np.abs(Dn), norm=colors.LogNorm(vmin=np.nanmin(np.abs(Dn_analytical2)),
                                                                  vmax=np.nanmax(np.abs(Dn_analytical2))))
    ax2[i//5, i % 5].set_title(f"$w_0$ = {'{:.2e}'.format(waists_murad[i])} m")
    divider2 = make_axes_locatable(ax2[i//5, i % 5])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig2.colorbar(im2, cax=cax2)
fig.suptitle("Analytical $\Delta n$")
fig2.suptitle("MC $\Delta n$")
alpha = -np.log(0.6)/L
Itilde = I*(1-np.exp(-alpha*L))/(alpha*L)
Itilde *= 1/(1+Itilde/I_sat_murad)
Dn_c_err = 0.028*Dn_c
Dn_murad = n2_murad*Itilde
Dn_murad_err = (Isat_err/I_sat_murad + n2_err/n2_murad)*Dn_murad
popt, cov = curve_fit(power_fit, waists_murad, np.abs(Dn_c))
popt_lin, cov_lin = curve_fit(
    lin_fit, np.log(waists_murad), np.log(np.abs(Dn_c)), sigma=np.log(np.abs(Dn_c_err)))
# fitted_comp = np.exp(lin_fit(np.log(waists_murad), popt_lin[0], popt_lin[1]))
fitted_comp = power_fit(waists_murad, popt[0], popt[1])
perr = np.sqrt(np.diag(cov))
perr_lin = np.sqrt(np.diag(cov_lin))
popt1, cov1 = curve_fit(power_fit, waists_murad, Dn_murad,
                        sigma=Dn_murad_err, maxfev=10000)
perr1 = np.sqrt(np.diag(cov1))
popt1_lin, cov1_lin = curve_fit(lin_fit, np.log(
    waists_murad), np.log(Dn_murad), sigma=np.log(Dn_murad_err))
fitted_exp = power_fit(waists_murad, popt1[0], popt1[1])
# fitted_exp = np.exp(lin_fit(np.log(waists_murad), popt1_lin[0], popt1_lin[1]))
popt2_lin, cov2_lin = curve_fit(lin_fit, np.log(
    waists_murad), np.log(np.abs(Dn_analytical_c2)))
popt3_lin, cov3_lin = curve_fit(lin_fit, np.log(
    waists_murad), np.log(np.abs(Dn_analytical_c)))
fitted_a = np.exp(lin_fit(np.log(waists_murad), popt2_lin[0], popt2_lin[1]))
perr1 = np.sqrt(np.diag(cov1))
perr1_lin = np.sqrt(np.diag(cov1_lin))
perr2_lin = np.sqrt(np.diag(cov2_lin))
perr3_lin = np.sqrt(np.diag(cov3_lin))
ax1.scatter(waists_murad, np.abs(Dn_c), label="Computed", color="tab:blue")
ax1.errorbar(waists_murad, np.abs(Dn_c), yerr=Dn_c_err, fmt='none',
             label="Computed", color="tab:blue", capsize=3.0)
ax1.scatter(waists_murad, Dn_murad,
            label="Exp", color="tab:orange")
ax1.errorbar(waists_murad, Dn_murad, yerr=Dn_murad_err, xerr=waists_err, fmt='none',
             label="Exp", color="tab:orange", capsize=3.0)
ax1.scatter(waists_murad, np.abs(Dn_analytical_c2),
            label="Analytical transit", color="tab:green")
ax1.scatter(waists_murad, np.abs(Dn_analytical_c),
            label="Analytical no transit", color="tab:red")
# ax1.plot(waists_murad, fitted_comp,
#          label=f"Power fit comp : p = {'{:.2f}'.format(popt[1])}"+"$\pm$ "+'{:.3f}'.format(
#              perr[0]),
#          color="tab:blue")
ax1.plot(waists_murad, fitted_comp,
         label=f"Lin fit comp : p = {'{:.2f}'.format(popt_lin[0])} "+"$\pm$ "+'{:.2f}'.format(
             perr_lin[0]),
         color="tab:blue")
ax1.fill_between(waists_murad, np.abs(Dn_c)+Dn_c_err,
                 np.abs(Dn_c)-Dn_c_err,
                 color='tab:blue', alpha=0.3)
# ax1.plot(waists_murad, fitted_exp,
#          label=f"Power fit exp : p = {'{:.2f}'.format(popt1[1])}"+"$\pm$ "+'{:.3f}'.format(
#              perr1[0]),
#          color="tab:orange")
ax1.plot(waists_murad, fitted_exp,
         label=f"Lin fit exp : p = {'{:.2f}'.format(popt1_lin[0])}"+"$\pm$ "+'{:.2f}'.format(
             perr1_lin[0]),
         color="tab:orange")
# ax1.fill_between(waists_murad, fitted_exp+popt1_lin[0]*fitted_exp*perr1_lin[0],
#                  fitted_exp-popt1_lin[0]*fitted_exp*perr1_lin[0],
#                  color="tab:orange", alpha=0.3)
ax1.fill_between(waists_murad, Dn_murad+Dn_murad_err,
                 Dn_murad-Dn_murad_err,
                 color="tab:orange", alpha=0.3)
# ax1.plot(waists_murad, fitted_a,
#          label=f"Lin fit analytical : p = {'{:.2f}'.format(popt2_lin[0])}"+"$\pm$ "+'{:.2f}'.format(perr2_lin[0]),
#          color="tab:green")
# ax1.fill_between(waists_murad, fitted_a+popt2_lin[0]*fitted_a*perr2_lin[0],
#          fitted_a-popt2_lin[0]*fitted_a*perr2_lin[0],
#          color="tab:green", alpha=0.3)
# ax1.plot(waists_murad, np.exp(lin_fit(np.log(waists_murad), popt3_lin[0], popt3_lin[1])),
#          label=f"Lin fit analytical : p = {'{:.2f}'.format(popt3_lin[0])}"+"$\pm$ "+'{:.2f}'.format(perr3_lin[0]),
#          color="tab:red")
# ax1.fill_between(waists_murad, np.exp(lin_fit(np.log(waists_murad), popt3_lin[0]-perr3_lin[0], popt3_lin[1])),
#          np.exp(lin_fit(np.log(waists_murad), popt3_lin[0]+perr3_lin[0], popt3_lin[1])),
#          color="tab:red", alpha=0.3)
ax1.set_xlabel("Beam waist $w_0$ (m)", fontsize=16)
ax1.set_ylabel("$\Delta n$", fontsize=16)
# ax1.set_xscale('log')
# ax1.set_yscale('log')
ax1.legend()
plt.xticks(fontsize=20)
fig1.suptitle(
    "Non linear index $\Delta n$ against beam waist size", fontsize=20)


# treat power run
fig2, ax2 = plt.subplots(2, 5)
fig3, ax3 = plt.subplots()
idx = 8
dataset = "Wed May  4 14:00:37 2022"
Dn_center = np.load(f"results/Dn_center_w_P_{dataset}.npy")
Dn_analytical = np.load(
    f"results/Dn_center_w_P_{dataset}_analytical.npy")
for counter_p, power in enumerate(powers):
    Dn_P_murad[counter_p, :, :] = np.load(
        f'results/Dn_w{counter_p}_murad_{dataset}.npy')
for counter_p, power in enumerate(powers):
    im = ax2[counter_p//5, counter_p % 5].imshow(np.abs(Dn_P_murad[counter_p, :, :]),
                                                 vmin=np.nanmin(
        np.abs(Dn_P_murad)),
        vmax=np.nanmax(np.abs(Dn_P_murad)))
    ax2[counter_p//5, counter_p % 5].set_title("$P_{0}$ = " +
                                               f"{np.round(power*1e3, decimals=2)} mW")
    fig2.colorbar(im, ax=ax[counter_p//5, counter_p % 5])


def fit_Isat(P, n2_0, Isat):
    I = P/(np.pi*waists_murad[idx]**2)
    alpha = -np.log(0.6)/L
    Itilde = I*(1-np.exp(-alpha*L))/(alpha*L)
    Itilde *= 1/(1+Itilde/Isat)
    return n2_0*Itilde


Dn_murad = np.abs(fit_Isat(powers, n2_murad[idx], I_sat_murad[idx]))
Dn_murad_err = (Isat_err[idx]/I_sat_murad[idx] +
                n2_err[idx]/n2_murad[idx])*Dn_murad
popt, pcov = curve_fit(fit_Isat, powers, Dn_center,
                       p0=[Dn_center[0]/I, 1.0], maxfev=16000)
popt1, pcov1 = curve_fit(fit_Isat, powers, Dn_analytical,
                         p0=[Dn_center[0]/I, 1.0], maxfev=16000)
print(popt)
leg1 = "Fit : $n_{2}$ = "+"{:.2e}".format(popt[0])+" $m^{2}/W$, " +\
    "$I_{sat}$ = "+f"{popt[1]:.2e} "+"$W/m^{2}$"
leg2 = "Data : $n_{2}$ = "+"{:.2e}".format(-n2_murad[idx])+" $m^{2}/W$, " +\
    "$I_{sat}$ = "+f"{I_sat_murad[idx]:.2e} "+"$W/m^{2}$"
leg3 = "Analytical : $n_{2}$ = "+"{:.2e}".format(popt1[0])+" $m^{2}/W$, " +\
    "$I_{sat}$ = "+f"{popt1[1]:.2e} "+"$W/m^{2}$"
ax3.plot(powers*1e3, np.abs(fit_Isat(powers, popt[0], popt[1])),
         label=leg1, color='tab:blue')
ax3.errorbar(powers*1e3, Dn_murad, yerr=Dn_murad_err,
             label=leg2, color='tab:orange', marker='o', linestyle='None', capsize=3.0)
ax3.fill_between(powers*1e3, Dn_murad+Dn_murad_err,
                 Dn_murad-Dn_murad_err,
                 color="tab:orange", alpha=0.3)
ax3.scatter(powers*1e3, np.abs(Dn_analytical), label=leg3, color='tab:green')
ax3.errorbar(powers*1e3, np.abs(Dn_center), yerr=0.028*np.abs(Dn_center),
             label="Computed", color='tab:blue', marker='o', linestyle='None', capsize=3.0)
ax3.fill_between(powers*1e3, 1.028*np.abs(Dn_center),
                 0.972*np.abs(Dn_center),
                 color="tab:blue", alpha=0.3)
ax3.legend()
ax3.set_title("$\Delta n$ vs power, $w_{0}$ = " +
              f"{waists_murad[idx]*1e3:.2f} mm")
fig2.suptitle("$\Delta n$ vs power, $w_{0}$ = " +
              f"{waists_murad[idx]*1e3:.2f} mm")
ax3.set_xlabel("Power in mW")
ax3.set_ylabel("$\Delta n$")
# ax1.set_yscale('log')
plt.show()
