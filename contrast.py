# -*-coding:utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, uniform_filter1d
from skimage.restoration import unwrap_phase
from skimage import filters, measure, morphology
from skimage.segmentation import clear_border, flood
import tools
from azim_avg import azimuthalAverage as az_avg
from scipy.optimize import curve_fit


def cache(radius, center=np.array([1024, 1024]), out=True, nb_pix=2048):

    Y, X = np.ogrid[:nb_pix, :nb_pix]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    if out:
        mask = dist_from_center <= radius
    else:
        mask = dist_from_center > radius

    return mask


def cache_2(x, y, center=np.array([1024, 1024]), out=True, nb_pix=2048):

    Y, X = np.ogrid[:nb_pix, :nb_pix]
    dist_from_center_x = np.abs(X - center[0])
    dist_from_center_y = np.abs(Y - center[1])

    if out:
        mask_x = dist_from_center_x <= x
        mask_y = dist_from_center_y <= y

        return mask_x & mask_y

    else:
        mask_x = np.where(dist_from_center_x > x, 1, 0)
        mask_y = np.where(dist_from_center_y > y, 1, 0)

        return np.where(mask_x + mask_y > 0, True, False)


def maximum(im):
    im_fft = np.fft.fftshift(np.fft.fft2(im))

    # freq = np.sum(np.abs(im_fft), axis=1)
    # freq = np.sum(np.abs(im_fft), axis=0)

    freq = np.abs(im_fft)[im.shape[0]//2, :]
    freq = freq/np.max(freq)

    maxi = []
    prom = 0.2
    while len(maxi) != 3:
        maxi = find_peaks(freq, distance=10, prominence=prom)[0]

        prom *= 0.9
        if prom <= 0.005:
            raise Exception("No fringe detected")

    return list(maxi)


def im_osc(im: np.ndarray,  cont: bool = True, plot: bool = False):
    """Separates the continuous and oscillating components of an image using
    Fourier filtering.

    :param np.ndarray im: Description of parameter `im`.
    :param bool cont: Returns or not the continuons component
    :param bool plot: Plots a visualization of the analysis result
    :return np.ndarray: The oscillating component of the image, or both
    components

    """

    nb_pix = len(im[0, :])
    im_fft = np.fft.fftshift(np.fft.fft2(im))
    im_fft_fringe = im_fft.copy()
    im_fft_cont = im_fft.copy()
    fft_filt = gaussian_filter(np.abs(im_fft), 1e-3*im_fft.shape[0])
    cont_size = 20
    mask_cont_flood = cache(cont_size, out=False, center=[nb_pix//2, nb_pix//2],
                            nb_pix=nb_pix)
    dbl_gradient = np.log(np.abs(np.gradient(fft_filt, axis=0)) +
                          np.abs(np.gradient(fft_filt, axis=1)))
    m_value = np.nanmean(dbl_gradient[dbl_gradient != -np.infty])
    dbl_gradient[np.bitwise_not(mask_cont_flood)] = m_value
    dbl_gradient_int = (dbl_gradient*(dbl_gradient > 0.7 *
                        np.nanmax(dbl_gradient))).astype(np.uint8)
    threshold = filters.threshold_otsu(dbl_gradient_int)
    mask = dbl_gradient_int > threshold
    mask = morphology.remove_small_objects(mask, 1)
    mask = morphology.remove_small_holes(mask, 1)
    mask = clear_border(mask)
    mask = morphology.remove_small_holes(mask, 1, connectivity=2)
    labels = measure.label(mask)
    props = measure.regionprops(labels, dbl_gradient_int)
    # takes the spot with the maximum area
    areas = [prop.area for prop in props]
    maxi_area = np.where(areas == max(areas))[0][0]
    label_osc = props[maxi_area].label
    # labels_cont = props[1].label
    contour_osc = measure.find_contours(labels == label_osc, 0.5)[0]
    # contour_cont = measure.find_contours(labels == labels_cont, 0.5)[0]
    y, x = contour_osc.T
    # y1, x1 = contour_cont.T
    y = y.astype(int)
    x = x.astype(int)
    mask_osc = np.zeros(im_fft.shape)
    mask_osc[y, x] = 1
    mask_osc_flood = flood(mask_osc, (y[0]+1, x[0]+1), connectivity=1)
    im_fft_fringe[mask_osc_flood] = 0
    im_fft_cont[mask_cont_flood] = 0
    im_fringe = np.fft.ifft2(np.fft.fftshift(im_fft_fringe))
    im_cont = np.fft.ifft2(np.fft.fftshift(im_fft_cont))
    if plot:
        circle = plt.Circle((nb_pix//2, nb_pix//2), cont_size//2, color='b',
                            fill=False)
        fig, ax = plt.subplots(1, 4)
        im0 = ax[0].imshow(im, cmap='gray')
        ax[0].set_xlim((712, 1336))
        ax[0].set_ylim((712, 1336))
        ax[0].set_title("Real space")
        fig.colorbar(im0, ax=ax[0])
        # im = ax[1].imshow(np.log(np.abs(im_fft)))
        im = ax[1].imshow(np.abs(im_fft),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                          vmin=np.min(np.abs(im_fft)),
                          vmax=np.max(np.abs(im_fft)),
                          base=10))
        fig.colorbar(im, ax=ax[1])
        ax[1].plot(x, y, color='r', ls='--')
        ax[1].add_patch(circle)
        ax[1].set_xlim((912, 1136))
        ax[1].set_ylim((912, 1136))
        ax[1].set_title("Fourier space")
        ax[1].legend(["Oscillating", "Continuous"])
        im = ax[2].imshow(np.abs(im_fft_fringe),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                          vmin=np.min(np.abs(im_fft)),
                          vmax=np.max(np.abs(im_fft)),
                          base=10))
        fig.colorbar(im, ax=ax[2])
        ax[2].set_xlim((912, 1136))
        ax[2].set_ylim((912, 1136))
        ax[2].set_title("Filtered Fourier signal")
        im = ax[3].imshow(np.angle(im_fringe), cmap="twilight")
        fig.colorbar(im, ax=ax[3])
        ax[3].set_xlim((712, 1336))
        ax[3].set_ylim((712, 1336))
        ax[3].set_title("Phase of filtered signal")
        plt.show()
    if cont:
        return im_cont, im_fringe

    return im_fringe


def delta_n(im0: np.ndarray,  cont: bool = False, plot: bool = False, err: bool = False):
    """Delta n

    :param np.ndarray im: Image to extract Dn
    :param bool cont: Returns or not the continuons component
    :param bool plot: Plots a visualization of the analysis result
    :param bool err: Returns the error 
    :return np.ndarray: The oscillating component of the image, or both
    components with or without the error on Dn

    """
    #objective function for Dn vs I fitting
    def fit_phi_vs_I(I, n2, Isat, offset):
        phi = n2*(I/(1+(I/Isat)))+offset
        return phi
    im = np.copy(im0)
    im = im - np.mean(im[0:100, 0:100])
    im = im/np.max(im)
    Y, X = np.indices(im.shape)
    centre_x, centre_y = tools.centre(im)
    im_avg = az_avg(im, center=[centre_x, centre_y])
    rad_max = np.argmin(np.where(im_avg < 0.07, -1, 0))
    nb_pix = im.shape[0]
    im_fft = np.fft.fftshift(np.fft.fft2(im))
    im_fft_fringe = im_fft.copy()
    im_fft_cont = im_fft.copy()
    fft_filt = gaussian_filter(np.abs(im_fft), 1e-3*im_fft.shape[0])
    cont_size = 20
    mask_cont_flood = cache(cont_size, out=False, center=[nb_pix//2, nb_pix//2],
                            nb_pix=nb_pix)
    dbl_gradient = np.log(np.abs(np.gradient(fft_filt, axis=0)) +
                          np.abs(np.gradient(fft_filt, axis=1)))
    m_value = np.nanmean(dbl_gradient[dbl_gradient != -np.infty])
    dbl_gradient[np.bitwise_not(mask_cont_flood)] = m_value
    dbl_gradient_int = (dbl_gradient*(dbl_gradient > 0.7 *
                        np.nanmax(dbl_gradient))).astype(np.uint8)
    # convert to binary using appropriate threshold
    threshold = filters.threshold_otsu(dbl_gradient_int)
    mask = dbl_gradient_int > threshold
    mask = morphology.remove_small_objects(mask, 1)
    mask = morphology.remove_small_holes(mask, 1)
    mask = clear_border(mask)
    mask = morphology.remove_small_holes(mask, 1, connectivity=2)
    labels = measure.label(mask)
    props = measure.regionprops(labels, dbl_gradient_int)
    # takes the spot with the maximum area
    areas = [prop.area for prop in props]
    maxi_area = np.where(areas == max(areas))[0][0]
    label_osc = props[maxi_area].label
    center_osc = np.round(props[maxi_area].centroid_weighted).astype(int)
    # labels_cont = props[1].label
    contour_osc = measure.find_contours(labels == label_osc, 0.5)[0]
    # contour_cont = measure.find_contours(labels == labels_cont, 0.5)[0]
    y, x = contour_osc.T
    # y1, x1 = contour_cont.T
    y = y.astype(int)
    x = x.astype(int)
    mask_osc_flood = np.zeros(im_fft.shape)
    mask_osc_flood[y, x] = 1
    mask_osc_flood = flood(mask_osc_flood, (y[0]+1, x[0]+1), connectivity=1)
    # r_osc = int(np.max(np.sqrt((y-center_osc[0])**2 + (x-center_osc[1])**2)))
    # mask_osc_flood = cache(int(1.2*r_osc), center=np.flip(center_osc), out=False, nb_pix=im.shape[0])
    im_fft_fringe[mask_osc_flood] = 0
    im_fft_cont[mask_cont_flood] = 0
    # bring osc part to center to remove tilt
    im_fft_fringe = np.roll(im_fft_fringe,
                            (im_fft_fringe.shape[0]//2-center_osc[0],
                             im_fft_fringe.shape[1]//2-center_osc[1]),
                            axis=(-2, -1))
    im_fringe = np.fft.ifft2(np.fft.fftshift(im_fft_fringe))
    im_cont = np.fft.ifft2(np.fft.fftshift(im_fft_cont))
    centre_x, centre_y = tools.centre(im_cont)
    cont_avg = az_avg(np.abs(im_cont), center=[centre_x, centre_y])
    if plot:
        circle = plt.Circle((nb_pix//2, nb_pix//2), cont_size//2, color='b',
                            fill=False)
        fig, ax = plt.subplots(1, 4)
        i0 = ax[0].imshow(im, cmap='gray')
        # ax[0].set_xlim((712, 1336))
        # ax[0].set_ylim((712, 1336))
        ax[0].set_title("Real space")
        fig.colorbar(i0, ax=ax[0])
        # im = ax[1].imshow(np.log(np.abs(im_fft)))
        i = ax[1].imshow(np.abs(im_fft),
                         norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                vmin=np.min(np.abs(im_fft)),
                                                vmax=np.max(np.abs(im_fft)),
                                                base=10))
        fig.colorbar(i, ax=ax[1])
        ax[1].plot(x, y, color='r', ls='--')
        ax[1].add_patch(circle)
        # ax[1].set_xlim((912, 1136))
        # ax[1].set_ylim((912, 1136))
        ax[1].set_title("Fourier space")
        ax[1].legend(["Oscillating", "Continuous"])
        i = ax[2].imshow(np.abs(im_fft_fringe),
                         norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                vmin=np.min(np.abs(im_fft)),
                                                vmax=np.max(np.abs(im_fft)),
                                                base=10))
        fig.colorbar(i, ax=ax[2])
        # ax[2].set_xlim((912, 1136))
        # ax[2].set_ylim((912, 1136))
        ax[2].set_title("Filtered Fourier signal")
        i = ax[3].imshow(np.angle(im_fringe), cmap="twilight")
        fig.colorbar(i, ax=ax[3])
        # ax[3].set_xlim((712, 1336))
        # ax[3].set_ylim((712, 1336))
        ax[3].set_title("Phase of filtered signal")
        plt.show()
    phase = np.angle(im_fringe)
    analytic = np.abs(im_fringe)
    contr = 2*analytic/im_cont
    threshold_contrast = 0.2
    mask = np.where(contr > threshold_contrast, 1, 0)*np.where(im > 0.02, 1, 0)
    phase = unwrap_phase(phase)
    phi_flat = phase.flatten()
    cont_flat = im_cont.flatten()
    order = np.argsort(cont_flat)
    cont_flat_order = cont_flat[order]
    phi_flat_order = phi_flat[order]
    phi_flat_filt = uniform_filter1d(phi_flat_order, 16000)
    # phi_flat_filt = gaussian_filter1d(phi_flat_order, 16000, truncate=2)
    cont_flat_fit = cont_flat_order[np.linspace(
        0, len(cont_flat)-1, 4000, dtype=int)]
    phi_flat_fit = phi_flat_order[np.linspace(
        0, len(phi_flat)-1, 4000, dtype=int)]
    y, x = np.indices(im.shape)
    r = np.hypot(x - centre_x, y - centre_y)
    phi_avg = az_avg(phase, center=[centre_x, centre_y])
    try:
        popt, pcov = curve_fit(fit_phi_vs_I, cont_flat_fit,
                               phi_flat_fit, maxfev=3200, bounds=[(-np.inf, -np.inf, -np.inf),
                                                                 (np.inf, 1e5, np.inf)])
        # gets fitting covariance/error for each parameter
        perr = np.sqrt(np.diag(pcov))
        phase_tot = np.abs(fit_phi_vs_I(np.nanmax(cont_flat),
                           popt[0], popt[1], popt[2])-popt[-1])
        # phase_tot = np.abs(phi_flat_fit[-1]-phi_flat_fit[0])
        # phase_tot = np.abs(phi_flat_filt[-1]-popt[-1])
        # computes dominating error
        phi_err = max([np.abs(fit_phi_vs_I(np.nanmax(cont_flat), popt[0]-perr[0], popt[1], popt[2]) -
                              fit_phi_vs_I(np.nanmax(cont_flat), popt[0]+perr[0], popt[1], popt[2])),
                       np.abs(fit_phi_vs_I(np.nanmax(cont_flat), popt[0], popt[1]-perr[1], popt[2]) -
                              fit_phi_vs_I(np.nanmax(cont_flat), popt[0], popt[1]+perr[1], popt[2])),
                       np.abs(fit_phi_vs_I(np.nanmax(cont_flat), popt[0], popt[1], popt[2]-perr[2]) -
                              fit_phi_vs_I(np.nanmax(cont_flat), popt[0], popt[1], popt[2]+perr[2]))])
        fitted = True
    except Exception:
        phase_tot = np.abs(phi_flat_fit[-1]-phi_flat_fit[0])
        phi_err = 0.05*phase_tot
        fitted = False
    if plot:
        fig, ax = plt.subplots(1, 5)
        # i0 = ax[0].imshow(im0, cmap='gray')
        # ax[0].set_title("Raw interferogram")
        # fig.colorbar(i0, ax=ax[0])
        i0 = ax[0].imshow(np.abs(im_cont))
        ax[0].set_title("Dc intensity")
        fig.colorbar(i0, ax=ax[0])
        # ax[1] = plt.subplot(142, projection='3d')
        # i1 = ax[1].plot_surface(X, Y, phase, vmin=np.nanmin(-phase[mask == 1]),
        #                         vmax=np.nanmax(-phase[mask == 1]), cmap='viridis')
        i1 = ax[1].imshow(phase, vmin=np.nanmin(phase[mask == 1]),
                                vmax=np.nanmax(phase[mask == 1]), cmap='viridis')
        ax[1].set_title("Unwrapped phase")
        # ax[0].set_xlim((712, 1336))
        # ax[0].set_ylim((712, 1336))
        fig.colorbar(i1, ax=ax[1])
        ax[2].plot(cont_avg, phi_avg, label="Unwrapped phase")
        if fitted:
            ax[2].plot(cont_avg, fit_phi_vs_I(cont_avg, popt[0], popt[1], popt[2]),
                       label="Fit")
            ax[2].hlines(popt[-1], np.min(cont_avg), np.max(cont_avg),
                         label='Phase reference', linestyles='dashed', color='red')
        ax[2].plot(cont_avg, gaussian_filter(phi_avg, 10),
                   label="Unwrapped phase filtered")
        ax[2].set_title("Azimuthal average")
        ax[2].set_xlabel("Az avg intensity")
        ax[2].set_ylabel("Phase in rad")
        # ax[2].set_xlim((0, 1024))
        ax[2].legend()
        ax[3].plot(cont_flat_order[0:-1:800], phi_flat_filt[0:-1:800],
                   label="Flattened phase filtered")
        ax[3].plot(cont_flat_order[0:-1:800],
                   phi_flat_order[0:-1:800],
                   label="Flattened phase", color='grey', alpha=0.5)
        if fitted:
            ax[3].plot(cont_flat, fit_phi_vs_I(cont_flat, popt[0], popt[1], popt[2]),
                       label="Fit")
            ax[3].hlines(popt[-1], np.min(cont_avg), np.max(cont_avg),
                         label='Phase reference', linestyles='dashed', color='red')
        ax[3].set_title("Azimuthal slice")
        ax[3].set_ylabel("Phase")
        ax[3].set_xlabel("Intensity")
        # ax[3].set_xlim((0, 1024))
        ax[3].legend()
        # ax[3].set_xscale('log')
        i2 = ax[4].imshow(fit_phi_vs_I(np.abs(im_cont), popt[0], popt[1], popt[2]), 
                          vmin=np.nanmin(phase[mask == 1]), vmax=np.nanmax(phase[mask == 1]))
        fig.colorbar(i2, ax=ax[4])
        ax[4].set_title("Fitted phase 2D")
        plt.show()
    if cont:
        if err:
            return im_cont, im_fringe, (phase_tot, phi_err)
        return im_cont, im_fringe, phase_tot
    if err:
        return phase_tot, phi_err

    return phase_tot


def contr(im, maxi=None):

    im_cont, im_fringe = im_osc(im, maxi=maxi)
    analytic = np.abs(im_fringe)

    return 2*analytic/im_cont


def phase(im, maxi=None):

    im_fringe = im_osc(im, maxi=maxi, cont=False)
    im_phase = unwrap_phase(np.angle(im_fringe))

    return im_phase


if __name__ == "__main__":

    path = "murad/Test2"

    im_j = tools.open_tif(path, 'Im2_180')
    im_j = im_j/np.max(im_j)

    maxi = maximum(im_j)
    phase(im_j, maxi)
