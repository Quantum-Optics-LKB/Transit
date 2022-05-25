# -*-coding:utf-8 -*

from calendar import c
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, uniform_filter1d, label
from skimage.restoration import unwrap_phase
from skimage import filters, measure, morphology
from skimage.segmentation import clear_border, flood
from azim_avg import az_avg
from scipy.optimize import curve_fit


def gauss_fit(x, maxi, std, rtd):
    return maxi*np.exp(-(x-rtd)**2/std**2)


def centre(im):
    """Fits the center of the image using gaussian fitting

    Args:
        im (np.ndarray): The image to fit

    Returns:
        Tuple(int): The coordinates of the fitted center.
    """
    out_x = np.sum(im, axis=0)
    out_x = out_x/np.max(out_x)
    out_y = np.sum(im, axis=1)
    out_y = out_y/np.max(out_y)

    absc = np.linspace(0, im.shape[1]-1, im.shape[1])
    ordo = np.linspace(0, im.shape[0]-1, im.shape[0])
    ptot, pcov = curve_fit(gauss_fit, absc, out_x, p0=[1, 100, 1000])
    centre_x = int(ptot[2])
    ptot, pcov = curve_fit(gauss_fit, ordo, out_y, p0=[1, 100, 1000])
    centre_y = int(ptot[2])
    return centre_x, centre_y


def cache(radius: int, center: tuple = (1024, 1024), out: bool = True,
          nb_pix: tuple = (2048, 2048)) -> np.ndarray:
    """Defines a circular mask

    Args:
        radius (int): Radius of the mask
        center (tuple, optional): Center of the mask. Defaults to (1024, 1024).
        out (bool, optional): Masks the outside of the disk. Defaults to True.
        nb_pix (tuple, optional): Shape of the mask. Defaults to (2048, 2048).

    Returns:
        np.ndarray: The array of booleans defining the mask
    """
    Y, X = np.ogrid[:nb_pix[0], :nb_pix[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    if out:
        mask = dist_from_center <= radius
    else:
        mask = dist_from_center > radius

    return mask


def im_osc(im: np.ndarray,  cont: bool = False, plot: bool = False, return_mask: bool = False, big: bool = False) -> tuple:
    """Separates the continuous and oscillating components of an image using
    Fourier filtering.

    :param np.ndarray im: Description of parameter `im`.
    :param bool cont: Returns or not the continuons component
    :param bool plot: Plots a visualization of the analysis result
    :return np.ndarray: The oscillating component of the image, or both
    components

    """

    im_fft = np.fft.fftshift(np.fft.fft2(im))
    im_fft_fringe = im_fft.copy()
    im_fft_cont = im_fft.copy()
    fft_filt = gaussian_filter(np.abs(im_fft), 1e-3*im_fft.shape[0])
    cont_size = 80
    mask_cont_flood = cache(cont_size, out=False, center=(im.shape[0]//2, im.shape[1]//2),
                            nb_pix=im.shape)
    dbl_gradient = np.log(np.abs(np.gradient(fft_filt, axis=0)) +
                          np.abs(np.gradient(fft_filt, axis=1)))
    m_value = np.nanmean(dbl_gradient[dbl_gradient != -np.infty])
    dbl_gradient[np.bitwise_not(mask_cont_flood)] = m_value
    dbl_gradient_int = (dbl_gradient*(dbl_gradient > 0.8 *
                                      np.nanmax(dbl_gradient)))
    dbl_gradient_int /= np.nanmax(dbl_gradient_int)
    dbl_gradient_int = (255*dbl_gradient_int).astype(np.uint8)
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
    center_osc = np.round(props[maxi_area].centroid).astype(int)
    contour_osc = measure.find_contours(labels == label_osc, 0.5)[0]
    y, x = contour_osc.T
    y = y.astype(int)
    x = x.astype(int)
    mask_osc = np.zeros(im_fft.shape)
    mask_osc[y, x] = 1
    mask_osc_flood = flood(mask_osc, (y[0]+1, x[0]+1), connectivity=1)
    if big:
        r_osc = min(center_osc)
        # r_osc = 5.3*np.max([[np.hypot(x[i]-x[j], y[i]-y[j])
        #                      for j in range(len(x))] for i in range(len(x))])
        mask_osc_flood = cache(r_osc, out=False, center=(
            center_osc[1], center_osc[0]))
    im_fft_fringe[mask_osc_flood] = 0
    im_fft_cont[mask_cont_flood] = 0
    # bring osc part to center to remove tilt
    im_fft_fringe = np.roll(im_fft_fringe,
                            (im_fft_fringe.shape[0]//2-center_osc[0],
                             im_fft_fringe.shape[1]//2-center_osc[1]),
                            axis=(-2, -1))
    im_fringe = np.fft.ifft2(np.fft.fftshift(im_fft_fringe))
    im_cont = np.fft.ifft2(np.fft.fftshift(im_fft_cont))
    if plot:
        circle = plt.Circle((im.shape[1]//2, im.shape[0]//2), cont_size//2, color='b',
                            fill=False)
        fig, ax = plt.subplots(1, 4)
        im0 = ax[0].imshow(im, cmap='gray')
        ax[0].set_title("Real space")
        fig.colorbar(im0, ax=ax[0])
        im = ax[1].imshow(np.abs(im_fft),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        fig.colorbar(im, ax=ax[1])
        ax[1].plot(x, y, color='r', ls='--')
        ax[1].add_patch(circle)
        if big:
            circle_big = plt.Circle((center_osc[1], center_osc[0]), r_osc, color='r',
                                    fill=False)
            ax[1].add_patch(circle_big)

        ax[1].set_title("Fourier space")
        ax[1].legend(["Oscillating", "Continuous"])
        im = ax[2].imshow(np.abs(im_fft_fringe),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title("Filtered Fourier signal")
        im = ax[3].imshow(np.angle(im_fringe), cmap="twilight")
        fig.colorbar(im, ax=ax[3])
        ax[3].set_title("Phase of filtered signal")
        plt.show()
    if cont:
        if return_mask:
            return im_cont, im_fringe, mask_cont_flood, mask_osc_flood, center_osc
        return im_cont, im_fringe
    if return_mask:
        return im_fringe, mask_cont_flood, mask_osc_flood, center_osc
    return im_fringe


def im_osc_center(im: np.ndarray, center: tuple, mask_osc_flood: np.ndarray = None,  cont: bool = False, plot: bool = False, big: bool = False) -> tuple:
    """Separates the continuous and oscillating components of an image using
    Fourier filtering.

    :param np.ndarray im: Description of parameter `im`.
    :param tuple center: i,j position of the 1st order
    :param np.ndarray mask_osc_flood: mask for the 1st order
    :param bool cont: Returns or not the continuons component
    :param bool plot: Plots a visualization of the analysis result
    :return np.ndarray: The oscillating component of the image, or both
    components

    """

    im_fft = np.fft.fftshift(np.fft.fft2(im))
    im_fft_fringe = im_fft.copy()
    im_fft_cont = im_fft.copy()
    fft_filt = gaussian_filter(np.abs(im_fft), 1e-3*im_fft.shape[0])
    cont_size = 20
    mask_cont_flood = cache(cont_size, out=False, center=(im.shape[0]//2, im.shape[1]//2),
                            nb_pix=im.shape)
    if mask_osc_flood is None:
        dbl_gradient = np.log(np.abs(np.gradient(fft_filt, axis=0)) +
                              np.abs(np.gradient(fft_filt, axis=1)))
        m_value = np.nanmean(dbl_gradient[dbl_gradient != -np.infty])
        dbl_gradient[np.bitwise_not(mask_cont_flood)] = m_value
        dbl_gradient_int = (dbl_gradient*(dbl_gradient > 0.8 *
                                          np.nanmax(dbl_gradient)))
        dbl_gradient_int /= np.nanmax(dbl_gradient_int)
        dbl_gradient_int = (255*dbl_gradient_int).astype(np.uint8)
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
        center_osc = center
        contour_osc = measure.find_contours(labels == label_osc, 0.5)[0]
        y, x = contour_osc.T
        y = y.astype(int)
        x = x.astype(int)
        mask_osc = np.zeros(im_fft.shape)
        mask_osc[y, x] = 1
        mask_osc_flood = flood(mask_osc, (y[0]+1, x[0]+1), connectivity=1)
        if big:
            r_osc = np.max([[np.hypot(x[i]-x[j], y[i]-y[j])
                             for j in range(len(x))] for i in range(len(x))])
            mask_osc_flood = cache(r_osc, out=False, center=(
                center, center))
            # mask_osc_flood = np.zeros(mask_cont_flood.shape, dtype=bool)
            # mask_osc_flood[0:mask_osc_flood.shape[0] //
            #                2, 0:mask_osc_flood.shape[1]//2] = True
            # mask_osc_flood = np.logical_not(np.logical_and(
            #     mask_osc_flood, mask_cont_flood))
    im_fft_fringe[mask_osc_flood] = 0
    im_fft_cont[mask_cont_flood] = 0
    # bring osc part to center to remove tilt
    im_fft_fringe = np.roll(im_fft_fringe,
                            (im_fft_fringe.shape[0]//2-center[0],
                             im_fft_fringe.shape[1]//2-center[1]),
                            axis=(-2, -1))
    im_fringe = np.fft.ifft2(np.fft.fftshift(im_fft_fringe))
    im_cont = np.fft.ifft2(np.fft.fftshift(im_fft_cont))
    if plot:
        circle = plt.Circle((im.shape[1]//2, im.shape[0]//2), cont_size//2, color='b',
                            fill=False)
        fig, ax = plt.subplots(1, 4)
        im0 = ax[0].imshow(im, cmap='gray')
        ax[0].set_title("Real space")
        fig.colorbar(im0, ax=ax[0])
        im = ax[1].imshow(np.abs(im_fft),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        fig.colorbar(im, ax=ax[1])
        if mask_osc_flood is None:
            ax[1].plot(x, y, color='r', ls='--')
        else:
            ax[1].imshow(mask_osc_flood, alpha=0.35, cmap='gray')
        ax[1].add_patch(circle)
        if big and mask_osc_flood is None:
            circle_big = plt.Circle((center[1], center[0]), r_osc, color='r',
                                    fill=False)
            ax[1].add_patch(circle_big)

        ax[1].set_title("Fourier space")
        ax[1].legend(["Oscillating", "Continuous"])
        im = ax[2].imshow(np.abs(im_fft_fringe),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title("Filtered Fourier signal")
        im = ax[3].imshow(np.angle(im_fringe), cmap="twilight")
        fig.colorbar(im, ax=ax[3])
        ax[3].set_title("Phase of filtered signal")
        plt.show()
    if cont:
        return im_cont, im_fringe
    return im_fringe


def im_osc_mask(im: np.ndarray, masks: tuple,  cont: bool = True, plot: bool = False) -> tuple:
    """Separates the continuous and oscillating components of an image using
    Fourier filtering.

    :param np.ndarray im: Description of parameter `im`.
    :param tuple masks: Continuous and oscillating masks
    :param bool cont: Returns or not the continuons component
    :param bool plot: Plots a visualization of the analysis result
    :return np.ndarray: The oscillating component of the image, or both
    components

    """
    mask_cont_flood, mask_osc_flood, center_osc = masks
    center_osc[0] = int(center_osc[0])
    center_osc[1] = int(center_osc[1])
    im_fft = np.fft.fftshift(np.fft.fft2(im))
    im_fft_fringe = im_fft.copy()
    im_fft_cont = im_fft.copy()
    im_fft_fringe[mask_osc_flood] = 0
    im_fft_cont[mask_cont_flood] = 0
    # bring osc part to center to remove tilt
    im_fft_fringe = np.roll(im_fft_fringe,
                            (im_fft_fringe.shape[0]//2-center_osc[0],
                             im_fft_fringe.shape[1]//2-center_osc[1]),
                            axis=(-2, -1))
    im_fringe = np.fft.ifft2(np.fft.fftshift(im_fft_fringe))
    im_cont = np.fft.ifft2(np.fft.fftshift(im_fft_cont))
    if plot:
        fig, ax = plt.subplots(1, 4)
        im0 = ax[0].imshow(im, cmap='gray')
        ax[0].set_title("Real space")
        fig.colorbar(im0, ax=ax[0])
        im = ax[1].imshow(np.abs(im_fft),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        ax[1].scatter(center_osc[1], center_osc[0], color='red')
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("Fourier space")
        im = ax[2].imshow(np.abs(im_fft_fringe),
                          norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                 vmin=np.min(np.abs(im_fft)),
                                                 vmax=np.max(np.abs(im_fft)),
                                                 base=10))
        fig.colorbar(im, ax=ax[2])
        ax[2].set_title("Filtered Fourier signal")
        im = ax[3].imshow(np.angle(im_fringe), cmap="twilight")
        fig.colorbar(im, ax=ax[3])
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
    # objective function for Dn vs I fitting
    def fit_phi_vs_I(I, n2, Isat, offset):
        phi = n2*(I/(1+(I/Isat)))+offset
        return phi
    im = np.copy(im0)
    im = im - np.mean(im[0:100, 0:100])
    im = im/np.max(im)
    centre_x, centre_y = centre(im)
    im_cont, im_fringe = im_osc(im, cont=True, plot=plot)
    centre_x, centre_y = centre(im_cont)
    cont_avg = az_avg(np.abs(im_cont), center=[centre_x, centre_y])
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
        i0 = ax[0].imshow(np.abs(im_cont))
        ax[0].set_title("Dc intensity")
        fig.colorbar(i0, ax=ax[0])
        i1 = ax[1].imshow(phase, vmin=np.nanmin(phase[mask == 1]),
                          vmax=np.nanmax(phase[mask == 1]), cmap='viridis')
        ax[1].set_title("Unwrapped phase")
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
        ax[3].legend()
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


def contr(im: np.ndarray) -> np.ndarray:
    """Computes the contrast of an interferogram

    Args:
        im (np.ndarray): The interferogram

    Returns:
        np.ndarray: The contrast map
    """
    im_cont, im_fringe = im_osc(im, cont=True)
    analytic = np.abs(im_fringe)
    cont = np.abs(im_cont)
    return 2*analytic/cont


def phase(im: np.ndarray, plot: bool = False, masks: tuple = None, big: bool = False) -> np.ndarray:
    """Returns the phase from an interfogram

    Args:
        im (np.ndarray): The interferogram
        plot (bool) : whether to plot something

    Returns:
        np.ndarray: The unwrapped phase
    """
    if masks is not None:
        im_fringe = im_osc_mask(im, masks, cont=False, plot=plot, big=big)
    else:
        im_fringe = im_osc(im, cont=False, plot=plot, big=big)
    im_phase = unwrap_phase(np.angle(im_fringe))

    return im_phase


def phase_center(im: np.ndarray, center: tuple, mask_osc_flood: np.ndarray = None, plot: bool = False, masks: tuple = None, big: bool = False, unwrap=True) -> np.ndarray:
    """Returns the phase from an interfogram

    Args:
        im (np.ndarray): The interferogram
        plot (bool) : whether to plot something

    Returns:
        np.ndarray: The unwrapped phase
    """
    im_fringe = im_osc_center(
        im, center, mask_osc_flood=mask_osc_flood, cont=False, plot=plot, big=big)
    if unwrap:
        return unwrap_phase(np.angle(im_fringe))
    return np.angle(im_fringe)
