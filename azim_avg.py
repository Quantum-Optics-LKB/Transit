import numpy as np
import matplotlib.pyplot as plt


def az_avg(image: np.ndarray, center: tuple = None, r=None, all_pt=False) -> np.ndarray:
    """Calculates the azimuthally averaged radial profile.

    Args:
        image (np.ndarray): The 2D image
        center (tuple, optional): The [x,y] pixel coordinates used as the center. Defaults to None, 
        which then uses the center of the image (including fractional pixels). 
        r (np.ndarray, optional): Array of radii. Defaults to None.
        all_pt (bool, optional): Includes all pixels. Defaults to False.

    Returns:
        np.ndarray: [description]
    """
    # Calculate the indices from the image

    if center is not None:
        y, x = np.indices(image.shape)
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    if type(r) is not np.ndarray:
        y, x = np.indices(image.shape)
        r = np.hypot(x - center[0], y - center[1])
        ind = np.argsort(r.flat)

    else:
        ind = np.argsort(r)

    # Get sorted radii
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    if all_pt:
        return r_sorted, i_sorted

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    # csim = np.cumsum(i_sorted, dtype=float)
    csim = np.nancumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
