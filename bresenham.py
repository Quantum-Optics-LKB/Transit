from numba import jit
import numpy as np
import time

# function for line generation


# @jit(nopython=True)
# def bresenham(x1, y1, x2, y2):
#
#     m_new = 2 * (y2 - y1)
#     slope_error_new = m_new - (x2 - x1)
#
#     y = y1
#     coords = []
#     for x in range(x1, x2+1):
#         coords.append([x, y])
#         # Add slope to increment angle formed
#         slope_error_new += m_new
#
#         # Slope error reached limit, time to
#         # increment y and update slope error.
#         if (slope_error_new >= 0):
#             y += 1
#             slope_error_new -= 2 * (x2 - x1)
#
#     return coords
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


# driver function

if __name__ == '__main__':
    # simulation of speed for 256x256 array
    N = 2048
    N_loop = 1000
    x1, x2 = 0, N
    t0 = time.time()
    for i in range(N_loop):
        y1, y2 = np.random.randint(N, size=2)
        coords = bresenham(x1, y1, x2, y2)
    t1 = time.time() - t0
    print(f"Time elapsed : {t1} s")
