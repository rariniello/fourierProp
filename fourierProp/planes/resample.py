import numpy as np
from scipy.interpolate import interp1d, interp2d

from fourierProp.planes import Plane
from fourierProp import Grid


class ResamplePlane(Plane):
    def __init__(
        self, z: float, n: float, name: str, grid: Grid, cylSymmetric: bool = False
    ):
        super().__init__(z, n, name)
        self.grid = grid
        self.cylSymmetric = cylSymmetric

    def isResample(self) -> bool:
        return True

    def resampleFromGrid(self, E, grid):
        """Resamples the field from the given grid to this objects grid.

        Args:
            E: The field on grid that will be resampled to self.grid.
            grid: The grid the field is defined on.

        Returns:
            A numpy array representing the field on the new grid.
        """
        grid1 = grid
        grid2 = self.grid
        if self.cylSymmetric:
            E = resampleRadialInterpolation(E, grid1, grid2)
        else:
            E = resample2dInterpolation(E, grid1, grid2)
        return E


def resampleRadialInterpolation(E: np.ndarray, grid1: Grid, grid2: Grid) -> np.ndarray:
    """Resamples the given field between grids by interpolating in the radial direction.

    The field must be cylindrically symmetric for this function to faithfully
    resample the field.

    Args:
        E: The field on grid that will be resampled to self.grid.
        grid1: The grid the field is defined on.
        grid2: The grid to resample the field to.

    Returns:
        A numpy array representing the field on grid2.
    """
    r = grid1.x
    data = np.array(E[:, int(grid1.Ny / 2)])
    x = grid2.x
    y = grid2.y
    return radialInterpolationToGrid(r, data, x, y)


def radialInterpolationToGrid(
    r: np.ndarray, data: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Fills a 2D grid with a field by interpolating from a radial lineout of the field.

    Args:
        r: Radial corrdinates of the lineout. Does not need to be uniformly spaced.
        data: Field values along the lineout.
        x: x coordinates of the grid.
        y: y coordinates of the grid.

    Returns:
        A numpy array of the field on the (x, y) grid.
    """
    dataOfR = interp1d(r, data, bounds_error=False, fill_value=0.0, kind="cubic")
    return dataOfR(np.sqrt(x[:, None] ** 2 + y[None, :] ** 2))


def resample2dInterpolation(E: np.ndarray, grid1: Grid, grid2: Grid) -> np.ndarray:
    """Resamples the given complex field between grids by interpolating in 2D.

    The real and imaginary components are interpolated seperately.

    Args:
        E: The field on grid that will be resampled to self.grid.
        grid1: The grid the field is defined on.
        grid2: The grid to resample the field to.

    Returns:
        A numpy array representing the field on grid2.
    """
    f_r = interp2d(
        grid1.x, grid1.y, E.real, "cubic", bounds_error=False, fill_value=0.0
    )
    f_i = interp2d(
        grid1.x, grid1.y, E.imag, "cubic", bounds_error=False, fill_value=0.0
    )
    E_r = f_r(grid2.x, grid2.y)
    E_i = f_i(grid2.x, grid2.y)
    return E_r + 1j * E_i
