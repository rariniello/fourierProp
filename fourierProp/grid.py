import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift
from scipy.special import jn_zeros


class BaseGrid:
    def __init__(self):
        self.shape = None
        self.gridInd = None
        self.gridType = None
        self.name = None


class Grid(BaseGrid):
    def __init__(self, Nx, Ny, X, Y):
        super().__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.shape = (Nx, Ny)
        self.X = X
        self.Y = Y
        self.gridType = "Cartesian"

        # Create the x and y grid
        self.x, self.dx = np.linspace(
            -X / 2, X / 2, Nx, endpoint=False, retstep=True, dtype="double"
        )
        self.y, self.dy = np.linspace(
            -Y / 2, Y / 2, Ny, endpoint=False, retstep=True, dtype="double"
        )
        # XXX This creates another full size array, could be removed for memory
        self.r = np.sqrt(self.x[:, None] ** 2 + self.y[None, :] ** 2)

        # Calculate the fourier space equivalent of the grid
        dx = self.dx
        dy = self.dy
        self.kx_unshifted = 2 * np.pi * fftfreq(Nx, dx)
        self.ky_unshifted = 2 * np.pi * fftfreq(Ny, dy)
        self.fx = fx = fftshift(fftfreq(Nx, dx))
        self.fy = fy = fftshift(fftfreq(Ny, dy))
        self.dfx = fx[1] - fx[0]
        self.dfy = fy[1] - fy[0]

    def getSaveData(self) -> tuple[dict, dict]:
        """Creates two dictonaries fully describing the grid.

        Neither dictionary should be nested, i.e., they should only be one layer deep.

        Returns:
            Two dict, the first is attributes describing the grid. The second is any data
            that is larger than a simple attribute that must be saved to fully define the grid.
        """
        attr = {
            "Nx": self.Nx,
            "X": self.X,
            "dx": self.dx,
            "dfx": self.dfx,
            "Ny": self.Ny,
            "Y": self.Y,
            "dy": self.dy,
            "dfy": self.dfy,
            "type": self.gridType,
            "name": self.name,
        }

        data = {"x": self.x, "fx": self.fx, "y": self.y, "fy": self.fy}
        return attr, data


# TODO make into a constructor method
def gridFromFileData(attrs):
    Nx = attrs["Nx"]
    X = attrs["X"]
    Ny = attrs["Ny"]
    Y = attrs["Y"]
    return Grid(Nx, Ny, X, Y)


class CylGrid(BaseGrid):
    """A radial grid with uniform grid spacing."""

    def __init__(self, Nr, R, m=0):
        super().__init__()
        self.Nr = Nr
        self.shape = (Nr,)
        self.R = R
        self.m = m
        self.gridType = "Cylindrical"

        # Create the r grid
        self.r, self.dr = np.linspace(
            0.0, R, Nr, endpoint=False, retstep=True, dtype="double"
        )

        # Calculate the fourier space equivalent of the grid
        alpha = jn_zeros(self.m, Nr + 1)
        alpha = alpha[:-1]
        v = alpha / (2 * np.pi * R)
        self.kr = 2 * np.pi * v
        self.fr = self.kr / (2 * np.pi)

    def getSaveData(self) -> tuple[dict, dict]:
        """Creates two dictonaries fully describing the grid.

        Neither dictionary should be nested, i.e., they should only be one layer deep.

        Returns:
            Two dict, the first is attributes describing the grid. The second is any data
            that is larger than a simple attribute that must be saved to fully define the grid.
        """
        attr = {
            "Nr": self.Nr,
            "R": self.R,
            "dr": self.dr,
            "m": self.m,
            "type": self.gridType,
            "name": self.name,
        }

        data = {"r": self.r, "fr": self.fr}
        return attr, data


class CylGridSym(BaseGrid):
    """A radial grid for the symmetric quasi-discrete Fourier transform, nonuniform grid."""

    def __init__(self, Nr, R, m=0):
        super().__init__()
        self.Nr = Nr
        self.shape = (Nr,)
        self.R = R
        self.m = m
        self.gridType = "Cylindrical_SymmetricHankel"

        # Calculate the fourier space equivalent of the grid
        alpha = jn_zeros(self.m, Nr + 1)
        alpha_Np1 = alpha[-1]
        self.alpha = alpha = alpha[:-1]
        self.r = R * alpha / alpha_Np1
        v = alpha / (2 * np.pi * R)
        V = alpha_Np1 / (2 * np.pi * R)
        self.kr = 2 * np.pi * v
        self.fr = self.kr / (2 * np.pi)
        self.S = 2 * np.pi * R * V

    def getSaveData(self) -> tuple[dict, dict]:
        """Creates two dictonaries fully describing the grid.

        Neither dictionary should be nested, i.e., they should only be one layer deep.

        Returns:
            Two dict, the first is attributes describing the grid. The second is any data
            that is larger than a simple attribute that must be saved to fully define the grid.
        """
        attr = {
            "Nr": self.Nr,
            "R": self.R,
            "m": self.m,
            "type": self.gridType,
            "name": self.name,
        }

        data = {"r": self.r, "fr": self.fr}
        return attr, data
