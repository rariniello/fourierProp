import numpy as np

import fourierProp.load as load
import fourierProp.grid as grid
from fourierProp.planes import Plane
from fourierProp.planes import resample
from fourierProp import Grid


class Source(Plane):
    """Describes a plane that acts a source for the electric field.

    This class acts as a source for the electric field. The modifyField method returns
    an electric field independent of the passed field. It also provides the initial
    simulation grid to the Fourier simulation. Fields can be calculated external to the
    class and then applied by setting the E attribute. A child class can also be created
    to define the field, see modifyField for more details.

    Attributes:
        grid: The grid object the field is defined on.
        E: The electric field on the plane in complex representation [V/m].
    """

    def __init__(self, z: float, n: float, name: str, grid: Grid):
        super().__init__(z, n, name)
        self.grid = grid

    def modifyField(self, E: np.ndarray, lam: float) -> np.ndarray:
        """Returns the field of the source.

        Subclasses should implement this if they want to generate the field at simulation time.
        Otherwise, they can generate the field when the class is initialized and set E.

        Args:
            E: Electric field on the upstream side of the plane, complex representation [V/m].
                If the source plane is the first plane, E will be None.
            lam: Wavelength of the light in vacuum [m].

        Returns:
            A numpy array with the electric field on the second plane in complex
            representation [V/m]. Must have the same dimensions as the input array.
        """
        return self._E

    @property
    def E(self) -> np.ndarray:
        return self._E

    @E.setter
    def E(self, value: np.ndarray):
        if np.shape(value) != (self.grid.Nx, self.grid.Ny):
            raise ValueError(
                "E size does not match grid size, E:({}, {}), grid ({}, {}).".format(
                    np.shape(value)[0], np.shape(value)[1], self.grid.Nx, self.grid.Ny
                )
            )
        else:
            self._E = value

    def isSource(self) -> bool:
        return True

    def isIndependentOfWavelength(self) -> bool:
        return True


# TODO: A source where the field depends on lam

# TODO: Create gaussian source.

# TODO: Create super-Gaussian source.


class LoadField(Source):
    def __init__(
        self,
        z: float,
        n: float,
        name: str,
        grid: Grid,
        loadPath,
        ind=None,
        loadName=None,
    ):
        super().__init__(z, n, name, grid)
        self.loadPath = loadPath
        self.ind = ind
        self.loadName = loadName
        params = load.loadSimulation(loadPath)
        # TODO load diagnostics
        self.cylSymmetry = params["cylSymmetry"]

    def modifyField(self, E: np.ndarray, lam: float) -> np.ndarray:
        # Load the grid and field at the specified plane
        # TODO implement for cylindrically symmetric grids
        attrs, data = load.loadGridAtPlane(self.loadPath, self.ind, self.loadName)
        grid1 = grid.gridFromFileData(attrs)
        grid2 = self.grid

        E = load.loadFieldAtPlane(self.loadPath, self.ind, self.loadName)

        if self.cylSymmetry:
            E = resample.resampleRadialInterpolation(E, grid1, grid2)
        else:
            E = resample.resample2dInterpolation(E, grid1, grid2)
        self.E = E
        return self._E
