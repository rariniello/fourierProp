import numpy as np

from fourierProp.planes import Plane
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

    def modifyField(self, E) -> np.ndarray:
        """Returns the field of the source.

        Subclasses should implement this if they want to generate the field at simulation time.
        Otherwise, they can generate the field when the class is initialized and set E.

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


# TODO: Create gaussian source.

# TODO: Create super-Gaussian source.
