import numpy as np

from fourierProp.planes import Plane
from fourierProp import Grid


class Optic(Plane):
    """Describes a plane that acts as an optic.

    This class acts as an optic by applying a phase change and a transmission mask
    to the field on the upstream side to get the field on the downstream side.
    Optics can be created by

    Attributes:
        grid: The grid object the field is defined on.
        phi: The electric field on the plane in complex representation [V/m]. If None,
            it will be ignored when modifying the field.
        transmission: The electric field on the plane in complex representation [V/m].
            If None, it will be ignored when modifying the field.
    """

    def __init__(self, z: float, n: float, name: str, grid: Grid):
        super().__init__(z, n, name)
        self.grid = grid
        self._phi = None
        self._t = None

    def modifyField(self, E: np.ndarray, lam: float) -> np.ndarray:
        """Returns the field after the optic.

        Subclasses should implement this if they want to generate the phase and transmission
        mask at simulation time. Otherwise, they can generate the phase and transmission mask
        at intiialization and set the phi and transmission attributes.

        Args:
            E: Electric field on the upstream side of the plane, complex representation [V/m].
            lam: Wavelength of the light in vacuum [m].

        Returns:
            A numpy array with the electric field on the second plane in complex
            representation [V/m]. Must have the same dimensions as the input array.
        """
        if self._phi is not None:
            E = E * np.exp(1j * self._phi)
        if self._t is not None:
            E = E * self._t
        return E

    def getSaveData(self):
        attr, data = super().getSaveData()
        if self._phi is not None:
            data["phi"] = self._phi
        if self._t is not None:
            data["t"] = self._t
        return attr, data

    @property
    def phi(self) -> np.ndarray | None:
        return self._phi

    @phi.setter
    def phi(self, value: np.ndarray | None):
        if (value is not None) and (np.shape(value) != (self.grid.Nx, self.grid.Ny)):
            raise ValueError(
                "Phi size does not match grid size, Phi:({}, {}), grid ({}, {})..".format(
                    np.shape(value)[0], np.shape(value)[1], self.grid.Nx, self.grid.Ny
                )
            )
        else:
            self._phi = value

    @property
    def transmission(self) -> np.ndarray | None:
        return self._phi

    @transmission.setter
    def transmission(self, value: np.ndarray | None):
        if (value is not None) and (np.shape(value) != (self.grid.Nx, self.grid.Ny)):
            raise ValueError(
                "Transmission size does not match grid size, Transmission:({}, {}), grid ({}, {})..".format(
                    np.shape(value)[0], np.shape(value)[1], self.grid.Nx, self.grid.Ny
                )
            )
        else:
            self._t = value


class SphericalLens(Optic):
    def __init__(self, z: float, n: float, name: str, grid: Grid, lam: float, f: float):
        super().__init__(z, n, name, grid)
        k = 2 * np.pi / lam
        self.phi = -k * grid.r**2 / (2 * f)
