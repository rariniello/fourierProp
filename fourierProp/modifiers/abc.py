import numpy as np

from fourierProp.modifiers.modifier import Modifier


class AbsorbingBoundaryCondition(Modifier):
    def __init__(self, name, gamma, Nw, beta=1):
        super().__init__(name)
        self.gamma = gamma
        self.Nw = Nw
        self.beta = beta
        self.calculateAbsorption()

    def calculateAbsorption(self):
        Nw = self.Nw

        m = np.arange(Nw)
        self.T = 0.5 * (1 - self.gamma) * np.cos(
            np.pi * (Nw - m) / Nw
        ) ** self.beta + 0.5 * (1 + self.gamma)
        self.Tr = np.flip(self.T)

    def modifyField(self, E: np.ndarray, lam: float, grid) -> np.ndarray:
        """Applies the absorbing boundary conditions to the plane.

        Args:
            E: Electric field on the plane, complex representation [V/m].
            lam: Wavelength of the light in vacuum [m].

        Returns:
            A numpy array with the absorbing boundary conditions applied in complex
            representation [V/m]. Must have the same dimensions as the input array.
        """
        Nw = self.Nw
        if grid.gridType == "Cartesian":
            E[:Nw, :] *= self.T[:, None]
            E[-Nw:, :] *= self.Tr[:, None]
            E[:, :Nw] *= self.T[None, :]
            E[:, -Nw:] *= self.Tr[None, :]
        elif (
            grid.gridType == "Cylindrical"
            or grid.gridType == "Cylindrical_SymmetricHankel"
        ):
            E[-Nw:] *= self.Tr
        return E

    def getSaveData(self) -> tuple[dict, dict]:
        """Creates two dictonaries fully describing the plane.

        Neither dictionary should be nested, i.e., they should only be one layer deep.

        Returns:
            Two dict, the first is attributes describing the modifier. The second is any data
            that is larger than a simple attribute that must be saved to fully define the modifier.
        """
        attr, data = super().getSaveData()
        attr["gamma"] = self.gamma
        attr["Nw"] = self.Nw
        attr["beta"] = self.beta
        return attr, data
