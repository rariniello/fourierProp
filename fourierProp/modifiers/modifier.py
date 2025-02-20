import numpy as np


class Modifier:
    def __init__(self, name: str):
        self.name = name
        self.index = None

    def modifyField(self, E: np.ndarray, lam: float, grid) -> np.ndarray:
        """Modifes the field at a plane.

        Subclasses implement this to modify the field on the parent plane.

        It is up to the plane to define if the modifier is applied before or after the plane.

        Args:
            E: Electric field on the plane, complex representation [V/m].
            lam: Wavelength of the light in vacuum [m].

        Returns:
            A numpy array with the modofied electric field in complex
            representation [V/m]. Must have the same dimensions as the input array.
        """
        return E

    def getSaveData(self) -> tuple[dict, dict]:
        """Creates two dictonaries fully describing the plane.

        Neither dictionary should be nested, i.e., they should only be one layer deep.

        Returns:
            Two dict, the first is attributes describing the plane. The second is any data
            that is larger than a simple attribute that must be saved to fully define the plane.
        """
        attr = {
            "name": self.name,
        }
        data = {}
        return attr, data
