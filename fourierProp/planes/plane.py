import numpy as np


class Plane:
    """Describes a plane perpendicular to the z axis that the field is defined on.

    This class can be used directly as a measurement plane, i.e., a plane where the
    field is calculated at, but otherwise doesn't modify the plane at all.

    Attributes:
        z: Location of the plane along the z-axis.
        n: Index of refraction of the medium on the downstream side of the plane.
        name: Name of the plane, only a referene for the user. Does not have to be unique.
    """

    def __init__(self, z: float, n: float, name: str):
        self.z = z
        self.n = n
        self.name = name

    def modifyField(self, E: np.ndarray) -> np.ndarray:
        """Modifes the given field on the upstream side of the plane to get the field on the downstream side.

        Subclasses should implement this if they want to change the field as it passes the plane.

        Args:
            E: Electric field on the upstream side of the plane, complex representation [V/m].

        Returns:
            A numpy array with the electric field on the second plane in complex
            representation [V/m]. Must have the same dimensions as the input array.
        """
        return E

    def isVolume(self) -> bool:
        return False

    def isResample(self) -> bool:
        return False

    def getSaveData(self) -> tuple[dict, dict]:
        """Creates two dictonaries fully describing the plane.

        Neither dictionary should be nested, i.e., they should only be one layer deep.

        Returns:
            Two dict, the first is attributes describing the plane. The second is any data
            that is larger than a simple attribute that must be saved to fully define the plane.
        """
        attr = {
            "name": self.name,
            "z": self.z,
            "n": self.n,
        }
        data = {}
        return attr, data
