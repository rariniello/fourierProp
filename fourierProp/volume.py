import numpy as np

from fourierProp import Plane


class Volume:
    """Describes a volume between two planes through which the electric field should be calculated.

    The volume is converted to a series of planes, one at each point in z, at simulation time.

    Attributes:
        z: Array of z locations to calculate the field within the volume.
        n: Index of refraction within the volume. Will also be after the volume until the next plane.
        z_i: Location of the plane defining the start of the volume.
        z_f: Location of the plane defining the end of the volume.
        Nz: Number of planes within the volume, inclusive.
    """

    def __init__(self, name, n, z_i=None, z_f=None, N=None, z=None):
        """Initializes the class.

        Args:
            lam: The wavelength of light being propagated [m].
            grid: The transverse grid the field is defined on.
            threads: Number of threads to use in the fft and ifft.
        """
        self.n = n
        self.name = name

        if z is not None:
            self.z_i = z[0]
            self.z_f = z[-1]
            self.N = len(z)
            self.z = z
        elif (z_i is not None) and (z_f is not None) and (N is not None):
            self.z_i = z_i
            self.z_f = z_f
            self.N = N
            self.z = np.linspace(z_i, z_f, N)
        else:
            raise RuntimeError("Specify either z or z_i, z_f, and N")

    def generatePlanes(self) -> list:
        """Generates the planes at each z point in the volume.

        Returns:
            A list of planes of len Nz, one for each element of z.
        """
        P = []
        for i in range(self.N):
            p = Plane(self.z[i], self.n, f"{self.name}_{i}")
            P.append(p)
        return P

    def isVolume(self):
        return True
