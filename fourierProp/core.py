import os
import numpy as np
import h5py
import pyfftw
import logging

from fourierProp import Plane
from fourierProp import Grid
from fourierProp import Source

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def fourierPropagate(
    lam: float,
    planes: list,
    savePath: str | os.PathLike,
    threads: int = 1,
    cylSymmetry: bool = False,
):
    """Performs the Fourier optics simulation.

    Sets up the simulation then propagates the field from the source plane to each subsequent
    plane using Fourier optics. Before starting the simulation, volumes are decomposed
    into a series of planes. Output is saved to disk for later analysis.

    Args:
        lam: Wavelength of the light [m].
        planes: List of planes and volumes to the simulate the propagation through.
        savePath: Path to save the output files to.
        threads: Number of threads to run the FFTs on.
        cylSymmetry: Set to True to only save a 1D lineout of the field at each plane, defaults to False.
    """
    # Expand any volumes into lists of planes
    P = createPlaneList(planes)
    V = collectVolumes(planes)
    checkPlanes(P)

    grid = P[0].grid
    gridInd = 0  # Index used for planes to reference their grid
    saveGrid(grid, gridInd, savePath)
    savePlanes(P, V, savePath)

    E = None
    N = len(P)
    fp = FourierPropagator(lam, grid, threads)
    for i in range(N):
        # Apply any phase masks/transmission functions
        E = P[i].modifyField(E)

        if P[i].isResample():
            logger.info(
                "Resampling grid from grid {} to grid {}.".format(gridInd, gridInd + 1)
            )
            E = P[i].resampleFromGrid(E, grid)
            grid = P[i].grid
            gridInd += 1
            saveGrid(grid, gridInd, savePath)
            fp = FourierPropagator(lam, grid, threads)

        saveField(E, P[i], i, gridInd, savePath, cylSymmetry)
        if isLastPlane(i, N):
            break

        logger.info(
            "Propagating from plane {} of {} ({}) to plane {} ({})".format(
                i + 1, N, P[i].name, i + 2, P[i + 1].name
            )
        )
        deltaZ = P[i + 1].z - P[i].z
        if deltaZ == 0.0:
            continue
        n = P[i].n
        E = fp.propagateField(E, deltaZ, n)


class FourierPropagator:
    """Fourier propagator that can transfer the field from one plane to another.

    This class acts as a convient data structure to store the fft and ifft plan.
    It also implements some basic caching of the kz grid to reduce computation.
    Typically objects of this class aren't instantiated directly, rather they are
    instatiated as part of the simulation process of fourierPropagate.

    Attributes:
        grid: The grid object the FourierPropagator is defined for.
        fft: FFT plan from fftw.
        ifft: Inverse FFT plan from fftw.
        k: Wavenumber of the light [rad/m].
    """

    def __init__(self, lam: float, grid: Grid, threads: int = 1):
        """Initializes the class.

        Args:
            lam: The wavelength of light being propagated [m].
            grid: The transverse grid the field is defined on.
            threads: Number of threads to use in the fft and ifft.
        """
        self.grid = grid
        self.fft, self.ifft = createFFTPlan(grid, threads)
        self.k = 2 * np.pi / lam
        self.n_previous = None

    def propagateField(self, E: np.ndarray, z: float, n: float) -> np.ndarray:
        """Fourier propagats the field between planes seperated by the given distance and index.

        Args:
            E: Electric field on the first plane, complex representation [V/m].
            z: Distance between the planes [m].
            n: Index of refraction between the two planes.

        Returns:
            A numpy array with the electric field on the second plane in complex
            representation [V/m].
        """
        grid = self.grid
        if n != self.n_previous:
            self._kz = kz_RS(self.k, grid.kx_unshifted, grid.ky_unshifted, n)
            self.n_previous = n
        e = self.fft(E)
        e *= np.exp(1j * z * self._kz)
        E = self.ifft(e)
        return E


def kz_RS(k: float, kx: np.ndarray, ky: np.ndarray, n: float) -> np.ndarray:
    """Caluclates the spatial wavenumber in z for each grid point in Fourier space.

    Args:
        k: Wavenumber of the light [rad/m].
        kx: Coordinates in kx of each grid point in Fourier space [rad/m].
        ky: Coordinates in kx of each grid point in Fourier space [rad/m].
        n: Index of refraction between the two planes.

    Returns:
        A numpy array with the spatial wavenumber in z at each point in Fourier space.
    """
    return np.sqrt(k**2 * n - kx[:, None] ** 2 - ky[None, :] ** 2)


def createFFTPlan(grid: Grid, threads: int = 1):
    """Creates the fftw plan for the grid.

    Args:
        grid: The grid object that determines the plan size.
        threads: Number of threads to run the fft on.

    Returns:
        The fft plan and the ifft plan for a compex field on the passed grid.
        The plans can be called directly with the field as an argument to
        perform the transforms.
    """
    efft = pyfftw.empty_aligned((grid.Nx, grid.Ny), dtype="complex128")
    fft = pyfftw.builders.fftn(
        efft, overwrite_input=True, avoid_copy=True, threads=threads, axes=(0, 1)
    )
    ifft = pyfftw.builders.ifftn(
        efft, overwrite_input=True, avoid_copy=True, threads=threads, axes=(0, 1)
    )
    return fft, ifft


def checkPlanes(P: list):
    """Checks that the set of planes is internally consistent.

    1. Check that z always increases or remains the same through the planes.
    2. Check that the first plane is a source.

    Args:
        P: List of plane objects.

    Raises:
        ValueError: Z decreased between to subsequent planes.
        TypeError: The first plane is not a source.
    """
    z_previous = P[0].z
    N = len(P)
    for i in range(N):
        p = P[i]
        z_new = p.z
        if z_new < z_previous:
            raise ValueError(
                "Z decreased between two subsequent planes: {} -> {}, Delta z:{:0.2e}".format(
                    p.name, P[i - 1].name, z_new
                )
            )
        z_previous = z_new

    if not isinstance(P[0], Source):
        raise TypeError("The first object in the planes list must be a field source.")


def createPlaneList(planes: list) -> list:
    """Expands any volumes in the planes list to create a list of only planes.

    Args:
        planes: List of planes and volumes.

    Returns:
        A list of planes with all given volumes expanded into a series of planes.
    """
    # XXX If we ever want more complicated behavior, we can make a planes object
    # The object would include methods such as is isLastPlane
    P = []
    for p in planes:
        if p.isVolume():
            planes = p.generatePlanes()
            P += planes
        else:
            P.append(p)
    return P


def isLastPlane(i: int, N: int):
    """Checks if the given index is the final index in the list."""
    return i == N - 1


def collectVolumes(planes: list) -> list:
    """Makes a new list containing all the volumes in planes.

    Args:
        planes: List of planes and volumes.
    """
    V = []
    for p in planes:
        if p.isVolume():
            V.append(p)
    return V


def saveGrid(grid: Grid, gridInd: int, savePath: str | os.PathLike):
    """Saves the given grid to a file.

    Args:
        grid: The grid to save.
        gridInd: Index of the grid, used to reference a particular grid and appears in the filename.
        savePath: Path to save the file at.
    """
    filename = os.path.join(savePath, "grid_{}.h5".format(gridInd))
    logger.info(f"Saving grid {gridInd} to {filename}")
    with h5py.File(filename, "w") as f:
        # x direction
        dset_x = f.create_dataset("x/x", shape=(grid.Nx,), dtype="double")
        dset_x[:] = grid.x
        dset_fx = f.create_dataset("x/fx", shape=(grid.Nx,), dtype="double")
        dset_fx[:] = grid.fx
        f["x"].attrs["Nx"] = grid.Nx
        f["x"].attrs["X"] = grid.X
        f["x"].attrs["dx"] = grid.dx
        f["x"].attrs["dfx"] = grid.dfx

        # y direction
        dset_y = f.create_dataset("y/y", shape=(grid.Ny,), dtype="double")
        dset_y[:] = grid.y
        dset_fy = f.create_dataset("y/fy", shape=(grid.Ny,), dtype="double")
        dset_fy[:] = grid.fy
        f["y"].attrs["Ny"] = grid.Ny
        f["y"].attrs["Y"] = grid.Y
        f["y"].attrs["dy"] = grid.dy
        f["y"].attrs["dfy"] = grid.dfy
    logger.info(f"Finished saving grid {gridInd}")


def savePlanes(P: list, V: list, savePath: str | os.PathLike):
    """Saves information about the planes to a file.

    Args:
        P: List of planes to save to file.
        V: List of volumes to save to file.
        savePath: Path to save the file at.
    """
    filename = os.path.join(savePath, "planes.h5")
    logger.info(f"Saving plane information to {filename}")
    N = len(P)
    M = len(V)
    with h5py.File(filename, "w") as f:
        for i in range(N):
            plane = P[i]
            saveObject(f, plane, f"p{i}")

        for i in range(M):
            volume = V[i]
            saveObject(f, volume, f"v{i}")

    logger.info(f"Finished saving plane information")


def saveObject(f: h5py.File, object, name: str):
    """Saves the objects save data into the given hdf5 file.

    Args:
        f: The hdf5 file object to save the data to.
        object: The object whose save data should be saved. The object must have
            a method called getSaveData that return two dictionaries: attrs and data.
        name: Name of the group representing the object in the dataset. Must be a
            unique identifier.
    """
    attr, data = object.getSaveData()
    # Save the attrs as hdf5 attrs
    group = f.create_group(name)
    group.attrs.update(attr)
    # Save the data items as hdf5 datasets
    for name, value in data.items():
        size = np.shape(value)
        dtype = value.dtype
        dset = group.create_dataset(name, size, dtype=dtype)
        dset[...] = value


def saveField(
    E,
    plane: Plane,
    ind: int,
    gridInd: int,
    savePath: str | os.PathLike,
    cylSymmetry: bool = False,
):
    """Saves the electric field and data about the plane to a file.

    Args:
        E: Electric field to save, complex representation [V/m].
        plane: Plane where the field is defined at.
        ind: Index of the plane in the plane list.
        gridInd: Index of the grid used for this plane.
        savePath: Path to save the file at.
        cylSymmetry: If True, will only save the plane along the central
            slice in the x direction. Reduces the file size of the output.
    """
    filename = os.path.join(savePath, "field_{}.h5".format(ind))
    logger.info(f"Saving field at plane {plane.name} to {filename}")
    with h5py.File(filename, "w") as f:
        size = np.shape(E)
        dtype = E.dtype
        if cylSymmetry:
            dset = f.create_dataset("E", shape=(size[0],), dtype=dtype)
            dset[...] = E[:, int(size[1] / 2)]
        else:
            dset = f.create_dataset("E", shape=(size), dtype=dtype)
            dset[...] = E
        dset.attrs["z"] = plane.z
        dset.attrs["index"] = ind
        dset.attrs["gridIndex"] = gridInd
    # logger.info(f"Finished saving field at plane {plane.name}")
