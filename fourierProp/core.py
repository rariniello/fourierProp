import os
import numpy as np
from scipy.special import jn_zeros, jv
import h5py
import pyfftw
import logging
import json

from fourierProp import Plane
from fourierProp import Grid, CylGrid, CylGridSym
from fourierProp import Source

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# NOTE cylindrical symmetry argument kept for backward compatibility, cylindircal symmetry
# is implemented in the grid the propagator will automatically select the method based on the grids
# XXX should be able to specify diagnostics on a plane by plane basis
def fourierPropagate(
    lam: float,
    planes: list,
    savePath: str | os.PathLike,
    threads: int = 1,
    diagnostics: str = "xy",
    cylSymmetry: bool = False,
):
    """Performs the Fourier optics simulation.

    Sets up the simulation then propagates the field from the source plane to each subsequent
    plane using Fourier optics. Before starting the simulation, volumes are decomposed
    into a series of planes. Output is saved to disk for later analysis.

    Args:
        lam: Wavelength of the light in vacuum [m].
        planes: List of planes and volumes to the simulate the propagation through.
        savePath: Path to save the output files to.
        threads: Number of threads to run the FFTs on.
        diagnostics: Set which output dimensions to save to disk. Options are 'x' or 'y' for 1D lineouts
            or 'xy' for the full 2D output. Ignored if cylSymmetry is True, defaults to 'xy'.
        cylSymmetry: Set to True to only save a 1D lineout of the field at each plane, defaults to False.
            Deprecated, use diagnostics instead, setting to True is equivalent to diagnostics='x'.
    """
    # Expand any volumes into lists of planes
    P = createPlaneList(planes)
    V = collectVolumes(planes)
    M = collectModifiers(planes)
    checkPlanes(P)
    G = assignGridsToPlanes(P)

    # Find all the modifiers and associate them with planes

    grid = P[0].grid
    createSimulationDirectory(savePath)
    saveSimulation(savePath, lam, cylSymmetry, diagnostics)
    savePlanes(P, V, M, G, savePath)

    E = None
    N = len(P)
    propagator = getPropagator(lam, grid, threads)
    for i in range(N):
        # Apply any phase masks/transmission functions
        E = P[i].modifyField(E, lam)

        if P[i].isResample():
            logger.info(
                "Resampling grid from grid {} to grid {}.".format(
                    grid.gridInd, P[i].grid.gridInd
                )
            )
            E = P[i].resampleFromGrid(E, grid)
            grid = P[i].grid
            propagator = getPropagator(lam, grid, threads)

        saveField(E, P[i], i, grid.gridInd, savePath, diagnostics, cylSymmetry)
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
        E = propagator.propagateField(E, deltaZ, n)


def getPropagator(lam: float, grid, threads: int = 1):
    if grid.gridType == "Cartesian":
        return FourierPropagator(lam, grid, threads)
    if grid.gridType == "Cylindrical":
        return UniformHankelPropagator(lam, grid, threads)
    if grid.gridType == "Cylindrical_SymmetricHankel":
        return SymmetricHankelPropagator(lam, grid, threads)
    else:
        raise RuntimeError()


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
            lam: The wavelength of light being propagated, in vacuum [m].
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
            self._kz = kz_RS(self.k, grid, n)
            self.n_previous = n
        e = self.fft(E)
        e *= np.exp(1j * z * self._kz)
        E = self.ifft(e)
        return E


class UniformHankelPropagator:
    def __init__(self, lam: float, grid: CylGrid, threads: int = 1):
        self.grid = grid
        logger.info(
            "Beginning calculation of Hankel transform matrix and inverse matrix"
        )
        self.invT = jv(grid.m, grid.r[:, None] * grid.kr[None, :])
        self.T = np.linalg.inv(self.invT)
        logger.info(
            "Finished calculation of Hankel transform matrix and inverse matrix"
        )
        self.k = 2 * np.pi / lam
        self.n_previous = None

    def propagateField(self, E: np.ndarray, z: float, n: float) -> np.ndarray:
        grid = self.grid
        if n != self.n_previous:
            self._kz = kz_RS(self.k, grid, n)
            self.n_previous = n
        e = np.matmul(self.T, E)
        e *= np.exp(1j * z * self._kz)
        E = np.matmul(self.invT, e)
        return E


class SymmetricHankelPropagator:
    def __init__(self, lam: float, grid: CylGridSym, threads: int = 1):
        self.grid = grid
        m = grid.m
        alpha = grid.alpha
        logger.info("Beginning calculation of Hankel transform matrix")
        self.T = (
            2
            * jv(m, alpha[:, None] * alpha[None, :] / grid.S)
            / (abs(jv(m + 1, alpha[:, None])) * abs(jv(m + 1, alpha[None, :])) * grid.S)
        )
        logger.info("Finished calculation of Hankel transform matrix")
        self.J = abs(jv(m + 1, alpha)) / grid.R
        self.k = 2 * np.pi / lam
        self.n_previous = None

    def propagateField(self, E: np.ndarray, z: float, n: float) -> np.ndarray:
        grid = self.grid
        if n != self.n_previous:
            self._kz = kz_RS(self.k, grid, n)
            self.n_previous = n
        e = np.matmul(self.T, E / self.J)
        e *= np.exp(1j * z * self._kz)
        E = np.matmul(self.T, e) * self.J
        return E


def kz_RS(k: float, grid, n: float) -> np.ndarray:
    """Caluclates the spatial wavenumber in z for each grid point in Fourier space.

    Args:
        k: Wavenumber of the light in vacuum [rad/m].
        grid
        n: Index of refraction between the two planes.

    Returns:
        A numpy array with the spatial wavenumber in z at each point in Fourier space.
    """
    if grid.gridType == "Cartesian":
        return np.sqrt(
            (k * n) ** 2
            - grid.kx_unshifted[:, None] ** 2
            - grid.ky_unshifted[None, :] ** 2
        )
    if grid.gridType == "Cylindrical" or grid.gridType == "Cylindrical_SymmetricHankel":
        return np.sqrt((k * n) ** 2 - grid.kr**2)
    else:
        raise RuntimeError()


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
    # XXX we probably shouldn't let two planes have the same name
    # It will confuse the user when they try to select the plane by name and unwittingly select the first plane
    z_previous = P[0].z
    N = len(P)
    for i in range(N):
        p = P[i]
        z_new = p.z
        if z_new < z_previous:
            raise ValueError(
                "Z decreased between two subsequent planes: {} -> {}, Delta z:{:0.2e}".format(
                    p.name, P[i - 1].name, z_new - z_previous
                )
            )
        z_previous = z_new

    if not isinstance(P[0], Source):
        raise TypeError("The first object in the planes list must be a field source.")


def assignGridsToPlanes(P: list) -> list:
    """Assigns the correct grid object to each plane.

    Args:
        P: List of plane objects.
    """
    grid = P[0].grid
    G = [grid]
    clearGridInd(P)
    gridInd = 0
    for i, p in enumerate(P):
        if i == 0:  # Skip the source plane
            grid.gridInd = gridInd
        elif p.isResample():
            grid = p.grid
            # Only increment the gridInd if the grid has not already been used
            if grid.gridInd is None:
                gridInd += 1
                grid.gridInd = gridInd
                G.append(grid)
        else:
            p.grid = grid
            p.gridInd = gridInd
    return G


def clearGridInd(P):
    grid = P[0].grid
    for i, p in enumerate(P):
        if i == 0:  # Skip the source plane
            grid.gridInd = None
        elif p.isResample():
            grid = p.grid
            grid.gridInd = None


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


def collectModifiers(planes):
    """Makes a new list containing all the modifiers applied to planes."""
    M = []
    i = 0
    for p in planes:
        if p.modifiers is not None:
            for mod in p.modifiers:
                if mod not in M:
                    mod.index = i
                    M.append(mod)
                    i += 1
    return M


# TODO move saving functions to their own file
# TODO abstract filenames into one place where both load and save can find them


def createSimulationDirectory(savePath):
    if not os.path.exists(savePath):
        os.makedirs(savePath)


def saveSimulation(savePath, lam, cylSymmetry, diagnostics):
    """Saves simulation input parameters to a file.

    Args:
        lam: Wavelength of the light in vacuum [m].
        cylSymmetry: Whether the simulation saves only the y=0 slice.
    """
    filename = os.path.join(savePath, "simulation.json")
    logger.info(f"Saving simulation input parameters.")
    parameters = {"lam": lam, "cylSymmetry": cylSymmetry, "diagnostics": diagnostics}
    with open(filename, "w") as fp:
        json.dump(parameters, fp, indent=4)


# TODO, consider saving geometry as a json file so it can be version controlled
# Would have to remove all dataset information (phase on a plane)
# All classes should be able to be reconstructed from the attributes anyways
def savePlanes(P: list, V: list, M: list, G: list, savePath: str | os.PathLike):
    """Saves information about the planes to a file.

    Args:
        P: List of planes to save to file.
        V: List of volumes to save to file.
        M: List of modifiers to save to file.
        G: List of grids to save to file.
        savePath: Path to save the file at.
    """
    filename = os.path.join(savePath, "planes.h5")
    logger.info(f"Saving plane information to {filename}")
    with h5py.File(filename, "w") as f:
        for i, plane in enumerate(P):
            modifiers = getModifierAttr(plane)
            saveObject(
                f,
                plane,
                f"p{i}",
                i,
                modifierIndexes=modifiers,
                gridInd=plane.grid.gridInd,
            )

        for i, volume in enumerate(V):
            modifiers = getModifierAttr(volume)
            saveObject(f, volume, f"v{i}", i, modifierIndexes=modifiers)

        for i, modifier in enumerate(M):
            saveObject(f, modifier, f"m{i}", i)

        for i, grid in enumerate(G):
            grid.name = f"g{i}"
            saveObject(f, grid, f"g{i}", i)

    logger.info("Finished saving plane information")


def saveObject(f: h5py.File, object, name: str, ind, **kwargs):
    """Saves the objects save data into the given hdf5 file.

    Args:
        f: The hdf5 file object to save the data to.
        object: The object whose save data should be saved. The object must have
            a method called getSaveData that returns two dictionaries: attrs and data.
        name: Name of the group representing the object in the dataset. Must be a
            unique identifier.
        ind: Index of the object in the planes or volumes list.
        **kwargs: Any additional attrs to add, will overwrite keys in attr from getSaveData.
    """
    attr, data = object.getSaveData()
    attr["index"] = ind
    attr = attr | kwargs
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
    diagnostics: str,
    cylSymmetry: bool = False,
):
    """Saves the electric field and data about the plane to a file.

    Args:
        E: Electric field to save, complex representation [V/m].
        plane: Plane where the field is defined at.
        ind: Index of the plane in the plane list.
        gridInd: Index of the grid used for this plane.
        savePath: Path to save the file at.
        diagnostics: Set which output dimensions to save to disk. Options are 'x' or 'y' for 1D lineouts
            or 'xy' for the full 2D output. Ignored if cylSymmetry is True, defaults to 'xy'.
        cylSymmetry: If True, will only save the plane along the central
            slice in the x direction. Reduces the file size of the output.
    """
    if cylSymmetry:
        diagnostics = "x"
    filename = os.path.join(savePath, "field_{}.h5".format(ind))
    logger.info(f"Saving field at plane {plane.name} to {filename}")
    with h5py.File(filename, "w") as f:
        size = np.shape(E)
        dtype = E.dtype
        if diagnostics == "x" and len(size) == 2:
            dset = f.create_dataset("E", shape=(size[0],), dtype=dtype)
            dset[...] = E[:, int(size[1] / 2)]
        if diagnostics == "y" and len(size) == 2:
            dset = f.create_dataset("E", shape=(size[0],), dtype=dtype)
            dset[...] = E[int(size[0] / 2), :]
        else:
            dset = f.create_dataset("E", shape=(size), dtype=dtype)
            dset[...] = E
        dset.attrs["z"] = plane.z
        dset.attrs["index"] = ind
        dset.attrs["gridIndex"] = gridInd
    logger.info(f"Finished saving field at plane {plane.name}")


def getModifierAttr(plane):
    modifiers = []
    if plane.modifiers is not None:
        for mod in plane.modifiers:
            modifiers.append(mod.index)
    return modifiers
