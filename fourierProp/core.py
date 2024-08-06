import os
import numpy as np
import h5py
import pyfftw
import logging
import json

from fourierProp import Plane
from fourierProp import Grid
from fourierProp import Source

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# TODO cylindrical symmetry argument kept for backward compatibility, cylindircal symmetry
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

    # Find all the modifiers and associated them with planes

    grid = P[0].grid
    gridInd = 0  # Index used for planes to reference their grid
    createSimulationDirectory(savePath)
    saveSimulation(savePath, lam, cylSymmetry)
    saveGrid(grid, gridInd, savePath)
    savePlanes(P, V, M, savePath)

    E = None
    N = len(P)
    # TODO replace by a function that gets the Fourier propagator based on the grid
    fp = FourierPropagator(lam, grid, threads)
    for i in range(N):
        # Apply any phase masks/transmission functions
        E = P[i].modifyField(E, lam)

        if P[i].isResample():
            logger.info(
                "Resampling grid from grid {} to grid {}.".format(gridInd, gridInd + 1)
            )
            E = P[i].resampleFromGrid(E, grid)
            grid = P[i].grid
            gridInd += 1
            saveGrid(grid, gridInd, savePath)
            # TODO replace by a function that gets the Fourier propagator based on the grid
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
            self._kz = kz_RS(self.k, grid.kx_unshifted, grid.ky_unshifted, n)
            self.n_previous = n
        e = self.fft(E)
        e *= np.exp(1j * z * self._kz)
        E = self.ifft(e)
        return E


def kz_RS(k: float, kx: np.ndarray, ky: np.ndarray, n: float) -> np.ndarray:
    """Caluclates the spatial wavenumber in z for each grid point in Fourier space.

    Args:
        k: Wavenumber of the light in vacuum [rad/m].
        kx: Coordinates in kx of each grid point in Fourier space [rad/m].
        ky: Coordinates in kx of each grid point in Fourier space [rad/m].
        n: Index of refraction between the two planes.

    Returns:
        A numpy array with the spatial wavenumber in z at each point in Fourier space.
    """
    return np.sqrt((k * n) ** 2 - kx[:, None] ** 2 - ky[None, :] ** 2)


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


def saveSimulation(savePath, lam, cylSymmetry):
    """Saves simulation input parameters to a file.

    Args:
        lam: Wavelength of the light in vacuum [m].04-PulsePropagationTandemLens-Copy1
        cylSymmetry: Whether the simulation saves only the y=0 slice.
    """
    filename = os.path.join(savePath, "simulation.json")
    logger.info(f"Saving simulation input parameters.")
    parameters = {"lam": lam, "cylSymmetry": cylSymmetry}
    with open(filename, "w") as fp:
        json.dump(parameters, fp, indent=4)


# TODO change to use getSaveData in the grid class, then call saveObject
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


def savePlanes(P: list, V: list, M: list, savePath: str | os.PathLike):
    """Saves information about the planes to a file.

    Args:
        P: List of planes to save to file.
        V: List of volumes to save to file.
        M: List of modifiers to save to file.
        savePath: Path to save the file at.
    """
    # TODO: Save which grid is used for each plane in the planes file
    filename = os.path.join(savePath, "planes.h5")
    logger.info(f"Saving plane information to {filename}")
    Np = len(P)
    Nv = len(V)
    Nm = len(M)
    with h5py.File(filename, "w") as f:
        for i in range(Np):
            plane = P[i]
            modifiers = getModifierAttr(plane)
            saveObject(f, plane, f"p{i}", i, modifierIndexes=modifiers)

        for i in range(Nv):
            volume = V[i]
            modifiers = getModifierAttr(volume)
            saveObject(f, volume, f"v{i}", i, modifierIndexes=modifiers)

        for i in range(Nm):
            modifier = M[i]
            saveObject(f, modifier, f"m{i}", i)

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


# TODO implement diagnostics and cylindrically symmetric grids
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
    logger.info(f"Finished saving field at plane {plane.name}")


def getModifierAttr(plane):
    modifiers = []
    if plane.modifiers is not None:
        for mod in plane.modifiers:
            modifiers.append(mod.index)
    return modifiers
