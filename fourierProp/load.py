import os

import numpy as np
import h5py


def loadGrid(savePath: str | os.PathLike, ind):
    """Loads the grid at the given grid index.

    Args:
        savePath: Path to the simulation output.
        ind: Index of the grid.

    Returns:
        x: Grid coordinates in the x direction.
        y: Grid coordinates in the y direction.
        dx: Grid cell size in x.
        dy: Grid cell size in y.
    """
    filename = os.path.join(savePath, "grid_{}.h5".format(ind))
    f = h5py.File(filename, "r")
    dset_x = f["x/x"]
    dset_y = f["y/y"]
    x = np.array(dset_x)
    y = np.array(dset_y)
    dx = f["x"].attrs["dx"]
    dy = f["y"].attrs["dy"]
    dset_fx = f["x/fx"]
    dset_fy = f["y/fy"]
    fx = np.array(dset_fx)
    fy = np.array(dset_fy)
    f.close()
    return x, y, dx, dy, fx, fy


def loadGridAtPlane(savePath: str | os.PathLike, ind=None, name=None):
    """Loads the grid at the given plane.

    Either an index or a plane name should be specfied. If both are specified,
    index takes presedence. If multiple planes have the same name, the first
    will be returned.

    Args:
        savePath: Path to the simulation output.
        ind: Index of the plane in the plane array.
        name: Name of the plane.

    Returns:
        x: Grid coordinates in the x direction.
        y: Grid coordinates in the y direction.
        dx: Grid cell size in x.
        dy: Grid cell size in y.
    """
    if ind is not None:
        pass
    elif name is not None:
        ind = getPlaneIndexFromName(savePath, name)
    else:
        raise RuntimeError("Must specify either ind or planes arguments.")
    filename = os.path.join(savePath, "field_{}.h5".format(ind))
    f = h5py.File(filename, "r")
    dset = f["E"]
    gridInd = dset.attrs["gridIndex"]
    f.close()
    return loadGrid(savePath, gridInd)


def loadFieldAtPlane(savePath: str | os.PathLike, ind=None, name=None):
    """Loads the electric field at the given plane.

    Either an index or a plane name should be specfied. If both are specified,
    index takes presedence. If multiple planes have the same name, the first
    will be returned.

    Args:
        savePath: Path to the simulation output.
        ind: Index of the plane in the plane array.
        name: Name of the plane.

    Returns:
        A numpy array with the electric field on the the plane in complex
        representation [V/m].
    """
    if ind is not None:
        pass
    elif name is not None:
        ind = getPlaneIndexFromName(savePath, name)
    else:
        raise RuntimeError("Must specify either ind or planes arguments.")
    filename = os.path.join(savePath, "field_{}.h5".format(ind))
    f = h5py.File(filename, "r")
    dset = f["E"]
    field = np.array(dset)
    f.close()
    return field


def loadXZPlaneFromVolume(savePath: str | os.PathLike, name: str):
    volume = None
    filename = os.path.join(savePath, "planes.h5")
    f = h5py.File(filename, "r")
    for key, group in f.items():
        if group.attrs["name"] == name:
            volume = group
            break
    if volume is None:
        raise RuntimeError(f"Volume not found with name {name}")
    Nz = volume.attrs["Nz"]
    f.close()

    startInd = getPlaneIndexFromName(savePath, f"{name}_0")
    x, y, dx, dy, fx, fy = loadGridAtPlane(savePath, ind=startInd)
    Nx = len(x)
    Ny = len(y)
    field = np.zeros((Nz, Nx), dtype="complex128")
    z = np.zeros(Nz)
    for i in range(Nz):
        ind = i + startInd
        filename = os.path.join(savePath, "field_{}.h5".format(ind))
        f = h5py.File(filename, "r")
        dset = f["E"]
        field[i, :] = np.array(dset[:, int(Ny / 2)])
        z[i] = dset.attrs["z"]
        f.close()
    return field, x, z

    # Load the plane list
    # Figure out how big the volume is
    # Preallocate the array to store the data in
    # Load each plane in the volume and extract the X lineout
    raise NotImplementedError


def getPlaneIndexFromName(savePath: str | os.PathLike, name: str) -> int:
    """Load the plane list and find the first plane that has the given name.

    Args:
        savePath: Path to the simulation output.
        name: The name of the plane.

    Returns:
        Index of the plane in the plane list.
    """
    ind = None
    filename = os.path.join(savePath, "planes.h5")
    f = h5py.File(filename, "r")
    for key, group in f.items():
        if group.attrs["name"] == name:
            ind = group.attrs["index"]
            break
    f.close()
    if ind is None:
        raise RuntimeError(f"Plane not found with name {name}")
    return ind
