import os

import numpy as np
import h5py
import json


def loadGroup(f, name):
    """Loads the given hdf5 group as dictionaries from the given file.

    Args:
        f: File object for the hdf5 file.
        name: Name of the group in the file.

    Returns:
        attrs:
        data:
    """
    attrs = dict(f[name].attrs.items())
    data = {}
    for name, value in f[name].items():
        data[name] = np.array(value)
    return attrs, data


def loadSimulation(savePath: str | os.PathLike):
    """Loads the simulation input parameters from disk."""
    filename = os.path.join(savePath, "simulation.json")
    with open(filename, "r") as fp:
        parameters = json.load(fp)
    return parameters


def loadGrid(savePath: str | os.PathLike, ind):
    """Loads the grid at the given grid index.

    Args:
        savePath: Path to the simulation output.
        ind: Index of the grid.

    Returns:
        attrs: Attributes describing the grid, see saveGrid for details.
        data: Grids in x, y, fx and fy.
    """
    filename = os.path.join(savePath, "grid_{}.h5".format(ind))
    f = h5py.File(filename, "r")
    attrs_x, data_x = loadGroup(f, "x")
    attrs_y, data_y = loadGroup(f, "y")
    attrs = {"x": attrs_x, "y": attrs_y}
    data = {"x": data_x, "y": data_y}
    f.close()
    return attrs, data


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
        attrs: Attributes describing the grid, see saveGrid for details.
        data: Grids in x, y, fx and fy.
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


def _loadOnAxisPoint(filename, cylSymmetry, Nx, Ny):
    f = h5py.File(filename, "r")
    dset = f["E"]
    if cylSymmetry:
        field = np.array(dset[int(Nx / 2)])
    else:
        field = np.array(dset[int(Nx / 2), int(Ny / 2)])
    attrs = dict(dset.attrs.items())
    f.close()
    return field, attrs


def _loadXSlice(filename, cylSymmetry, Ny):
    f = h5py.File(filename, "r")
    dset = f["E"]
    if cylSymmetry:
        field = np.array(dset[:])
    else:
        field = np.array(dset[:, int(Ny / 2)])
    attrs = dict(dset.attrs.items())
    f.close()
    return field, attrs


def loadXFieldAtPlane(savePath: str | os.PathLike, ind=None, name=None):
    """Loads the electric field at y=0 at the given plane.

    Either an index or a plane name should be specfied. If both are specified,
    index takes presedence. If multiple planes have the same name, the first
    will be returned.

    Args:
        savePath: Path to the simulation output.
        ind: Index of the plane in the plane array.
        name: Name of the plane.

    Returns:
        A numpy array with the electric field on the the plane at y=0 in complex
        representation [V/m].
    """
    if ind is not None:
        pass
    elif name is not None:
        ind = getPlaneIndexFromName(savePath, name)
    else:
        raise RuntimeError("Must specify either ind or planes arguments.")
    attrs, data = loadGridAtPlane(savePath, ind=ind)
    Ny = attrs["y"]["Ny"]
    params = loadSimulation(savePath)
    cylSymmetry = params["cylSymmetry"]

    filename = os.path.join(savePath, "field_{}.h5".format(ind))
    return _loadXSlice(filename, cylSymmetry, Ny)[0]


def loadOnAxisFieldAtPlane(savePath: str | os.PathLike, ind=None, name=None):
    """Loads the electric field at (x, y)=(0, 0) at the given plane.

    Either an index or a plane name should be specfied. If both are specified,
    index takes presedence. If multiple planes have the same name, the first
    will be returned.

    Args:
        savePath: Path to the simulation output.
        ind: Index of the plane in the plane array.
        name: Name of the plane.

    Returns:
        The electric field on the the plane at (x, y)=(0, 0) in complex
        representation [V/m].
    """
    if ind is not None:
        pass
    elif name is not None:
        ind = getPlaneIndexFromName(savePath, name)
    else:
        raise RuntimeError("Must specify either ind or planes arguments.")
    attrs, data = loadGridAtPlane(savePath, ind=ind)
    Nx = attrs["x"]["Nx"]
    Ny = attrs["y"]["Ny"]
    params = loadSimulation(savePath)
    cylSymmetry = params["cylSymmetry"]

    filename = os.path.join(savePath, "field_{}.h5".format(ind))
    return _loadOnAxisPoint(filename, cylSymmetry, Nx, Ny)[0]


def loadXZPlaneFromVolume(savePath: str | os.PathLike, name: str):
    # Load the planes file to figure out which planes are in the volume
    attrs, data = loadItem(savePath, name)
    Nz = attrs["Nz"]

    # Set the start index and the load the grid to get the array size and x array
    startInd = getPlaneIndexFromName(savePath, f"{name}_0")
    attrs, data = loadGridAtPlane(savePath, ind=startInd)
    Nx = attrs["x"]["Nx"]
    Ny = attrs["y"]["Ny"]
    x = data["x"]["x"]

    # Load the simulation parameters to see if it is cylindrically symmetric
    params = loadSimulation(savePath)
    cylSymmetry = params["cylSymmetry"]

    # Load the field at each plane and populate the array
    field = np.zeros((Nz, Nx), dtype="complex128")
    z = np.zeros(Nz)
    for i in range(Nz):
        ind = i + startInd
        filename = os.path.join(savePath, "field_{}.h5".format(ind))
        field[i, :], attrs = _loadXSlice(filename, cylSymmetry, Ny)
        z[i] = attrs["z"]
    return field, x, z


def loadOnAxisFromVolume(savePath: str | os.PathLike, name: str):
    # Load the planes file to figure out which planes are in the volume
    attrs, data = loadItem(savePath, name)
    Nz = attrs["Nz"]

    # Set the start index and the load the grid to get the array size and x array
    startInd = getPlaneIndexFromName(savePath, f"{name}_0")
    attrs, data = loadGridAtPlane(savePath, ind=startInd)
    Nx = attrs["x"]["Nx"]
    Ny = attrs["y"]["Ny"]

    # Load the simulation parameters to see if it is cylindrically symmetric
    params = loadSimulation(savePath)
    cylSymmetry = params["cylSymmetry"]

    # Load the field at each plane and populate the array
    field = np.zeros(Nz, dtype="complex128")
    z = np.zeros(Nz)
    for i in range(Nz):
        ind = i + startInd
        filename = os.path.join(savePath, "field_{}.h5".format(ind))
        field[i], attrs = _loadOnAxisPoint(filename, cylSymmetry, Nx, Ny)
        z[i] = attrs["z"]
    return field, z


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


def loadItem(savePath: str | os.PathLike, name: str):
    item = None
    filename = os.path.join(savePath, "planes.h5")
    f = h5py.File(filename, "r")
    for key, group in f.items():
        if group.attrs["name"] == name:
            attrs, data = loadGroup(f, key)
            item = True
            break
    if item is None:
        raise RuntimeError(f"Volume not found in planes file with name {name}")
    f.close()
    return attrs, data


def loadPlane(savePath: str | os.PathLike, ind: str):
    filename = os.path.join(savePath, "planes.h5")
    f = h5py.File(filename, "r")
    attrs, data = loadGroup(f, f"p{ind}")
    f.close()
    return attrs, data
