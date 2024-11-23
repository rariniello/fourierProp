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


def loadPlane(savePath: str | os.PathLike, ind: str):
    """Loads the plane at the given plane index.

    Args:
        savePath: Path to the simulation output.
        ind: Index of the plane.

    Returns:
        attrs: Attributes describing the plane, see the grid definition for details.
        data: Additional plane data, see the plane definition for details.
    """
    filename = os.path.join(savePath, "planes.h5")
    f = h5py.File(filename, "r")
    attrs, data = loadGroup(f, f"p{ind}")
    f.close()
    return attrs, data


def loadGrid(savePath: str | os.PathLike, ind):
    """Loads the grid at the given grid index.

    Args:
        savePath: Path to the simulation output.
        ind: Index of the grid.

    Returns:
        attrs: Attributes describing the grid, see the grid definition for details.
        data: Grids coordinates, see the grid definition for details.
    """
    filename = os.path.join(savePath, "planes.h5")
    f = h5py.File(filename, "r")
    attrs, data = loadGroup(f, f"g{ind}")
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
    filename = os.path.join(savePath, "planes.h5")
    f = h5py.File(filename, "r")
    attrs, _ = loadGroup(f, f"p{ind}")
    gridInd = attrs["gridInd"]
    attrs, data = loadGroup(f, f"g{gridInd}")
    f.close()
    return attrs, data


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


def _loadOnAxisPoint(filename, diagnostics, gridAttrs):
    f = h5py.File(filename, "r")
    dset = f["E"]
    if gridAttrs["type"] == "Cartesian":
        if diagnostics == "x":
            field = np.array(int(gridAttrs["Nx"] / 2))
        elif diagnostics == "xy":
            field = np.array(dset[int(gridAttrs["Nx"] / 2), int(gridAttrs["Ny"] / 2)])
    if (
        gridAttrs["type"] == "Cylindrical"
        or gridAttrs["type"] == "Cylindrical_SymmetricHankel"
    ):
        field = np.array(dset[0])
    attrs = dict(dset.attrs.items())
    f.close()
    return field, attrs


def _loadXSlice(filename, diagnostics, gridAttrs):
    f = h5py.File(filename, "r")
    dset = f["E"]
    if gridAttrs["type"] == "Cartesian":
        if diagnostics == "x":
            field = np.array(dset[:])
        elif diagnostics == "xy":
            field = np.array(dset[:, int(gridAttrs["Ny"] / 2)])
    if (
        gridAttrs["type"] == "Cylindrical"
        or gridAttrs["type"] == "Cylindrical_SymmetricHankel"
    ):
        field = np.array(dset[:])
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
    params = loadSimulation(savePath)
    diagnostics = params["diagnostics"]

    filename = os.path.join(savePath, "field_{}.h5".format(ind))
    return _loadXSlice(filename, diagnostics, attrs)[0]


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
    params = loadSimulation(savePath)
    diagnostics = params["diagnostics"]

    filename = os.path.join(savePath, "field_{}.h5".format(ind))
    return _loadOnAxisPoint(filename, diagnostics, attrs)[0]


def loadXZPlaneFromVolume(savePath: str | os.PathLike, name: str):
    # Load the planes file to figure out which planes are in the volume
    attrs, data = loadItem(savePath, name)
    Nz = attrs["Nz"]

    # Set the start index and the load the grid to get the array size and x array
    startInd = getPlaneIndexFromName(savePath, f"{name}_0")
    gridAttrs, data = loadGridAtPlane(savePath, ind=startInd)
    if gridAttrs["type"] == "Cartesian":
        Nx = gridAttrs["Nx"]
        x = data["x"]
    if (
        gridAttrs["type"] == "Cylindrical"
        or gridAttrs["type"] == "Cylindrical_SymmetricHankel"
    ):
        Nx = gridAttrs["Nr"]
        x = data["r"]

    # Load the simulation parameters to see if it is cylindrically symmetric
    params = loadSimulation(savePath)
    diagnostics = params["diagnostics"]

    # Load the field at each plane and populate the array
    field = np.zeros((Nz, Nx), dtype="complex128")
    z = np.zeros(Nz)
    for i in range(Nz):
        ind = i + startInd
        filename = os.path.join(savePath, "field_{}.h5".format(ind))
        field[i, :], attrs = _loadXSlice(filename, diagnostics, gridAttrs)
        z[i] = attrs["z"]
    return field, x, z


def loadOnAxisFromVolume(savePath: str | os.PathLike, name: str):
    # Load the planes file to figure out which planes are in the volume
    attrs, data = loadItem(savePath, name)
    Nz = attrs["Nz"]

    # Set the start index and the load the grid to get the array size and x array
    startInd = getPlaneIndexFromName(savePath, f"{name}_0")
    gridAttrs, data = loadGridAtPlane(savePath, ind=startInd)

    # Load the simulation parameters to see if it is cylindrically symmetric
    params = loadSimulation(savePath)
    diagnostics = params["diagnostics"]

    # Load the field at each plane and populate the array
    field = np.zeros(Nz, dtype="complex128")
    z = np.zeros(Nz)
    for i in range(Nz):
        ind = i + startInd
        filename = os.path.join(savePath, "field_{}.h5".format(ind))
        field[i], attrs = _loadOnAxisPoint(filename, diagnostics, gridAttrs)
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
