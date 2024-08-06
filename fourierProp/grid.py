import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift


class Grid:
    def __init__(self, Nx, Ny, X, Y):
        self.Nx = Nx
        self.Ny = Ny
        self.X = X
        self.Y = Y

        # Create the x and y grid
        self.x, self.dx = np.linspace(
            -X / 2, X / 2, Nx, endpoint=False, retstep=True, dtype="double"
        )
        self.y, self.dy = np.linspace(
            -Y / 2, Y / 2, Ny, endpoint=False, retstep=True, dtype="double"
        )
        # XXX This creates another full size array, could be removed for memory
        self.r = np.sqrt(self.x[:, None] ** 2 + self.y[None, :] ** 2)

        # Calculate the fourier space equivalent of the grid
        dx = self.dx
        dy = self.dy
        self.kx_unshifted = 2 * np.pi * fftfreq(Nx, dx)
        self.ky_unshifted = 2 * np.pi * fftfreq(Ny, dy)
        self.fx = fx = fftshift(fftfreq(Nx, dx))
        self.fy = fy = fftshift(fftfreq(Ny, dy))
        self.dfx = fx[1] - fx[0]
        self.dfy = fy[1] - fy[0]


def gridFromFileData(attrs):
    Nx = attrs["x"]["Nx"]
    X = attrs["x"]["X"]
    Ny = attrs["y"]["Ny"]
    Y = attrs["y"]["Y"]
    return Grid(Nx, Ny, X, Y)


class CylGrid:
    def __init__(self, Nr, R, m=[0]):
        self.Nr = Nr
        self.R = R
        self.M

        # Create the r grid
        self.r, self.dr = np.linspace(
            0.0, R, endpoint=False, retstep=True, dtype="double"
        )

        # Calculate the fourier space equivalent of the grid
        dr = self.dr
        for i, order in enumerate(m):
            alpha = jn_zeros(order, Nr + 1)
            alpha_Np1 = alpha[-1]
            alpha = alpha[:-1]
        # self.kr_unshifted = 2 * np.pi * fftfreq(Nx, dx)
        # self.fr = fr = fftshift(fftfreq(Nx, dx))
        # self.dfr = fr[1] - fr[0]
