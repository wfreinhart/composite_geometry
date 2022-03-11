import numpy as np


class Lattice(object):
    def __init__(self, spacing, angles, unit_cell):
        self.spacing = spacing
        self.angles = angles
        self.unit_cell = unit_cell

    def generate(self, build_volume, tol=0):

        # get the lattice constants
        a, b, c = self.spacing

        # get the angles
        alpha, beta, gamma = [float(x)/180*np.pi for x in self.angles]

        # compute the lattice vectors
        A = a * np.array([1, 0, 0])
        B = b * np.array([np.cos(gamma), np.sin(gamma), 0])
        theta = (np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)
        C = c * np.array([np.cos(beta),
                        theta,
                        np.sqrt(1-np.cos(beta)**2-theta**2)])

        lattice_vectors = np.vstack([A, B, C])

        # compute extents to determine how many points to generate
        extents = build_volume.extents(tol=tol)
        low = int( np.floor(extents[0] / lattice_vectors.max(axis=0)).min() )
        high = int( np.ceil(extents[1] / lattice_vectors.max(axis=0)).max() )

        neighbor_cells = []
        for h in range(low, high):
            for k in range(low, high):
                for l in range(low, high):
                    new_cell = self.unit_cell + np.array([h, k, l])
                    neighbor_cells.append(new_cell)

        neighbor_cells = np.vstack(neighbor_cells)

        coordinates = np.matmul(neighbor_cells, lattice_vectors)
        coordinates = coordinates[build_volume.is_inside(coordinates, tol=tol)]

        return coordinates


class BCCLattice(Lattice):
    def __init__(self, spacing):
        primitive_cell = np.array([[0, 0, 0], [1/2, 1/2, 1/2]])
        super().__init__(spacing*np.ones(3), 90*np.ones(3), primitive_cell)


class Grid(object):
    """Creates a 2D grid on the (x,y) plane based on a specified tiling.
    Inputs:
    * spacing (2,) float : lattice parameters
    * angle (1,) float : primitive cell angle in degrees
    * unit_cell (N,2) float : primitive cell vectors to be tiled
    """
    def __init__(self, spacing, angle, unit_cell):
        self.spacing = spacing
        self.angle = angle*np.pi/180
        self.unit_cell = unit_cell

    def generate(self, build_volume, tol=0, as_3d=False):
        """Generates the grid points to fill a specified `build_volume`.
        Inputs:
        * build_volume : `BuildVolume` type object describing the space to fill
        * tol float (optional) : tolerance to the build volume domain
        * as_3d bool (optional) : concat zeros onto the coordinates to make 3D
        Returns:
        * coordinates (-,2) or (-,3) using `as_3d=True` : the grid coordinates
        """
        # get the lattice constants
        a, b = self.spacing

        # compute the lattice vectors
        A = a * np.array([1, 0])
        B = b * np.array([np.cos(self.angle), np.sin(self.angle)])

        lattice_vectors = np.vstack([A, B])

        # compute extents to determine how many points to generate
        extents = build_volume.extents(tol=tol)
        low = int( np.floor(extents[0, :2] / lattice_vectors.max(axis=1)).min() )
        high = int( np.ceil(extents[1, :2] / lattice_vectors.max(axis=1)).max() )

        neighbor_cells = []
        for h in range(low, high):
            for k in range(low, high):
                new_cell = self.unit_cell + np.array([h, k])
                neighbor_cells.append(new_cell)

        neighbor_cells = np.vstack(neighbor_cells)

        coordinates = neighbor_cells @ lattice_vectors
        coordinates_3d = np.hstack([coordinates, np.zeros_like(coordinates[:, [0]])+tol])
        coordinates = coordinates[build_volume.is_inside(coordinates_3d, tol=tol)]

        if as_3d:
            return np.hstack([coordinates, np.zeros_like(coordinates[:, [0]])])
        else:
            return coordinates


class HexGrid(Grid):
    "Implements a hexagonal grid from base class Grid"
    def __init__(self, spacing):
        super().__init__(spacing*np.ones(2), 60, np.zeros([1, 2]))


class SquareGrid(Grid):
    "Implements a square grid from base class Grid"
    def __init__(self, spacing):
        super().__init__(spacing*np.ones(2), 90, np.zeros([1, 2]))
