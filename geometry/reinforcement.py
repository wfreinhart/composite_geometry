import numpy as np
from scipy.spatial.transform import Rotation


class SphereReinforcement(object):
    """
    Example call:
    SphereReinforcement(radius=5 * 1e-3, lattice=BCCLattice(spacing=12 * 1e-3))
    """
    # TODO: provide an option for offset
    def __init__(self, radius, lattice):
        self.radius = radius
        self.lattice = lattice

    def generate(self, geom, build_volume):
        reinforcements = []
        lattice_points = self.lattice.generate(build_volume, tol=self.radius)
        for pt in lattice_points:
            reinforcements.append(geom.add_ball(pt, self.radius))
        return reinforcements


class LayerReinforcement(object):
    # TODO: add option for non-[0 0 1] layering direction
    def __init__(self, height, spacing):
        self.height = height
        self.spacing = spacing

    def generate(self, geom, build_volume):
        reinforcements = []
        low, high = build_volume.extents()
        layer_base = np.arange(0, high[2], self.spacing)
        for pt in layer_base:
            x0 = [low[0], low[1], pt]
            ext = [high[0]-low[0], high[1]-low[1], self.height]
            reinforcements.append(geom.add_box(x0, ext))
        return reinforcements


class UnequalLayerReinforcement(object):
    """
    Example call:
    bitmap = np.random.rand(20).round()
    UnequalLayerReinforcement(bitmap)
    """
    # TODO: add option for non-[0 0 1] layering direction
    def __init__(self, bitmap):
        # find breaks in bitmap representation
        r_start = np.argwhere(np.diff(bitmap) > 0).flatten()+1
        r_stop = np.argwhere(np.diff(bitmap) < 0).flatten()+1
        if bitmap[0] == 1:
            r_start = np.hstack([0, r_start])
        if bitmap[-1] == 1:
            r_stop = np.hstack([r_stop, len(bitmap)])
        self.layer_base = r_start / len(bitmap)
        self.height = (r_stop - r_start) / len(bitmap)

    def generate(self, geom, build_volume):
        reinforcements = []
        low, high = build_volume.extents()
        layer_base = self.layer_base * (high[2]-low[2]) - low[2]
        height = self.height * (high[2]-low[2]) - low[2]
        for i, pt in enumerate(layer_base):
            x0 = [low[0], low[1], pt]
            ext = [high[0]-low[0], high[1]-low[1], height[i]]
            reinforcements.append(geom.add_box(x0, ext))
        return reinforcements


class FiberReinforcement(object):
    """
    Inputs:
        radius
        grid `Grid` object
        rotation (Euler angle tuple)
    Returns:

    Example call:
    rf = FiberReinforcement(radius, HexGrid(spacing), ('x', 90))
    """
    def __init__(self, radius, grid, rotation):
        self.radius = radius
        self.grid = grid
        self.rotation = Rotation.from_euler(*rotation, degrees=True)
        perp = self.rotation.inv().as_matrix()
        direction = perp @ np.array([0, 0, 1])  # Grids are defined on (x,y)
        self.direction = direction / np.linalg.norm(direction)

    def generate(self, geom, build_volume):
        reinforcements = []
        grid_points = self.grid.generate(build_volume, tol=self.radius, as_3d=True)
        grid_points = grid_points @ self.rotation.as_matrix()
        grid_points += build_volume.centroid() - np.mean(grid_points, axis=0)
        axis = self.direction
        low, high = build_volume.extents()
        length = np.linalg.norm(high - low)
        for pt in grid_points:
            reinforcements.append(geom.add_cylinder(pt - axis*length,
                                                    axis*2*length, self.radius))
        return reinforcements
