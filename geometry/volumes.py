import numpy as np


class BuildVolume(object):
    "This is an abstract class for Build Volumes."
    def __init__(self):
        raise NotImplementedError()
    def extents(self, tol=0):
        raise NotImplementedError()
    def is_inside(self, x, tol=0):
        raise NotImplementedError()
    def generate(self, geom):
        raise NotImplementedError()


class CylindricalBuildVolume(BuildVolume):
    """This implements a cylindrical build volume defined by a `radius` and
       a `height`. The cylinder has its base centered at the origin.

       Example call:
       bv = CylindricalBuildVolume(height=20*1e-3, radius=20*1e-3)
       """
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height
    def extents(self, tol=0):
        low = np.array([-self.radius, -self.radius, 0])
        high = np.array([self.radius, self.radius, self.height]) + tol
        return np.vstack([low - tol, high + tol])
    def centroid(self):
        return np.array([0, 0, self.height/2])
    def is_inside(self, x, tol=0):
        return np.logical_and(np.linalg.norm(x[:, :2], axis=1)<self.radius+tol,
                              np.logical_and(x[:, 2]>0-tol, x[:, 2]<self.height+tol))
    def generate(self, geom):
        return geom.add_cylinder([0.0, 0.0, 0.0], [0.0, 0.0, self.height], self.radius)
