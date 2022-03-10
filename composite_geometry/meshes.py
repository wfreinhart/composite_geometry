import meshio
import pygmsh
import trimesh
import numpy as np


def generate_meshes(bv, rf, length_scale=3*1e-3):
    # build the matrix and full meshes:
    # NOTE: this will require further modification when we include voids
    with pygmsh.occ.Geometry() as geom:
        geom.characteristic_length_max = length_scale
        # create the reinforcements
        build_volume = bv.generate(geom)

        if rf is None:  # return the full build volume
            gmsh_matrix = geom.generate_mesh()
            gmsh_full = geom.generate_mesh()
            return gmsh_matrix, gmsh_full

        if type(rf) is list:
            reinforcements = []
            for it in rf:
                reinforcements += it.generate(geom, bv)
        else:
            reinforcements = rf.generate(geom, bv)
        # create the matrix
        matrix = geom.boolean_difference(build_volume, reinforcements)
        gmsh_matrix = geom.generate_mesh()
        # create the full mesh
        build_volume = bv.generate(geom)
        everything = geom.boolean_union([build_volume] + matrix)
        gmsh_full = geom.generate_mesh()

    return gmsh_matrix, gmsh_full


def assign_materials(gmsh_matrix, gmsh_full):
    """use the trimesh package to evaluate whether centroids are inside or outside a sub-mesh
    Inputs:
        gmsh_matrix : the mesh defining the subdomain (for one material)
        gmsh_full : the mesh defining the full problem domain
    Returns:
        mat_id (N,) int : binary material ID (does not work for multiple materials yet)
    """
    centroids = gmsh_full.points[gmsh_full.cells_dict['tetra']].mean(axis=1)
    for cell in gmsh_matrix.cells:
        if cell.type == "triangle":
            triangles = cell.data
    tm = trimesh.Trimesh(vertices=gmsh_matrix.points, faces=triangles)
    sdf = trimesh.proximity.signed_distance(tm, centroids)
    return np.logical_not(sdf > 0).astype(int)  # the SDF uses positive to indicate interior (https://trimsh.org/trimesh.proximity.html#trimesh.proximity.signed_distance)


def is_watertight(this_gmsh):
    for cell in this_gmsh.cells:
        if cell.type == "triangle":
            triangles = cell.data
    tm_build = trimesh.Trimesh(vertices=this_gmsh.points, faces=triangles)
    return tm_build.is_watertight


def write_xdmf(msh, fname):
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif cell.type == "tetra":
            tetra_cells = cell.data

    tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})

    # Save mesh file as .xdmf
    meshio.write(fname, tetra_mesh)
