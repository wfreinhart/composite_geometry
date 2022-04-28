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
        # create the matrix mesh
        matrix = geom.boolean_difference(build_volume, reinforcements)
        gmsh_matrix = geom.generate_mesh()
        # create the reinforcement mesh
        build_volume = bv.generate(geom)
        all_reinforcements = geom.boolean_difference(build_volume, matrix)
        gmsh_reinf = geom.generate_mesh()

    return gmsh_matrix, gmsh_reinf


def combine_meshes(mesh_A, mesh_B, prec_digits=8):
    stack = np.vstack([mesh_A.points.round(prec_digits), mesh_B.points.round(prec_digits)])
    new_points, idx, inv = np.unique(stack, axis=0, return_inverse=True, return_index=True)

    new_tri_A = inv[mesh_A.cells_dict['triangle']]  # gmsh_matrix.cells_dict['triangle'][stack_idx[inv]]
    new_tri_B = inv[mesh_B.cells_dict['triangle'] + mesh_A.points.shape[0]]
    new_tri = np.vstack([new_tri_A, new_tri_B])

    new_tet_A = inv[mesh_A.cells_dict['tetra']]
    new_tet_B = inv[mesh_B.cells_dict['tetra'] + mesh_A.points.shape[0]]
    new_tet = np.vstack([new_tet_A, new_tet_B])

    mat_id = np.ones_like(new_tet[:, 0])
    mat_id[:new_tet_A.shape[0]] = 0

    tetra_mesh = meshio.Mesh(
        points=new_points,
        cells={'triangle': new_tri, 'tetra': new_tet},
        # cell_data={"material": [[], mat_id]},  # TODO
    )

    return tetra_mesh, mat_id


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
