import fenics as fem
import numpy as np
from scipy import optimize


class Problem(object):
    def __init__(self, msh, model, solver):
        # TODO: link Solver to Model implicitly
        pass

    def solve(self, input):
        return self.solver(self.u, self.v, self.du, self.bcs, self.model)


def hyperelastic_uniaxial_compression(delta_z, msh, model, u_init=None, friction=True, q_deg=None):

    # --------------------
    # Function spaces (superseded by the mesh only)
    # --------------------
    V = fem.VectorFunctionSpace(msh, "Lagrange", 1)  # displacement function space
    u_tr = fem.TrialFunction(V)
    u_test = fem.TestFunction(V)

    # Define functions
    du = fem.TrialFunction(V)  # Incremental displacement
    v = fem.TestFunction(V)  # Test function
    u = fem.Function(V)  # Displacement from previous iteration

    if u_init is not None:
        u.vector().set_local(u_init)

    # --------------------
    # Boundary conditions (superseded by the mesh and loading)
    # --------------------

    # Mark boundary subdomians
    def bot(x, on_boundary):
        return (on_boundary and fem.near(x[2], 0.0))

    def top(x, on_boundary):
        return (on_boundary and fem.near(x[2], msh.coordinates()[:, 2].max()))

    if friction:
        bc_bot = fem.DirichletBC(V, fem.Constant((0.0, 0.0, 0.0)), bot)  # fixed bottom
        bc_top = fem.DirichletBC(V, fem.Constant((0.0, 0.0, delta_z)), top)  # fixed top
    else:
        bc_bot = fem.DirichletBC(V.sub(2), 0.0, bot)  # fixed bottom z only
        bc_top = fem.DirichletBC(V.sub(2), delta_z, top)  # fixed top z only

    bcs = [bc_bot, bc_top]

    # ------------------------
    # Hyperelastic solver part (superseded by mesh, bcs, and material model)
    # ------------------------

    # Total potential energy
    if q_deg is None:
        Pi = model.psi(u) * fem.dx  # - dot(B, u)*dx # - dot(T, u)*ds(1)
    else:
        Pi = model.psi(u) * fem.dx(metadata={'quadrature_degree': q_deg})

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = fem.derivative(Pi, u, v)

    # Compute Jacobian of F
    J = fem.derivative(F, u, du)

    # Optimization options for the form compiler
    fem.parameters["form_compiler"]["cpp_optimize"] = True
    # additional options demonstrated here: https://fenicsproject.org/olddocs/dolfin/2016.2.0/python/demo/documented/hyperelasticity/python/demo_hyperelasticity.py.html

    problem = fem.NonlinearVariationalProblem(F, u, bcs, J)
    solver = fem.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['maximum_iterations'] = 10

    solver.solve()

    return u


def get_reaction_force_area(u, mesh, model, dim=2, use_pk=False):

    mesh_d = mesh

    boundary_markers = fem.MeshFunction("size_t", mesh_d, mesh_d.topology().dim() - 1, 0)

    class TopBoundary(fem.SubDomain):
        def inside(self, x, on_boundary):
            return fem.near(x[2], mesh_d.coordinates()[:, 2].max()) and on_boundary

    # Mark the top boundary
    top_boundary = TopBoundary()
    boundary_markers.set_all(0)  # Initialize to 0
    top_boundary.mark(boundary_markers, 1)  # Mark the top boundary with 1

    if use_pk:
        #
        I = fem.Identity(3)  # Identity tensor
        F = I + fem.grad(u)  # Deformation gradient
        J = fem.det(F)
        sigma = model.sigma(u)
        P = J * sigma * fem.inv(F)
        # Compute the reaction force
        ds = fem.Measure("ds", domain=mesh_d, subdomain_data=boundary_markers)
        n = fem.FacetNormal(mesh_d)
        rz = [fem.assemble(fem.dot(P, n)[i] * ds(1)) for i in range(3)]
    else:
        # Compute the reaction force
        sigma = model.sigma(u)
        ds = fem.Measure("ds", domain=mesh_d, subdomain_data=boundary_markers)
        n = fem.FacetNormal(mesh_d)
        rz = [fem.assemble(fem.dot(sigma, n)[i] * ds(1)) for i in range(3)]

    surf = fem.assemble(fem.Constant(1.0) * ds(1))

    return rz, surf


def get_area(u, mesh, model, dim=2):
    mesh_d = fem.Mesh(mesh)

    boundary_markers = fem.MeshFunction("size_t", mesh_d, mesh_d.topology().dim() - 1, 0)

    class TopBoundary(fem.SubDomain):
        def inside(self, x, on_boundary):
            return fem.near(x[2], mesh_d.coordinates()[:, 2].max()) and on_boundary

    # Mark the top boundary
    top_boundary = TopBoundary()
    boundary_markers.set_all(0)  # Initialize to 0
    top_boundary.mark(boundary_markers, 1)  # Mark the top boundary with 1

    # Deform the mesh
    fem.ALE.move(mesh_d, u)

    # Measure the area of the top surface
    ds = fem.Measure("ds", domain=mesh_d, subdomain_data=boundary_markers)
    surf = fem.assemble(fem.Constant(1.0) * ds(1))

    return surf


def generate_force_curve(model, mesh, bv, target_strain, max_stress=10e6, interp_mode="continue", alpha=0.50):

    u_sols = []

    force = np.zeros_like(target_strain) * np.nan

    pbar = tqdm.tqdm(enumerate(target_strain), total=len(target_strain))
    for i, eps in pbar:

        u_init = None
        if interp_mode == "continue":
            if i > 1:
                u_init = u_sols[i-1]
        elif interp_mode == "taylor":
            # nonlinear warm start
            if i == 1:    # do order-zero taylor expansion
                u_init = u_sols[0]
            elif i == 2:  # do order-one taylor expansion
                step = (u_sols[i-1] - u_sols[i-2])
                u_init = u_sols[i-1] + alpha * step
            elif i == 3:  # do order-two taylor expansion
                step = (u_sols[i-1] - u_sols[i-2]) + (u_sols[i-1] - 2*u_sols[i-2] + u_sols[i-3])
                u_init = u_sols[i-1] + alpha * step
            elif i == 4:  # do order-three taylor expansion
                step = (u_sols[i-1] - u_sols[i-2]) + (u_sols[i-1] - 2*u_sols[i-2] + u_sols[i-3]) + (u_sols[i-1] - 3*u_sols[i-2] + 3*u_sols[i-3] - u_sols[i-4])
                u_init = u_sols[i-1] + alpha * step
            elif i >= 5:  # do order-four taylor expansion
                step = (u_sols[i-1] - u_sols[i-2]) + (u_sols[i-1] - 2*u_sols[i-2] + u_sols[i-3]) + (u_sols[i-1] - 3*u_sols[i-2] + 3*u_sols[i-3] - u_sols[i-4]) + (u_sols[i-1] - 4*u_sols[i-2] + 6*u_sols[i-3] - 4*u_sols[i-4] + u_sols[i-5])
                u_init = u_sols[i-1] + alpha * step
        elif interp_mode == "proportional":
            if i > 1:
                u_init = u_sols[i-1] * target_strain[i] / target_strain[i-1]
        elif interp_mode == "secant":
            if i > 1:
                u_init = u_sols[i-1] - (target_strain[i-1] * (u_sols[i-1] - u_sols[i-2])) / (target_strain[i-1] - target_strain[i-2])

        disp = -eps*bv.height
        try:
            u = sim.scenarios.hyperelastic_uniaxial_compression(disp, mesh, model, u_init=u_init)
            u_sols.append(np.array(u.vector()))
        except RuntimeError:
            break  # save progress so far

        f, area = get_reaction_force_area(u, mesh, model)
        force[i] = f[2]
        
        stress = f[2] / area
        if np.abs(stress) > max_stress:
            break

    return force / area


def analytical_uniaxial_stress(stress_model, trueStrainVec):
    stress = np.zeros(len(trueStrainVec))
    for i in range(len(trueStrainVec)):
        # lam1 = np.exp(trueStrainVec[i])  # TODO: reconcile this choice with the simulation
        lam1 = 1. + trueStrainVec[i]
        calcS22Abs = lambda x : np.abs(stress_model([lam1, x.tolist()[0], x.tolist()[0]])[1, 1])
        lam2 = optimize.fmin(calcS22Abs, x0=1/np.sqrt(lam1), xtol=1e-9, ftol=1e-9, disp=False)
        stress[i] = stress_model([lam1, lam2.tolist()[0], lam2.tolist()[0]])[0, 0]
    return stress
