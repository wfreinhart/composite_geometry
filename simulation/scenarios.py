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


def stress_strain_curve(strain):
    # TODO - implement this function for convenience
    pass


def analytical_uniaxial_stress(stress_model, trueStrainVec):
    stress = np.zeros(len(trueStrainVec))
    for i in range(len(trueStrainVec)):
        # lam1 = np.exp(trueStrainVec[i])  # TODO: reconcile this choice with the simulation
        lam1 = 1. + trueStrainVec[i]
        calcS22Abs = lambda x : np.abs(stress_model([lam1, x.tolist()[0], x.tolist()[0]])[1, 1])
        lam2 = optimize.fmin(calcS22Abs, x0=1/np.sqrt(lam1), xtol=1e-9, ftol=1e-9, disp=False)
        stress[i] = stress_model([lam1, lam2.tolist()[0], lam2.tolist()[0]])[0, 0]
    return stress
