import fenics as fem
import numpy as np


class ConstitutiveModel(object):
    required_props = ('E', 'nu')

    def __init__(self, msh, mat_id, properties):

        if any([it not in properties for it in self.required_props]):
            raise ValueError(f'{self.required_props} values are required in properties dict.')

        # Lame's constants
        E = np.array(properties['E'])
        nu = np.array(properties['nu'])

        # TODO: move this to a dedicated preproc method
        properties['lambda'] = E * nu / (1 + nu) / (1 - 2 * nu)
        properties['mu'] = E / 2 / (1 + nu)

        V0 = fem.FunctionSpace(msh, 'DG', 0)

        self.properties_ = {}
        for k, v in properties.items():
            if k in ('E', 'nu'):
                continue  # don't record these
            prop = np.array(v)
            if len(prop.shape) > 0:
                var = fem.Function(V0, name=k)
                var.vector().set_local(prop[mat_id])
                var.vector().apply("insert")
                self.properties_[k] = var
            else:
                self.properties_[k] = v

    def sigma(self, u):
        "Compute the stress."
        pass

    def psi(self, u):
        "Compute the Helmholtz free energy."
        pass

    def analytical_stress(self, stretch):
        "Compute the stress from an analytical expression based on stretches"
        pass


class NeoHookeanModel(ConstitutiveModel):
    required_props = ('E', 'nu')

    def psi(self, u):
        """Stored strain energy density (compressible neo-Hookean model)"""

        # mu = self.material.mu_
        # lmbda = self.material.lambda_
        lmbda = self.properties_['lambda']
        mu = self.properties_['mu']
        kappa = lmbda + 2 / 3 * mu

        # Kinematics
        I = fem.Identity(3)  # Identity tensor
        F = I + fem.grad(u)  # Deformation gradient
        C = F.T * F  # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        J = fem.det(F)
        Ic = fem.tr(C)
        Icbar = J ** (-2 / 3) * Ic

        return (mu / 2) * (Icbar - 3) + kappa / 2 * (J - 1) ** 2

    def sigma(self, u):
        # material properties
        lmbda = self.properties_['lambda']
        mu = self.properties_['mu']
        kappa = lmbda + 2 / 3 * mu

        # Kinematics
        I = fem.Identity(3)  # Identity tensor
        F = I + fem.grad(u)  # Deformation gradient
        B = F * F.T  # Left Cauchy-Green tensor
        C = F.T * F  # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        # Ic = fem.tr(C)
        J = fem.det(F)

        # additional components
        Fstar = J ** (-1 / 3) * F
        Bstar = Fstar * Fstar.T
        # Icbar = J**(-2/3)*Ic

        return (mu / J) * fem.dev(Bstar) + kappa * (J - 1) * I

    def analytical_stress(self, stretch):
        lmbda = self.properties_['lambda']
        mu = self.properties_['mu']
        kappa = lmbda + 2 / 3 * mu
        # TODO: fix this to quiet scipy optimize calls with size-1 arrays
        F = np.array([[stretch[0], 0, 0], [0, stretch[1], 0], [0, 0, stretch[2]]], dtype=float)
        J = np.linalg.det(F)
        Fstar = J ** (-1 / 3) * F
        bstar = np.dot(Fstar, Fstar.T)
        dev_bstar = bstar - np.trace(bstar) / 3 * np.eye(3)
        sigma = mu / J * dev_bstar + kappa * (J - 1) * np.eye(3)
        return sigma


class YeohModel(ConstitutiveModel):
    required_props = ('E', 'nu', 'C1', 'C2', 'C3')

    # constraints are C1 > 0, C2 < 0, C3 > 0
    # note: for consistency with linear elasticity under small strains,
    # we require that mu = 2 C1, enforced as C1 = mu / 2

    def psi(self, u):
        """Stored strain energy density (compressible Mooney-Rivlin model)"""

        # material properties
        lmbda = self.properties_['lambda']
        mu = self.properties_['mu']
        kappa = lmbda + 2 / 3 * mu

        # model-specific material properties
        C1 = self.properties_['C1']
        # C1 = 0.5 * self.properties_['mu']
        C2 = self.properties_['C2']
        C3 = self.properties_['C3']

        # Kinematics
        I = fem.Identity(3)  # Identity tensor
        F = I + fem.grad(u)  # Deformation gradient
        C = F.T * F  # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        J = fem.det(F)
        Ic = fem.tr(C)
        Icbar = J ** (-2 / 3) * Ic

        # return (mu/2)*(Icbar - 3) + kappa/2*(J-1)**2
        return C1 * (Icbar - 3) + C2 * (Icbar - 3) ** 2 + C3 * (Icbar - 3) ** 3 + 0.5 * kappa * (J - 1) ** 2

    def sigma(self, u):
        # material properties
        lmbda = self.properties_['lambda']
        mu = self.properties_['mu']
        kappa = lmbda + 2 / 3 * mu

        # model-specific material properties
        C1 = self.properties_['C1']
        # C1 = 0.5 * self.properties_['mu']
        C2 = self.properties_['C2']
        C3 = self.properties_['C3']

        # Kinematics
        I = fem.Identity(3)  # Identity tensor
        F = I + fem.grad(u)  # Deformation gradient
        C = F.T * F  # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        J = fem.det(F)
        Ic = fem.tr(C)
        Icbar = J ** (-2 / 3) * Ic

        # Additional components
        Fstar = J ** (-1 / 3) * F
        Bstar = Fstar * Fstar.T

        return (2 / J) * (C1 + 2 * C2 * (Icbar - 3) + 3 * C3 * (Icbar - 3) ** 2) * fem.dev(Bstar) + kappa * (J - 1) * I

    def analytical_stress(self, stretch):
        # material properties
        lmbda = self.properties_['lambda']
        mu = self.properties_['mu']
        kappa = lmbda + 2 / 3 * mu

        # model-specific material properties
        C1 = self.properties_['C1']
        # C1 = 0.5 * self.properties_['mu']
        C2 = self.properties_['C2']
        C3 = self.properties_['C3']

        F = np.array([[stretch[0], 0, 0], [0, stretch[1], 0], [0, 0, stretch[2]]], dtype=float)
        J = np.linalg.det(F)
        bstar = J ** (-2 / 3) * np.dot(F, F.T)
        dev_bstar = bstar - np.trace(bstar) / 3 * np.eye(3)
        I1s = np.trace(bstar)
        sigma = 2 / J * (C1 + 2 * C2 * (I1s - 3) + 3 * C3 * (I1s - 3) ** 2) * dev_bstar + kappa * (J - 1) * np.eye(3)
        return sigma