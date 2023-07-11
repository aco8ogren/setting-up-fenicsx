import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot, io

lower_left = [0,0,0]
upper_right = [1,1,1]
box_bounds = [lower_left,upper_right]

mesh_resolution = [1,1,1]

cell_type = mesh.CellType.hexahedron

domain = mesh.create_box(MPI.COMM_WORLD,box_bounds, mesh_resolution, cell_type)
V = fem.VectorFunctionSpace(domain, ("CG", 1))

# ============================================================

from petsc4py.PETSc import ScalarType

E = 200e6 # [Pa]
nu = 0.3 # [-]
rho = 1e3 # [kg/m^3]

assert(nu != 0.5)

lambda_ = E*nu/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))

def epsilon(u):
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0*mu*epsilon(u)

# gravity = 9.81
# traction = fem.Constant(domain, ScalarType((0.0, 0.0, 0.0)))

# ds = ufl.Measure("ds", domain=domain)

#dx = ufl.Measure("dx",domain=domain)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# f = fem.Constant(domain, ScalarType((0.0, 0.0, -rho*gravity)))
# L = ufl.inner(f, v) * ufl.dx + ufl.inner(traction, v) * ds
# #L = ufl.inner(f, v) * dx + ufl.inner(traction, v) * ds

k_form = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
m_form = rho*ufl.inner(u,v)*ufl.dx

K = fem.petsc.assemble_matrix(fem.form(k_form))
K.assemble()

M = fem.petsc.assemble_matrix(fem.form(m_form))
M.assemble()

# =========================================================

viewer_K = PETSc.Viewer().createASCII("ex_get_element_stiffness.txt", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)
viewer_K(K)

viewer_M = PETSc.Viewer().createASCII("ex_get_element_mass.txt", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)
viewer_M(M)

# =========================================================

ki, kj, kv = K.getValuesCSR()

mi, mj, mv = M.getValuesCSR()

import scipy

K_scipy_sparse = scipy.sparse.csr_matrix((kv,kj,ki))
M_scipy_sparse = scipy.sparse.csr_matrix((mv,mj,mi))

matfile_dict = {'K':K_scipy_sparse,'M':M_scipy_sparse}
scipy.io.savemat('ex_element_stiffness_and_mass.mat',matfile_dict)

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')

# grid = np.random.random((10,10))

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))

# ax1.imshow(grid, extent=[0,100,0,1])
# ax1.set_title('Default')

# ax2.imshow(grid, extent=[0,100,0,1], aspect='auto')
# ax2.set_title('Auto-scaled Aspect')

# ax3.imshow(grid, extent=[0,100,0,1], aspect=100)
# ax3.set_title('Manually Set Aspect')

# plt.tight_layout()
# plt.show()

aaaa = 1