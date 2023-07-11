import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot, io

lower_left = [0,0,0]
upper_right = [1,1,1]
box_bounds = [lower_left,upper_right]

mesh_resolution = [4,4,4]

cell_type = mesh.CellType.hexahedron

domain = mesh.create_box(MPI.COMM_WORLD,box_bounds, mesh_resolution, cell_type)
V = fem.VectorFunctionSpace(domain, ("CG", 1))

# ============================================================

Q = fem.FunctionSpace(domain, ("DG", 0)) # This space is a discontinuous space, rather than a continuous space

def Omega_0(x):
    return x[1] <= 0.5

def Omega_1(x):
    return x[1] >= 0.5

lambda_ = fem.Function(Q)
cells_0 = mesh.locate_entities(domain, domain.topology.dim, Omega_0)
cells_1 = mesh.locate_entities(domain, domain.topology.dim, Omega_1)

E_0 = 200e6
E_1 = 200e9

# E = 200e6 # [Pa] # Define E as fem.Function on Q instead
nu = 0.3 # [-]
rho = 1e3 # [kg/m^3]

lambda_.x.array[cells_0] = np.full_like(cells_0, E_0*nu/((1+nu)*(1-2*nu)), dtype=PETSc.ScalarType)
lambda_.x.array[cells_1] = np.full_like(cells_1, E_1*nu/((1+nu)*(1-2*nu)), dtype=PETSc.ScalarType)
# ============================================================

from petsc4py.PETSc import ScalarType

# E = 200e6 # [Pa] # Define E as fem.Function on Q instead
# nu = 0.3 # [-]
# rho = 1e3 # [kg/m^3]

assert(nu != 0.5)

# lambda_ = E*nu/((1+nu)*(1-2*nu))
mu = E_0/(2*(1+nu))

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

# viewer_K = PETSc.Viewer().createASCII("ex_get_element_stiffness.txt", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)
# viewer_K(K)

# viewer_M = PETSc.Viewer().createASCII("ex_get_element_mass.txt", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)
# viewer_M(M)

# =========================================================

# ki, kj, kv = K.getValuesCSR()

# mi, mj, mv = M.getValuesCSR()

# import scipy

# K_scipy_sparse = scipy.sparse.csr_matrix((kv,kj,ki))
# M_scipy_sparse = scipy.sparse.csr_matrix((kv,kj,ki))

# matfile_dict = {'K':K_scipy_sparse,'M':M_scipy_sparse}
# scipy.io.savemat('ex_element_stiffness_and_mass.mat',matfile_dict)

# =========================================================

import sys, slepc4py
slepc4py.init(sys.argv)

from slepc4py import SLEPc

from dolfinx.io import XDMFFile

# Create and configure eigenvalue solver
N_eig = 20
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setDimensions(N_eig)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
st = SLEPc.ST().create(MPI.COMM_WORLD)
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.1)
st.setFromOptions()
eigensolver.setST(st)
eigensolver.setOperators(K,M)
eigensolver.setFromOptions()

# Compute eigenvalue-eigenvector pairs
eigensolver.solve()
evs = eigensolver.getConverged()
vr, vi = K.getVecs()
u_output = fem.Function(V)
u_output.name = "Eigenvector"
print( "Number of converged eigenpairs %d" % evs )

# E_output = fem.Function(Q)
# E_output.name = "Elastic modulus"


if evs > 0:
    with XDMFFile(MPI.COMM_WORLD, "subdomain_eigenvectors.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        # xdmf.write_function(lambda_)
        for i in range (min(N_eig, evs)):
            l = eigensolver.getEigenpair(i, vr, vi)
            freq = np.sqrt(l.real)/2/np.pi
            print(f"Mode {i}: {freq} Hz")
            u_output.x.array[:] = vr
            xdmf.write_function(u_output, i)
            

    with XDMFFile(MPI.COMM_WORLD, "subdomain_material_properties.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        lambda_.name = "First lame constant"
        xdmf.write_function(lambda_)

aaaa = 1