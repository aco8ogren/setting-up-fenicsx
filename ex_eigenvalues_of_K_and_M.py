import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot, io

lower_left = [0,0,0]
upper_right = [10, 5, 5]
box_bounds = [lower_left,upper_right]

mesh_resolution = [10, 10, 10]

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

if evs > 0:
    with XDMFFile(MPI.COMM_WORLD, "eigenvectors.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        for i in range (min(N_eig, evs)):
            l = eigensolver.getEigenpair(i, vr, vi)
            freq = np.sqrt(l.real)/2/np.pi
            print(f"Mode {i}: {freq} Hz")
            u_output.x.array[:] = vr
            xdmf.write_function(u_output, i)