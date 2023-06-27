import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot, io

lower_left = [0,0,0]
upper_right = [10, 5, 5]
box_bounds = [lower_left,upper_right]

mesh_resolution = [5, 3, 3]

cell_type = mesh.CellType.hexahedron

domain = mesh.create_box(MPI.COMM_WORLD,box_bounds, mesh_resolution, cell_type)
V = fem.VectorFunctionSpace(domain, ("CG", 2))

# ============================================================

from petsc4py.PETSc import ScalarType

lambda_ = 1.0
mu = 1.0

def epsilon(u):
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0*mu*epsilon(u)

density = 1.0
gravity = 9.81
traction = fem.Constant(domain, ScalarType((0.0, 0.0, 0.0)))

ds = ufl.Measure("ds", domain=domain)

#dx = ufl.Measure("dx",domain=domain)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, ScalarType((0.0, 0.0, -density*gravity)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(traction, v) * ds
#L = ufl.inner(f, v) * dx + ufl.inner(traction, v) * ds

K = fem.petsc.assemble_matrix(fem.form(a))

K.assemble()

# fem.petsc.assemble_matrix(a)

# fem.assemble(a)

# fem.petsc.assemble(a)

# problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# uh = problem.solve()

# viewer = PETSc.Viewer().createASCII("ex_write_stiffness_matrix.txt", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)

# viewer(A)

# with io.XDMFFile(domain.comm, "hexahedral_mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)

# =========================================================

import sys, slepc4py
slepc4py.init(sys.argv)

from slepc4py import SLEPc

from dolfinx.io import XDMFFile

# Create and configure eigenvalue solver
N_eig = 10
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setDimensions(N_eig)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
st = SLEPc.ST().create(MPI.COMM_WORLD)
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.1)
st.setFromOptions()
eigensolver.setST(st)
eigensolver.setOperators(K)
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

aaaa = 1