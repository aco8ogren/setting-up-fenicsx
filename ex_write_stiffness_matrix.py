import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot, io

lower_left = [0,0,0]
upper_right = [1, 1, 1]
box_bounds = [lower_left,upper_right]

mesh_resolution = [2, 1, 1]

cell_type = mesh.CellType.hexahedron

domain = mesh.create_box(MPI.COMM_WORLD,box_bounds, mesh_resolution, cell_type)
V = fem.VectorFunctionSpace(domain, ("CG", 1))

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

A = fem.petsc.assemble_matrix(fem.form(a))

A.assemble()

# fem.petsc.assemble_matrix(a)

# fem.assemble(a)

# fem.petsc.assemble(a)

# problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# uh = problem.solve()

viewer = PETSc.Viewer().createASCII("ex_write_stiffness_matrix.txt", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)

viewer(A)

# with io.XDMFFile(domain.comm, "hexahedral_mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)