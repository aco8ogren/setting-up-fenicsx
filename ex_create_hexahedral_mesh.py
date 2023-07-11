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
V = fem.VectorFunctionSpace(domain, ("CG", 2))

with io.XDMFFile(domain.comm, "hexahedral_mesh.xdmf", "w") as xdmf_file:
    xdmf_file.write_mesh(domain)

with io.VTKFile(domain.comm, "hexahedral_mesh.vtk", "w") as vtk_file:
    vtk_file.write_mesh(domain)
    # vtk_file.write_function([u._cpp_object])