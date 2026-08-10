"""Phase 0 smoke test: confirm FEniCS 2019.1.0 (dolfin) can solve Poisson."""

import dolfin as df

mesh = df.UnitSquareMesh(8, 8)
V = df.FunctionSpace(mesh, "CG", 1)
u = df.TrialFunction(V)
v = df.TestFunction(V)
f = df.Constant(1.0)
a = df.dot(df.grad(u), df.grad(v)) * df.dx
L = f * v * df.dx
bc = df.DirichletBC(V, df.Constant(0.0), "on_boundary")
sol = df.Function(V)
df.solve(a == L, sol, bc)
print("DOLFIN_OK, norm=", df.norm(sol))
