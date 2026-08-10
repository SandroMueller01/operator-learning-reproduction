import dolfin as df
print("dolfin OK", df.__version__)
import mshr
print("mshr OK")
domain = mshr.Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0))
mesh = mshr.generate_mesh(domain, 10)
print("mesh cells:", mesh.num_cells())
