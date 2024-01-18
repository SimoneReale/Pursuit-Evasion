import pulp as pl
solver_list = pl.listSolvers(onlyAvailable=True)

print(solver_list)

solver = pl.getSolver('CPLEX_PY')

print(solver)
