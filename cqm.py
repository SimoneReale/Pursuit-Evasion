import networkx as nx
from dimod import ConstrainedQuadraticModel, Binary, quicksum, BinaryArray
from dwave.system import LeapHybridCQMSampler
from dwave.preprocessing.presolve import Presolver
from utils import n_of_nodes, n_time, n_rows, n_cols, n_preys
from random import randint, choice
import dimod
from time_profiling import profile, print_prof_data, clear_prof_data
import sys
from alive_progress import alive_bar


def getSetOfMoves(starting_node: int, n_rows: int, n_columns: int):
    n_of_nodes = n_rows * n_columns
    pos = []
    # stay
    pos.append(starting_node)
    # up
    if starting_node - n_columns >= 0:
        pos.append(starting_node - n_columns)
    # down
    if starting_node + n_columns < n_of_nodes:
        pos.append(starting_node + n_columns)
    # right
    if (starting_node + 1) % n_columns != 0:
        pos.append(starting_node + 1)
    # left
    if starting_node % n_columns != 0:
        pos.append(starting_node - 1)

    return pos


def getAllPossibleTupleMovesSetTime(n_rows: int, n_columns: int, n_time: int):
    n_of_nodes = n_rows * n_columns
    move_dict = []

    for t in range(n_time):
        for x in range(n_of_nodes):
            # stay
            move_dict.append((x, x, t))
            # up
            if x - n_columns >= 0:
                move_dict.append((x, x - n_columns, t))
            # down
            if x + n_columns < n_of_nodes:
                move_dict.append((x, x + n_columns, t))
            # right
            if (x + 1) % n_columns != 0:
                move_dict.append((x, x + 1, t))
            # left
            if x % n_columns != 0:
                move_dict.append((x, x - 1, t))

    return move_dict


def createPolicyUnvisitedPath(
    start: int, length: int, n_rows: int, n_columns: int
) -> list[int]:
    length += 1
    unvisited_nodes = set(range(n_rows * n_columns))
    unvisited_nodes.remove(start)
    path = [start]

    for _ in range(length - 1):
        moves = set(getSetOfMoves(path[-1], n_rows, n_columns))
        intersection_with_unvisited = moves.intersection(unvisited_nodes)

        moves = list(moves)
        if intersection_with_unvisited:
            chosen_move = choice(list(intersection_with_unvisited))
            unvisited_nodes.remove(chosen_move)
            path.append(chosen_move)

        else:
            chosen_move = choice(moves)
            path.append(chosen_move)

    return [(path[t], path[t + 1], t) for t in range(len(path) - 1)]


def create_unvisited_policy_prey_path(
    indices, starting_node: int, length: int, n_rows: int, n_cols: int
):
    prey_path = {x: 0 for x in indices}
    lst_positions = createPolicyUnvisitedPath(starting_node, length, n_rows, n_cols)
    for x in lst_positions:
        prey_path[x] = 1
    return prey_path


def create_unvisited_path_and_states(
    indices, starting_node: int, length: int, n_rows: int, n_cols: int
):
    prey_path = {x: 0 for x in indices}
    lst_states = []
    lst_positions = createPolicyUnvisitedPath(starting_node, length, n_rows, n_cols)
    for x in lst_positions:
        prey_path[x] = 1
        lst_states.append((x[0], x[2]))
    return prey_path, lst_states


def getAllPossibleTupleMovesSet(n_rows: int, n_columns: int):
    n_of_nodes = n_rows * n_columns
    move_list = []

    for x in range(n_of_nodes):
        # stay
        move_list.append((x, x))
        # up
        if x - n_columns >= 0:
            move_list.append((x, x - n_columns))
        # down
        if x + n_columns < n_of_nodes:
            move_list.append((x, x + n_columns))
        # right
        if (x + 1) % n_columns != 0:
            move_list.append((x, x + 1))
        # left
        if x % n_columns != 0:
            move_list.append((x, x - 1))
    return move_list


def getAllPossibleTupleMovesSetTime(n_rows: int, n_columns: int, n_time: int):
    n_of_nodes = n_rows * n_columns
    move_dict = []

    for t in range(n_time):
        for x in range(n_of_nodes):
            # stay
            move_dict.append((x, x, t))
            # up
            if x - n_columns >= 0:
                move_dict.append((x, x - n_columns, t))
            # down
            if x + n_columns < n_of_nodes:
                move_dict.append((x, x + n_columns, t))
            # right
            if (x + 1) % n_columns != 0:
                move_dict.append((x, x + 1, t))
            # left
            if x % n_columns != 0:
                move_dict.append((x, x - 1, t))

    return move_dict


@profile
def createCQM():
    print("\nBuilding constrained quadratic model...")

    nodes = set(range(n_of_nodes))
    times = set(range(n_time))
    indices = getAllPossibleTupleMovesSetTime(n_rows, n_cols, n_time)
    costs = {x: randint(1, 50) for x in indices}
    path_prey = [
        create_unvisited_policy_prey_path(
            indices, randint(1, n_of_nodes - 1), n_time, n_rows, n_cols
        )
        for _ in range(n_preys)
    ]

    cqm = ConstrainedQuadraticModel()

    vars = {(i[0], i[1], i[2]): Binary(f"x_{i[0]}_{i[1]}_{i[2]}") for i in indices}
    obj = quicksum(vars[x[0], x[1], x[2]] * costs[x] for x in indices)
    cqm.set_objective(obj)

    # faccio un solo passaggio per istante di tempo
    for t in range(n_time):
        cst = quicksum(
            vars[x[0], x[1], t] for x in getAllPossibleTupleMovesSet(n_rows, n_cols)
        )
        cqm.add_constraint(cst == 1, label=f"One pass per time {t}")

    print(f"Constraint OK: One pass per time instant")

    # constraint dello start
    cqm.add_constraint(vars[0, 0, 0] == 1, label=f"Start in (0,0)")
    print(f"Constraint OK: Starting Point")

    # constraint della intersection
    var = []
    for index_prey, p in enumerate(path_prey):
        for t in times:
            for i1 in nodes:
                for i2 in nodes:
                    for j1 in nodes:
                        for j2 in nodes:
                            if (
                                j1 == j2
                                and j1 in getSetOfMoves(i1, n_rows, n_cols)
                                and j2 in getSetOfMoves(i2, n_rows, n_cols)
                            ):
                                var.append(vars[i1, j1, t] * p[i2, j2, t])

        cqm.add_constraint(
            quicksum(var) >= 1, label=f"Intersection with target {index_prey}"
        )

    print(f"Constraint OK: Intersection")

    # constraint del movimento su nodi adiacenti
    for t in times.difference({0}):
        for i in nodes:
            for j in nodes:
                if j in getSetOfMoves(i, n_rows, n_cols):
                    cqm.add_constraint(
                        (
                            vars[i, j, t]
                            - quicksum(
                                vars[k, i, t - 1]
                                for k in nodes
                                if i in getSetOfMoves(k, n_rows, n_cols)
                            )
                        )
                        <= 0,
                        label=f"Only adjacent nodes {t, i, j}",
                    )
    print(f"Constraint OK: Only adjacent nodes")
    print("Model creation OK")
    return cqm, path_prey, costs


possible_moves = {x: getSetOfMoves(x, n_rows, n_cols) for x in range(n_of_nodes)}


@profile
def createAdvancedCQM():
    print("\nBuilding constrained quadratic model...")

    nodes = set(range(n_of_nodes))
    times = set(range(n_time))
    indices = getAllPossibleTupleMovesSetTime(n_rows, n_cols, n_time)
    costs = {x: randint(5, 15) for x in indices}
    lst_prey_path = [
        create_unvisited_policy_prey_path(
            indices, randint(1, n_of_nodes - 1), n_time, n_rows, n_cols
        )
        for _ in range(n_preys)
    ]

    cqm = ConstrainedQuadraticModel()

    vars_x = {(i[0], i[1], i[2]): Binary(f"x_{i[0]}_{i[1]}_{i[2]}") for i in indices}
    vars_u = {(i): Binary(f"u_{i}") for i in times}
    vars_z = {(p, t): Binary(f"z_{p}_{t}") for p in range(n_preys) for t in range(n_time)}
    vars_w = {(p, t): Binary(f"w_{p}_{t}") for p in range(n_preys) for t in range(n_time)}

    obj_func = quicksum(
        vars_x[i_j_t[0], i_j_t[1], i_j_t[2]] * costs[i_j_t] * vars_u[i_j_t[2]]
        for i_j_t in indices
    )

    cqm.set_objective(obj_func)

    # # faccio un solo passaggio per istante di tempo
    for t in range(n_time):
        cst = quicksum(
            vars_x[move_i_j[0], move_i_j[1], t]
            for move_i_j in getAllPossibleTupleMovesSet(n_rows, n_cols)
        )
        cqm.add_constraint(cst == 1, label=f"One pass per time {t}")

    print(f"Constraint OK: One pass per time instant")

    # constraint dello start
    cqm.add_constraint(vars_x[0, 1, 0] == 1, label=f"Start in 0 then 1")
    print(f"Constraint OK: Starting Point")

    # constraint della intersection
    print("Building the interception constraint:")
    with alive_bar(n_time * n_preys * (n_of_nodes**3)) as bar:
        for index_prey, prey_path in enumerate(lst_prey_path):
            for t in times:
                temp = []
                for i1 in nodes:
                    for i2 in nodes:
                        for j in nodes:
                            bar()
                            if j in possible_moves[i1] and j in possible_moves[i2]:
                                temp.append(vars_x[i1, j, t] * prey_path[i2, j, t])

                cqm.add_constraint(
                    quicksum(temp) - vars_z[index_prey, t] == 0,
                    label=f"Intersection with target {index_prey} at time {t}",
                )
    print(f"Constraint OK: Interception")

    # constraint interception propagation
    for t in times.difference({0}):
        for index_prey in range(n_preys):
            cqm.add_constraint(
                vars_w[index_prey, t - 1]
                - vars_w[index_prey, t]
                <= 0,
                label=f"Propagation at time {t} for prey {index_prey}",
            )
    print(f"Constraint OK: Interception propagation")

    # constraint interception multiple and single
    for t in times:
        for index_prey in range(n_preys):
            cqm.add_constraint(
                vars_w[index_prey, t]
                - vars_z[index_prey, t]
                >= 0,
                label=f"Multiple {t} for prey {index_prey}",
            )
    print(f"Constraint OK: Interception multiple and single")

    for prey_path in range(n_preys):
        vars_temp = []
        for t in times:
            vars_temp.append(vars_z[prey_path, t])
        cqm.add_constraint(
            quicksum(vars_temp) >= 1, label=f"At the end {prey_path} is intercepted"
        )
    print(f"Constraint OK: At the end there is interception ")

    for t in times:
        vars_temp = []
        for index_prey in range(n_preys):
            vars_temp.append(vars_w[index_prey, t])
        cqm.add_constraint(
            sum(vars_temp) - n_preys * (1 - vars_u[t]) >= 0,
            label=f"All intercepted at time {t}",
        )
    print("Constraint OK: All intercepted")

    # constraint del movimento su nodi adiacenti
    print("Building the adjacency constraint:")
    with alive_bar((n_time - 1) * (n_of_nodes**2)) as bar:
        for t in times.difference({0}):
            for i in nodes:
                for j in nodes:
                    bar()
                    if j in possible_moves[i]:
                        cqm.add_constraint(
                            (
                                vars_x[i, j, t]
                                - quicksum(
                                    vars_x[k, i, t - 1]
                                    for k in nodes
                                    if i in possible_moves[k]
                                )
                            )
                            <= 0,
                            label=f"Only adjacent nodes {t, i, j}",
                        )
    print(f"Constraint OK: Only adjacent nodes")
    print("Model creation OK")

    # presolve = Presolver(cqm)
    # presolve.apply()
    # reduced_cqm = presolve.detach_model()

    return cqm, lst_prey_path, costs


# Maximum memory usage: 304.75390625 MiB
# Function createCQM called 1 times.
# Execution time max: 240.955, average: 240.955

if __name__ == "__main__":
    cqm, path_prey, costs = createAdvancedCQM()
    original_stdout = sys.stdout
    with open("cqm.txt", "w") as f:
        sys.stdout = f
        print(f"Number of biases: {cqm}")
        sys.stdout = original_stdout  # Reset the standard output to its original value

    # mem_usage = memory_usage(createCQM)
    # print('Memory usage (in chunks of .1 seconds): %s MiB' % mem_usage)
    # print('Maximum memory usage: %s MiB' % max(mem_usage))
    # print_prof_data()
    # clear_prof_data()
