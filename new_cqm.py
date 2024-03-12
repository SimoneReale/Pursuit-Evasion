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

    return [(path[t], path[t + 1], t) for t in range(1, len(path) - 1)]


def createPreyPathStates(
    indices, start: int, length: int, n_rows: int, n_columns: int
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

    temp = [(path[t], path[t + 1], t) for t in range(1, len(path) - 1)]
    prey_path = {x: 0 for x in indices}
    for x in temp:
        prey_path[x] = 1

    return {index: n for index, n in enumerate(path)}, prey_path


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
    lst_states = {0: starting_node}
    lst_positions = createPolicyUnvisitedPath(starting_node, length, n_rows, n_cols)
    for x in lst_positions:
        prey_path[x] = 1
        lst_states[x[2]] = x[0]
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


def getAllPossibleTupleMovesSetTimeMinus0(n_rows: int, n_columns: int, n_time: int):
    n_of_nodes = n_rows * n_columns
    move_dict = []

    for t in range(1, n_time):
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


possible_moves = {x: getSetOfMoves(x, n_rows, n_cols) for x in range(n_of_nodes)}


@profile
def createStatesCQM():
    print("\nBuilding constrained quadratic model...")

    nodes = set(range(n_of_nodes))
    times_all = set(range(n_time))
    times_minus_0 = set(range(1, n_time))

    indices = getAllPossibleTupleMovesSetTimeMinus0(n_rows, n_cols, n_time)

    costs = {x: randint(5, 15) for x in indices}

    # lst_prey_path = [
    #     create_unvisited_path_and_states(
    #         indices, randint(1, n_of_nodes - 1), n_time, n_rows, n_cols
    #     )
    #     for _ in range(n_preys)
    # ]

    lst_prey_path = [
        createPreyPathStates(
            indices, randint(1, n_of_nodes - 1), n_time, n_rows, n_cols
        )
        for _ in range(n_preys)
    ]

    cqm = ConstrainedQuadraticModel()

    vars_x = {(i[0], i[1], i[2]): Binary(f"x_{i[0]}_{i[1]}_{i[2]}") for i in indices}
    vars_s = {(n, t): Binary(f"s_{n}_{t}") for n in nodes for t in times_all}
    vars_w = {
        (p, t): Binary(f"w_{p}_{t}") for p in range(n_preys) for t in range(n_time)
    }
    vars_u = {(t): Binary(f"u_{t}") for t in times_all}

    obj_func = quicksum(
        vars_x[i_j_t[0], i_j_t[1], i_j_t[2]] * costs[i_j_t] * vars_u[i_j_t[2]]
        for i_j_t in indices
    )

    cqm.set_objective(obj_func)

    # un solo nodo occupato per volta
    for t in range(n_time):
        cst = quicksum(vars_s[n, t] for n in nodes)
        cqm.add_constraint(cst == 1, label=f"One node per time {t}")

    for t in times_minus_0:
        for move_i_j in getAllPossibleTupleMovesSet(n_rows, n_cols):
            cqm.add_constraint(
                vars_x[move_i_j[0], move_i_j[1], t] - (vars_s[move_i_j[0], t - 1]) <= 0,
                label=f"Inference of movement start ({move_i_j[0]}, {move_i_j[1]}) at time {t}",
            )

    for t in times_minus_0:
        for move_i_j in getAllPossibleTupleMovesSet(n_rows, n_cols):
            cqm.add_constraint(
                vars_x[move_i_j[0], move_i_j[1], t] - (vars_s[move_i_j[1], t]) <= 0,
                label=f"Inference of movement end ({move_i_j[0]}, {move_i_j[1]}) at time {t}",
            )

    for t in times_minus_0:
        for move_i_j in getAllPossibleTupleMovesSet(n_rows, n_cols):
            cqm.add_constraint(
                vars_x[move_i_j[0], move_i_j[1], t]
                - (vars_s[move_i_j[1], t])
                - (vars_s[move_i_j[0], t - 1])
                >= 1,
                label=f"Inference ({move_i_j[0]}, {move_i_j[1]}) at time {t}",
            )

    # for t in times_minus_0:
    #     for move_i_j in getAllPossibleTupleMovesSet(n_rows, n_cols):
    #         cqm.add_constraint(
    #             vars_x[move_i_j[0], move_i_j[1], t]
    #             - (vars_s[move_i_j[0], t - 1] * vars_s[move_i_j[1], t])
    #             == 0,
    #             label=f"Inference of movement ({move_i_j[0]}, {move_i_j[1]}) at time {t}",
    #         )

    # for t in times_minus_0:
    #     for n in nodes:
    #         cqm.add_constraint(
    #             vars_s[n, t]
    #             - quicksum([vars_s[j, t - 1] for j in possible_moves[n]])
    #             <= 0,
    #             label=f"Sequential node {n} at time {t}",
    #         )

    for t in times_minus_0:
        for n in nodes:
            temp = []
            for move_i_j in getAllPossibleTupleMovesSet(n_rows, n_cols):
                if move_i_j[0] == n and n in possible_moves[move_i_j[1]]:
                    temp.append(
                        vars_x[move_i_j[0], move_i_j[1], t] + vars_s[move_i_j[1], t - 1]
                    )

        cqm.add_constraint(
            vars_s[n, t] - quicksum(temp) <= 0,
            label=f"Sequential node {n} at time {t}",
        )

    for t in times_all:
        for index, p in enumerate(lst_prey_path):
            cur_state = p[0][t]
            cqm.add_constraint(
                vars_s[cur_state, t] - vars_w[index, t] >= 0,
                label=f"Capture at node {cur_state} of prey {index} at time {t}",
            )

    # constraint interception multiple and single
    for t in times_minus_0:
        for index_prey in range(n_preys):
            cqm.add_constraint(
                vars_w[index_prey, t] - vars_w[index_prey, t - 1] >= 0,
                label=f"Capture Propagation at time {t} for prey {index_prey}",
            )

    for p in range(n_preys):
        cqm.add_constraint(
            vars_w[p, n_time - 1] == 1, label=f"At the end {p} is intercepted"
        )

    for t in times_all:
        vars_temp = []
        for index_prey in range(n_preys):
            vars_temp.append(vars_w[index_prey, t])
        cqm.add_constraint(
            sum(vars_temp) - n_preys * (1 - vars_u[t]) >= 0,
            label=f"All intercepted at time {t}",
        )

    return cqm, [prey[1] for prey in lst_prey_path], costs


@profile
def createMiniModelCQM():

    print("\nBuilding constrained quadratic model...")

    nodes = set(range(n_of_nodes))
    times_all = set(range(n_time))
    times_minus_0 = set(range(1, n_time))

    indices = getAllPossibleTupleMovesSetTimeMinus0(n_rows, n_cols, n_time)

    costs = {x: randint(5, 15) for x in indices}

    # lst_prey_path = [
    #     create_unvisited_path_and_states(
    #         indices, randint(1, n_of_nodes - 1), n_time, n_rows, n_cols
    #     )
    #     for _ in range(n_preys)
    # ]

    lst_prey_path = [
        createPreyPathStates(
            indices, randint(1, n_of_nodes - 1), n_time - 1, n_rows, n_cols
        )
        for _ in range(n_preys)
    ]

    cqm = ConstrainedQuadraticModel()

    vars_s = {(n, t): Binary(f"s_{n}_{t}") for n in nodes for t in times_all}

    obj_func = quicksum(
        costs[i_j_t] * vars_s[i_j_t[0], i_j_t[2] - 1] * vars_s[i_j_t[1], i_j_t[2]]
        for i_j_t in indices
    )

    cqm.set_objective(obj_func)

    for t in times_minus_0:
        for n in nodes:
            temp = []
            for move_i_j in getAllPossibleTupleMovesSet(n_rows, n_cols):
                if move_i_j[0] == n and n in possible_moves[move_i_j[1]]:
                    temp.append(vars_s[move_i_j[1], t - 1])
            cqm.add_constraint(
                vars_s[n, t] - quicksum(temp) <= 0,
                label=f"Sequential node {n} at time {t}",
            )

    cqm.add_constraint(
        vars_s[0, 0] == 1,
        label=f"Starting point",
    )

    for t in times_all:
        cqm.add_constraint(
            quicksum([vars_s[n, t] for n in nodes]) == 1,
            label=f"One occupied per time {t}",
        )

    for index, p in enumerate(lst_prey_path):
        cqm.add_constraint(
            quicksum([vars_s[p[0][t], t] for t in times_all]) >= 1,
            label=f"Capture of prey {index}",
        )

    # print(f"\nPrey 1 path:\n{lst_prey_path[0][0]}")
    # print(f"\nPrey 2 path:\n{lst_prey_path[1][0]}")

    # bqm, invert = dimod.cqm_to_bqm(cqm)

    # original_stdout = sys.stdout
    # with open("new_bqm.txt", "w") as f:
    #     sys.stdout = f
    #     print(f"Num of variables: {bqm.num_variables}\nNum of interactions: {bqm.num_interactions}")
    #     sys.stdout = original_stdout

    return cqm, [prey[1] for prey in lst_prey_path], costs


# Maximum memory usage: 304.75390625 MiB
# Function createCQM called 1 times.
# Execution time max: 240.955, average: 240.955

if __name__ == "__main__":
    cqm, path_prey, costs = createMiniModelCQM()
    original_stdout = sys.stdout
    with open("new_cqm.txt", "w") as f:
        sys.stdout = f
        print(f"{cqm}")
        sys.stdout = original_stdout  # Reset the standard output to its original value

    # mem_usage = memory_usage(createCQM)
    # print('Memory usage (in chunks of .1 seconds): %s MiB' % mem_usage)
    # print('Maximum memory usage: %s MiB' % max(mem_usage))
    # print_prof_data()
    # clear_prof_data()
