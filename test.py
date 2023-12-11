# battery modes

import pulp
import numpy as np
from random import randint, choice

# # some data
# num_periods = 3
# rate_limits = { 'energy'    : 10,
#                 'freq'      : 20}
# price = 2  # this could be a table or double-indexed table of [t, m] or ....

# # SETS
# M = rate_limits.keys()   # modes.  probably others...  discharge?
# T = range(num_periods)   # the time periods

# TM = {(t, m) for t in T for m in M}

# model = pulp.LpProblem('Batts', pulp.LpMaximize)

# # VARS
# model.batt = pulp.LpVariable.dicts('batt_state', indexs=TM, lowBound=0, cat='Binary')
# model.op_mode = pulp.LpVariable.dicts('op_mode', indexs=TM, cat='Binary')

# # Constraints

# # only one op mode in each time period...
# for t in T:
#     model += sum(model.op_mode[t, m] for m in M) <= 1

# # Big-M constraint. limit rates for each rate, in each period.
# # this does 2 things:  it is equivalent to the upper bound parameter in the var declaration
# #                      It is a Big-M type of constraint which uses the binary var as a control <-- key point
# for t, m in TM:
#     model += model.batt[t, m] <= rate_limits[m] * model.op_mode[t, m]

# # OBJ
# model += sum(model.batt[t, m] * price for t, m in TM)

# print(model)


def getAllPossibleTupleMovesSet(n_rows: int, n_columns: int):
    n_of_nodes = n_rows * n_columns
    move_dict = []

    for x in range(n_of_nodes):
        # stay
        # move_dict.append((x, x))
        # up
        if x - n_columns >= 0:
            move_dict.append((x, x - n_columns))
        # down
        if x + n_columns < n_of_nodes:
            move_dict.append((x, x + n_columns))
        # right
        if (x + 1) % n_columns != 0:
            move_dict.append((x, x + 1))
        # left
        if x % n_columns != 0:
            move_dict.append((x, x - 1))
    return move_dict


def getAllPossibleTupleMovesSetTime(n_rows: int, n_columns: int, n_time: int):
    n_of_nodes = n_rows * n_columns
    move_dict = []

    for t in range(n_time):
        for x in range(n_of_nodes):
            # stay
            # move_dict.append((x, x))
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


def create_random_prey_path(indices, starting_node: int, length: int, n_rows: int, n_cols: int):

    prey_path = {x: 0 for x in indices}
    lst_positions = createPolicyUnvisitedPath(starting_node, length, n_rows, n_cols)
    for x in lst_positions:
        prey_path[x] = 1
    return prey_path


n_of_nodes = 9
n_rows = int(np.sqrt(n_of_nodes))
n_cols = int(np.sqrt(n_of_nodes))
n_time = 4
n_preys = 2

nodes = set(range(n_of_nodes))
times = set(range(n_time))

indices = getAllPossibleTupleMovesSetTime(n_rows, n_cols, n_time)

costs = {x: randint(1, 50) for x in indices}

path_prey_1 = create_random_prey_path(indices, 8, n_time, n_rows, n_cols)


model = pulp.LpProblem("Pursuit", pulp.LpMinimize)

# # VARS
model.var_path = pulp.LpVariable.dicts(
    "var_path", indices=indices, lowBound=0, cat="Binary"
)

# funzione obiettivo
model += sum(model.var_path[x[0], x[1], x[2]] * costs[x] for x in indices)

# faccio un solo passaggio per istante di tempo
for t in range(n_time):
    model += (
        sum(
            model.var_path[x[0], x[1], t]
            for x in getAllPossibleTupleMovesSet(n_rows, n_cols)
        )
        == 1
    )


model += sum(model.var_path[i1, j1, t] * path_prey_1[i2, j2, t] if j1 == j2 and i1 != i2 and j1 != j2 else 0 for i1 in nodes for i2 in nodes for j1 in nodes for j2 in nodes for t in times) >= 1

print(model)
