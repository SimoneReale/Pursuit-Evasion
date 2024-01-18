import pulp
import numpy as np
from random import randint, choice
import networkx as nx
from utils import *
from pygame import mixer
from time_profiling import profile, print_prof_data, clear_prof_data
from memory_profiler import memory_usage


mixer.init()
mixer.music.load("/home/ant0nius/Downloads/laser.wav")


def getAllPossibleTupleMovesSet(n_rows: int, n_columns: int):
    n_of_nodes = n_rows * n_columns
    move_dict = []

    for x in range(n_of_nodes):
        # stay
        move_dict.append((x, x))
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


def fromPathToSequenceOfNodes(path):
    seq = []
    for i in range(len(path) - 1):
        seq.append(path[i][0])
    seq.append(path[-1][0])
    seq.append(path[-1][1])
    return seq


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


def create_random_policy_prey_path(
    indices, starting_node: int, length: int, n_rows: int, n_cols: int
):
    prey_path = {x: 0 for x in indices}
    lst_positions = createRandomPath(starting_node, length, n_rows, n_cols)
    for x in lst_positions:
        prey_path[x] = 1
    return prey_path


nodes = set(range(n_of_nodes))
times = set(range(n_time))
indices = getAllPossibleTupleMovesSetTime(n_rows, n_cols, n_time)
# costs = {}
# for x in indices:
#     costs[x] = randint(1, 50) if x[0] != x[1] else 0
costs = {x: randint(1, 50) for x in indices}
path_preys = [
    create_unvisited_policy_prey_path(
        indices, randint(1, n_of_nodes - 1), n_time, n_rows, n_cols
    )
    for _ in range(n_preys)
]

# Maximum memory usage: 160.45703125 MiB
# Function create_model called 1 times.
# Execution time max: 332.870, average: 332.870

@profile
def create_model():
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

    var = []
    for p in path_preys:
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
                                # print(f"{(i1, j1)} e {(i2, j2)} al tempo {t}")
                                var.append(model.var_path[i1, j1, t] * p[i2, j2, t])

        model += sum(var) >= 1

    model += model.var_path[0, 1, 0] == 1

    for t in times.difference({0}):
        for i in nodes:
            for j in nodes:
                if j in getSetOfMoves(i, n_rows, n_cols):
                    model += (
                        model.var_path[i, j, t]
                        - sum(
                            model.var_path[k, i, t - 1]
                            for k in nodes
                            if i in getSetOfMoves(k, n_rows, n_cols)
                        )
                    ) <= 0

    return model


if __name__ == "__main__":
    # glpk = pulp.apis.GLPK_CMD()

    # status = glpk.actualSolve(model)
    model = create_model()

    status = model.solve()

    # solver = pulp.getSolver('CPLEX_PY')
    # status = solver.solve(model)

    if status == -1:
        print("Your problem is infeasible!")

    inv_var = {v: k for k, v in model.var_path.items()}


    soln_list = [inv_var[i] for i in model.variables() if i.varValue == 1]
    
    soln_list.sort(key=lambda tup: tup[2])
    print(soln_list)
    prey1 = [i for i, j in path_preys[0].items() if j == 1]

    # prey2 = [i for i, j in path_prey[1].items() if j == 1] ############################# PREY 2

    G = nx.grid_2d_graph(n_rows, n_cols)
    for x in G.nodes:
        G.add_edge(x, x)
    node_color_map = np.full((n_time + 1, n_rows * n_cols), "black")
    # aggiungo autoanelli
    edge_color_map = np.full(
        (n_time + 1, 2 * n_rows * n_cols - n_rows - n_cols + n_rows * n_cols), "black"
    )

    blue_dot = fromPathToSequenceOfNodes(prey1)
    # green_dot = fromPathToSequenceOfNodes(prey2)   ############################# PREY 2
    red_dot = fromPathToSequenceOfNodes(soln_list)

    score_prey_1 = calculateScore(red_dot, blue_dot)
    # score_prey_2 = calculateScore(red_dot, green_dot)  ############################# PREY 2

    cost_map = {x: {} for x in times}
    for key, value in costs.items():
        start = fromNumToCoord(key[0])
        end = fromNumToCoord(key[1])
        cost_map[key[2]][(start, end)] = value

    addPath(node_color_map, blue_dot, "blue")
    # addPath(node_color_map, green_dot, "green") ############################# PREY 2
    addPath(node_color_map, red_dot, "red")

    print(f"Preda 1 punto blu: {blue_dot}")
    # print(f"Preda 2 punto verde: {green_dot}")  ############################# PREY 2
    print(f"Catcher punto rosso: {red_dot}")
    print(f"\nscore1 {score_prey_1}")
    # print(f"\nscore2 {score_prey_2}\n")

    pos = {(x, y): (y, -x) for x, y in G.nodes()}
    nodes = nx.draw_networkx_nodes(
        G, node_color=node_color_map[0], pos=pos, node_size=250
    )
    edges = nx.draw_networkx_edges(G, edge_color=edge_color_map[0], pos=pos)
    plt.axis("off")

    def update(i):
        nodes = nx.draw_networkx_nodes(
            G, node_color=node_color_map[i], pos=pos, node_size=250
        )
        edge_labels = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=cost_map[i])
        plt.pause(1)
        # if i == score_prey_1 or i == score_prey_2:
        #     mixer.music.play()
        return (nodes, edge_labels)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    fig = plt.gcf()
    ani = FuncAnimation(fig, update, interval=1200, frames=n_time, blit=True)
    plt.show()

    print("\n")
    print_prof_data()

    # for name in model.constraints.keys():
    #     value = model.constraints.get(name).value()
    #     slack = model.constraints.get(name).slack
    #     print(f'constraint {name} has value: {value:0.2e} and slack: {slack:0.2e}')

    mem_usage = memory_usage(create_model)
    print('Memory usage (in chunks of .1 seconds): %s MiB' % mem_usage)
    print('Maximum memory usage: %s MiB' % max(mem_usage))
    print_prof_data()
    clear_prof_data()
