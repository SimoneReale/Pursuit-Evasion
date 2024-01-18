import pickle
from utils import *
import networkx as nx
from random import choice
import numpy as np


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


def fromPathToSequenceOfNodes(path):
    seq = []
    for i in range(len(path) - 1):
        seq.append(path[i][0])
    seq.append(path[-1][0])
    seq.append(path[-1][1])
    return seq


def calculateScore(catcher_path: list[int], prey_path: list[int]):
    time_encounter = [
        x for x, y in enumerate(zip(catcher_path, prey_path)) if y[0] == y[1]
    ]
    if not time_encounter:
        return -1
    else:
        return min(time_encounter)


def anim():
    with open("items_solution_quantum.pkl", "rb") as f:
        loaded = pickle.load(f)

        def extract_numbers(s):
            return tuple(map(int, s.split("_")[1:]))

        soln_list = [extract_numbers(s) for s in loaded["sol"]]
        soln_list.sort(key=lambda tup: tup[2])
        print(soln_list)

    nodes = set(range(n_of_nodes))
    times = set(range(n_time))
    path_prey = loaded["prey"]
    costs = loaded["costs"]
    prey1 = [i for i, j in path_prey[0].items() if j == 1]

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


anim()
