import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from random import choice, randint
from dataclasses import dataclass


n_of_nodes = 16
n_rows = int(np.sqrt(n_of_nodes))
n_cols = int(np.sqrt(n_of_nodes))
n_time = 12
n_preys = 2


################################################################## HO DISATTIVATO LO STAY


@dataclass
class BestPoint:
    prey_num: int
    target: int
    dist_from_target: int
    time_difference: int


# The distance d(u,v) between two vertices u and v of a finite graph
# is the minimum length of the paths connecting them (i.e., the length of a graph geodesic).
# If no such path exists (i.e., if the vertices lie in different connected components),
# then the distance is set equal to infinity.
# In a grid graph the distance between two vertices
# is the sum of the "vertical" and the "horizontal" distances.
# The matrix (d_(ij)) consisting of all distances
# from vertex v_i to vertex v_j is known as the all-pairs shortest path matrix,
# or more simply, the graph distance matrix.
def calculateDistance(vertex1: int, vertex2: int) -> int:
    coord1 = fromNumToCoord(vertex1)
    coord2 = fromNumToCoord(vertex2)
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def fromCoordToNum(coord) -> int:
    num = n_cols * coord[0] + coord[1]
    return num


def fromNumToCoord(num) -> (int, int):
    col = num % n_cols
    row = int(np.floor((num - col) / n_rows))
    return (row, col)


def addPath(mat_nodes, evo_pos, color):
    frame = 0
    for pos in evo_pos:
        mat_nodes[frame][pos] = color
        frame += 1


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


def getAllPossibleMoves(n_rows: int, n_columns: int):
    n_of_nodes = n_rows * n_columns
    move_dict = {}

    for x in range(n_of_nodes):
        pos = []
        # stay
        #pos.append(x)
        # up
        if x - n_columns >= 0:
            pos.append(x - n_columns)
        # down
        if x + n_columns < n_of_nodes:
            pos.append(x + n_columns)
        # right
        if (x + 1) % n_columns != 0:
            pos.append(x + 1)
        # left
        if x % n_columns != 0:
            pos.append(x - 1)

        move_dict[x] = pos

    return move_dict


def getAllPossibleTupleMovesSet(n_rows: int, n_columns: int):
    n_of_nodes = n_rows * n_columns
    move_dict = []

    for x in range(n_of_nodes):
        # stay
        #move_dict.append((x, x))
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
    move_dict_new = []
    [move_dict_new.append(x) for x in move_dict if (x[1], x[0]) not in move_dict_new]
    return {x : randint(1, 50) for x in move_dict_new}


def createPathToNode(start: int, dest: int) -> list[int]:
    coord_start = fromNumToCoord(start)
    coord_dest = fromNumToCoord(dest)

    diff_row = coord_start[0] - coord_dest[0]
    diff_col = coord_start[1] - coord_dest[1]

    path = []

    for i in range(1, abs(diff_row) + 1):
        if diff_row < 0:
            path.append(fromCoordToNum((coord_start[0] + i, coord_start[1])))
        else:
            path.append(fromCoordToNum((coord_start[0] - i, coord_start[1])))

    for i in range(1, abs(diff_col) + 1):
        if diff_col < 0:
            path.append(fromCoordToNum((coord_dest[0], coord_start[1] + i)))
        else:
            path.append(fromCoordToNum((coord_dest[0], coord_start[1] - i)))

    return path


def createRandomPath(start: int, length: int, n_rows: int, n_columns: int) -> list[int]:
    path = [start]
    for _ in range(length - 1):
        moves = getSetOfMoves(path[-1], n_rows, n_columns)
        path.append(choice(moves))
    return path


def calculateScore(catcher_path: list[int], prey_path: list[int]):
    time_encounter = [
        x for x, y in enumerate(zip(catcher_path, prey_path)) if y[0] == y[1]
    ]
    if not time_encounter:
        return -1
    else:
        return min(time_encounter)
