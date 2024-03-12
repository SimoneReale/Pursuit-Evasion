import numpy as np

def create_test_scenario(n_rows, n_columns, n_time, n_preys, optimal_catcher_path):
    # Initialize prey paths and edge costs
    prey_paths = {prey_id: [] for prey_id in range(n_preys)}
    edge_costs = {}

    # Define a simple cost function for edges
    base_cost = 1

    # Define the prey paths based on the optimal catcher path
    # For simplicity, this example makes preys move in a predictable pattern that aligns with the catcher's path at specific times
    for prey_id, step in enumerate(optimal_catcher_path):
        # Assuming the optimal path includes time, position tuples (t, row, col)
        t, row, col = step
        if prey_id < n_preys:
            # Set the prey path to be at this position at time t
            prey_paths[prey_id].append((t, row, col))

    # Generate costs for edges
    for row in range(n_rows):
        for col in range(n_columns):
            for t in range(n_time):
                # Assign a higher cost to edges not in the optimal path to ensure it remains optimal
                if (t, row, col) in optimal_catcher_path:
                    edge_costs[(row, col), t] = base_cost
                else:
                    edge_costs[(row, col), t] = base_cost + np.random.randint(1, 4)  # Random higher cost

    # The optimal cost is the sum of base costs along the catcher's optimal path
    optimal_cost = len(optimal_catcher_path) * base_cost

    return prey_paths, edge_costs, optimal_catcher_path, optimal_cost

# Example usage
n_rows = 2
n_columns = 2
n_time = 3
n_preys = 2

# Define an optimal path for the catcher (for simplicity, moving in a straight line)
optimal_catcher_path = [(t, 0, t % n_columns) for t in range(n_time)]

prey_paths, edge_costs, catcher_optimal_path, optimal_cost = create_test_scenario(n_rows, n_columns, n_time, n_preys, optimal_catcher_path)

print("Prey paths:", prey_paths)
print("Edge costs:", edge_costs)
print("Catcher's optimal path:", catcher_optimal_path)
print("Optimal cost:", optimal_cost)
