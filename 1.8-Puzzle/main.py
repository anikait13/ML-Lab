from heapq import heappush, heappop

def manhattan_distance(current, goal):
    """Calculate the Manhattan distance between two states"""
    distance = 0
    for i in range(9):
        current_row, current_col = divmod(current.index(i), 3)
        goal_row, goal_col = divmod(goal.index(i), 3)
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance

def a_star(start, goal):
    """Return a list of steps to reach the goal state from the start state using the A* algorithm"""
    heap = [(0, start)]
    visited = set()
    while heap:
        (steps, current) = heappop(heap)
        if current == goal:
            return steps
        current = tuple(current)
        if current in visited:
            continue
        visited.add(current)
        print(current)
        zero_index = current.index(0)
        row, col = divmod(zero_index, 3)
        for i, j in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if i < 0 or i > 2 or j < 0 or j > 2:
                continue
            new_state = list(current)
            new_index = i * 3 + j
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            heappush(heap, (steps + manhattan_distance(new_state, goal) + 1, new_state))
    return None


start = [1, 3, 2, 4, 5, 6, 7, 0, 8]
goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
steps = a_star(start, goal)
if steps is not None:
    print(steps)
    print(f"Found solution in {steps} steps")
else:
    print("No solution found")
