# AND-OR graph
graph = {
    'S': ('OR', ['A', 'B']),
    'A': ('AND', ['C', 'D']),
    'B': ('OR', ['E', 'F'])
}

# Terminal costs
cost = {
    'C': 4,
    'D': 6,
    'E': 10,
    'F': 3
}

# Heuristics
h = {
    'S': 8,
    'A': 5,
    'B': 6
}

# Store solution
solution = {}

def ao_star(node):
    if node in cost:
        return cost[node]

    node_type, children = graph[node]

    if node_type == 'OR':
        min_cost = float('inf')
        best_child = None

        for child in children:
            c = ao_star(child)
            if c < min_cost:
                min_cost = c
                best_child = child

        solution[node] = best_child
        return min_cost

    elif node_type == 'AND':
        total_cost = 0
        solution[node] = children

        for child in children:
            total_cost += ao_star(child)

        return total_cost


# Run AO*
final_cost = ao_star('S')

print("\nOptimal Solution Graph:")
for k, v in solution.items():
    print(f"{k} -> {v}")

print("\nMinimum Total Cost:", final_cost)