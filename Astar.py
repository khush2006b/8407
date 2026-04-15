import heapq

# Graph definition
graph = {
    'A': [('B', 2), ('C', 4)],
    'B': [('D', 7), ('E', 3)],
    'C': [('F', 5)],
    'D': [('G', 1)],
    'E': [('G', 6)],
    'F': [('G', 2)],
    'G': []
}

# Heuristic values
h = {
    'A': 7, 'B': 6, 'C': 5,
    'D': 1, 'E': 4, 'F': 2, 'G': 0
}

def astar(start, goal):
    pq = []
    heapq.heappush(pq, (h[start], 0, start, [start]))

    visited = {}

    print("\nNode Expansions:")

    while pq:
        f, g, node, path = heapq.heappop(pq)

        if node in visited and visited[node] <= g:
            continue

        visited[node] = g

        print(f"Expand {node}: g={g}, h={h[node]}, f={f}")

        if node == goal:
            return path, g

        for neighbor, cost in graph[node]:
            new_g = g + cost
            new_f = new_g + h[neighbor]
            heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')


# Run A*
path, cost = astar('A', 'G')

print("\nFinal Path:", path)
print("Total Cost:", cost)