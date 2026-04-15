# Greedy Best First Search

import heapq

graph = {
    'A': [('B', 2), ('C', 4)],
    'B': [('D', 7), ('E', 3)],
    'C': [('F', 5)],
    'D': [('G', 1)],
    'E': [('G', 6)],
    'F': [('G', 2)],
    'G': []
}

heuristic = {
    'A': 7, 'B': 6, 'C': 5,
    'D': 1, 'E': 4, 'F': 2, 'G': 0
}

def greedy(start, goal):
    pq = [(heuristic[start], start, [start], 0)]

    visited = set()

    while pq:
        h_val, node, path, cost = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, cost

        for neighbor, w in graph[node]:
            heapq.heappush(pq, (heuristic[neighbor], neighbor, path+[neighbor], cost+w))

    return None, None


path, cost = greedy('A', 'G')
print("Greedy Path:", path)
print("Cost:", cost)