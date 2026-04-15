import heapq

def astar_graph(start, goal, graph, heuristic):
    pq = [(heuristic[start], 0, start, [start])]

    visited = {}

    while pq:
        f,g,node,path = heapq.heappop(pq)

        if node in visited and visited[node] <= g:
            continue
        visited[node] = g

        if node == goal:
            return path, g

        for neighbor, cost in graph[node]:
            new_g = g + cost
            heapq.heappush(pq, (new_g + heuristic[neighbor], new_g, neighbor, path+[neighbor]))

    return None, None


path, cost = astar_graph('A','G',graph,heuristic)
print("A* Path:", path)
print("Cost:", cost)