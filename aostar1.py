graph = {
    'S': ('OR', ['A', 'B']),
    'A': ('AND', ['C', 'D']),
    'B': ('OR', ['E', 'F'])
}

cost = {'C':4,'D':6,'E':10,'F':3}

solution = {}

def ao_star(node):
    if node in cost:
        return cost[node]

    typ, children = graph[node]

    if typ == 'OR':
        min_cost = float('inf')
        best = None

        for c in children:
            val = ao_star(c)
            if val < min_cost:
                min_cost = val
                best = c

        solution[node] = best
        return min_cost

    else:
        total = 0
        solution[node] = children

        for c in children:
            total += ao_star(c)

        return total


print("Cost:", ao_star('S'))
print("Solution:", solution)