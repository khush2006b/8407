# Hill Climbing for Block World

goal = {'A': 'B', 'B': 'C', 'C': 'table'}

def heuristic(state):
    score = 0
    for k in state:
        if state[k] == goal[k]:
            score += 1
        else:
            score -= 1
    return score

def get_neighbors(state):
    neighbors = []
    for block in state:
        for target in list(state.keys()) + ['table']:
            if block != target:
                new = state.copy()
                new[block] = target
                neighbors.append(new)
    return neighbors

def hill_climbing(start):
    current = start
    current_score = heuristic(current)

    while True:
        neighbors = get_neighbors(current)

        best = current
        best_score = current_score

        for n in neighbors:
            s = heuristic(n)
            if s > best_score:
                best = n
                best_score = s

        if best_score <= current_score:
            return current

        current = best
        current_score = best_score


start = {'A': 'table', 'B': 'table', 'C': 'table'}

result = hill_climbing(start)

print("Final State:", result)