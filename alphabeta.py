# Minimax with Alpha-Beta Pruning (Game Tree)

def minimax_ab(depth, node_index, is_max, values, alpha, beta):
    if depth == 3:
        return values[node_index]

    if is_max:
        best = -1000

        for i in range(2):
            val = minimax_ab(depth+1, node_index*2+i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)

            if beta <= alpha:
                print("Pruned at MAX node")
                break

        return best

    else:
        best = 1000

        for i in range(2):
            val = minimax_ab(depth+1, node_index*2+i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)

            if beta <= alpha:
                print("Pruned at MIN node")
                break

        return best


values = [3, 5, 6, 9, 1, 2, 0, -1]

result = minimax_ab(0, 0, True, values, -1000, 1000)
print("Optimal Value:", result)