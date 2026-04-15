# Minimax for Tic-Tac-Toe

def print_board(board):
    for row in board:
        print(row)
    print()

def is_moves_left(board):
    return any(0 in row for row in board)

def evaluate(board):
    # Rows, columns, diagonals
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i]:
            return board[0][i]

    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]

    return 0

def minimax(board, depth, is_max):
    score = evaluate(board)

    if score == 1 or score == -1:
        return score
    if not is_moves_left(board):
        return 0

    if is_max:
        best = -1000
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = 1
                    best = max(best, minimax(board, depth+1, False))
                    board[i][j] = 0
        return best
    else:
        best = 1000
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = -1
                    best = min(best, minimax(board, depth+1, True))
                    board[i][j] = 0
        return best

def best_move(board):
    best_val = -1000
    move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = 1
                move_val = minimax(board, 0, False)
                board[i][j] = 0

                if move_val > best_val:
                    move = (i, j)
                    best_val = move_val

    return move


board = [
    [ 1, -1,  0],
    [ 0,  1, -1],
    [ 0,  0,  0]
]

print("Initial Board:")
print_board(board)

move = best_move(board)
print("Best Move:", move)