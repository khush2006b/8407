import heapq

goal = [[1,2,3],[4,5,6],[7,8,0]]

def misplaced(state):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal[i][j]:
                count += 1
    return count

def get_neighbors(state):
    neighbors = []
    x,y = [(i,j) for i in range(3) for j in range(3) if state[i][j]==0][0]

    moves = [(0,1),(0,-1),(1,0),(-1,0)]

    for dx,dy in moves:
        nx,ny = x+dx, y+dy
        if 0<=nx<3 and 0<=ny<3:
            new = [row[:] for row in state]
            new[x][y], new[nx][ny] = new[nx][ny], new[x][y]
            neighbors.append(new)

    return neighbors

def astar_8puzzle(start):
    pq = []
    heapq.heappush(pq, (misplaced(start), 0, start, []))

    visited = set()

    while pq:
        f,g,state,path = heapq.heappop(pq)

        if str(state) in visited:
            continue
        visited.add(str(state))

        if state == goal:
            return path+[state]

        for neighbor in get_neighbors(state):
            heapq.heappush(pq, (g+1+misplaced(neighbor), g+1, neighbor, path+[state]))

    return None


start = [[1,2,3],[4,0,6],[7,5,8]]
solution = astar_8puzzle(start)

for step in solution:
    print(step)