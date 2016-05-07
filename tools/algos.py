import heapq

def astar_goal(start, goal, max_cost, neighbors, cost, heuristic):
	frontier = []
	heapq.heappush(frontier, (start, 0))
	came_from = {start: None}
	cost_so_far = {start: 0}
	
	while len(frontier) > 0:
		current, tmp = heapq.heappop(frontier)
		if current == goal:
			break
		for next in neighbors(current):
			new_cost = cost_so_far[current] + cost(current, next)
			if new_cost <= max_cost and (next not in cost_so_far or new_cost < cost_so_far[next]):
				cost_so_far[next] = new_cost
				priority = new_cost + heuristic(goal, next)
				heapq.heappush(frontier, (next, priority))
				came_from[next] = current
	if goal in came_from:
		l = [goal]
		x = goal
		while x != start:
			x = came_from[x]
			l = [x] + l
		return l
	else:
		return None

# fgoal means function goal.
def breadth_first_fgoal(start, fgoal, neighbors):
	frontier = [start]
	visited = {start: True}
	came_from = {start: None}

	while len(frontier) > 0:
		current = frontier[0]
		del frontier[0]
		if fgoal(current):
			path = [current]
			while current != start:
				current = came_from[current]
				path.append(current)
			return path[::-1]
		for next in neighbors(current):
			if next not in visited:
				frontier.append(next)
				visited[next] = True
				came_from[next] = current
	return None

def breadth_first_search_all(start, depth, neighbors, crossable):
	frontier = [ (0, start) ]
	visited = {start: True}
	came_from = {start: None}
	path_len = {start: 0}
	reachable = []

	while len(frontier) > 0:
		current_depth, current = frontier[0]
		del frontier[0]
		if current_depth < depth:
			for next in neighbors(current):
				if next not in visited:
					reachable.append(next)
					if crossable(next):
						frontier.append((current_depth + 1, next))
					visited[next] = True
					came_from[next] = current
					path_len[next] = current_depth + 1
	return reachable, came_from, path_len

directions = [ (+1,  0), (+1, -1), ( 0, -1), (-1,  0), (-1, +1), ( 0, +1) ]

def cube_ring(center, radius):
	if radius == 0:
		return [center]
	q, r = center
	results = []
	for i in range(len(directions)):
		qa, ra = directions[i]
		qb, rb = directions[(i + 1) % len(directions)]
		qd, rd = qb - qa, rb - ra
		if qd != 0:
			qd = qd / abs(qd)
		if rd != 0:
			rd = rd / abs(rd)
		for j in range(radius):
			results.append((q + radius * qa + j * qd, r + radius * ra + j * rd))
	return results

def delta_to_direction(location_a, location_b):
	delta = location_b[0] - location_a[0], location_b[1] - location_a[1]
	return directions.index(delta) 

def get_tiles_distance(location_a, location_b):
	qa, ra = location_a
	qb, rb = location_b
	return (abs (qa - qb) + abs (qa + ra - qb - rb) + abs (ra - rb)) / 2
