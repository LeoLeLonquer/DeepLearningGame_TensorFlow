UNCLASSIFIED = False
NOISE = None

class Dbscan:

	def __init__(self, dist, eps, min_points):
		self.dist = dist
		self.eps = eps
		self.min_points = min_points

	def eps_neighborhood(self, p, q):
		return self.dist(p, q) < self.eps

	def region_query(self, point_id):
		seeds = []
		for i in range(0, self.n_points):
			if self.eps_neighborhood(self.data[point_id], self.data[i]):
				seeds.append(i)
		return seeds

	def expand_cluster(self, point_id):
		seeds = self.region_query(point_id)
		if len(seeds) < self.min_points:
			self.classifications[point_id] = NOISE
			return False
		else:
			self.classifications[point_id] = self.cluster_id
			for seed_id in seeds:
				self.classifications[seed_id] = self.cluster_id
			while len(seeds) > 0:
				current_point = seeds[0]
				results = self.region_query(current_point)
				if len(results) >= self.min_points:
					for i in range(0, len(results)):
						result_point = results[i]
						if self.classifications[result_point] == UNCLASSIFIED or \
						   self.classifications[result_point] == NOISE:
							if self.classifications[result_point] == UNCLASSIFIED:
								seeds.append(result_point)
							self.classifications[result_point] = self.cluster_id
				seeds = seeds[1:]
			return True

	def dbscan(self, data):
		self.cluster_id = 1
		self.n_points = len(data)
		self.data = data
		self.classifications = [UNCLASSIFIED] * self.n_points
		for point_id in range(0, self.n_points):
			if self.classifications[point_id] == UNCLASSIFIED:
				if self.expand_cluster(point_id):
					self.cluster_id = self.cluster_id + 1
		#return self.classifications
		result = {}
		alone = []
		for i in range(self.n_points):
			c = self.classifications[i]
			d = self.data[i]
			if c is None:
				alone.append([d])
			else:
				if c not in result:
					result[c] = []
				result[c].append(d)
		return result.values() + alone
