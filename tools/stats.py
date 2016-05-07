import algos

class Stats:

	def __init__(self, the_situation, the_continent):
		self.the_situation = the_situation
		self.the_continent = the_continent

	def update_piece(self, piece, pieces_alongside, pieces_on_group):
		location = piece.get_location()
		group = self.the_continent.get_group(location)
		piece_id = piece.piece_id
		piece_type_id = piece.piece_type_id
		# Add to pieces_on_group.
		identifier = group, piece_type_id
		if identifier not in pieces_on_group:
			pieces_on_group[identifier] = []
		pieces_on_group[identifier].append(piece_id)
		# If the piece is alongside another group, add to pieces_alongside.
		for direction in algos.directions:
			qd, rd = direction
			ql, rl = location
			neighbor_location = qd + ql, rd + rl
			if not self.the_situation.is_in_map(neighbor_location):
				continue
			neighbor_group = self.the_continent.get_group(neighbor_location)
			if neighbor_group is not None and neighbor_group != group:
				identifier = neighbor_group, piece_type_id
				if identifier not in pieces_alongside:
					pieces_alongside[identifier] = []
				pieces_alongside[identifier].append(piece_id)

	def update_city(self, city, cities_alongside, cities_on_group):
		location = city.get_location()
		group = self.the_continent.get_group(location)
		city_id = city.city_id
		# Add to cities_on_group.
		identifier = group
		if identifier not in cities_on_group:
			cities_on_group[identifier] = []
		cities_on_group[identifier].append(city_id)
		# If the city is alongside another group, add to city_alongside.
		for direction in algos.directions:
			qd, rd = direction
			ql, rl = location
			neighbor_location = qd + ql, rd + rl
			if not self.the_situation.is_in_map(neighbor_location):
				continue
			neighbor_group = self.the_continent.get_group(neighbor_location)
			if neighbor_group is not None and neighbor_group != group:
				identifier = neighbor_group
				if identifier not in cities_alongside:
					cities_alongside[identifier] = []
				cities_alongside[identifier].append(city_id)

	def update(self):
		self.player_pieces_alongside = {}
		self.player_pieces_on_group = {}
		self.player_cities_alongside = {}
		self.player_cities_on_group = {}

		self.enemy_pieces_alongside = {}
		self.enemy_pieces_on_group = {}
		self.enemy_cities_alongside = {}
		self.enemy_cities_on_group = {}

		self.free_cities_alongside = {}
		self.free_cities_on_group = {}

		for piece in self.the_situation.get_player_pieces():
			self.update_piece(piece, self.player_pieces_alongside, self.player_pieces_on_group)
		for piece in self.the_situation.get_enemy_pieces():
			self.update_piece(piece, self.enemy_pieces_alongside, self.enemy_pieces_on_group)
		for city in self.the_situation.get_player_cities():
			self.update_city(city, self.player_cities_alongside, self.player_cities_on_group)
		for city in self.the_situation.get_enemy_cities():
			self.update_city(city, self.enemy_cities_alongside, self.enemy_cities_on_group)
		for city in self.the_situation.get_free_cities():
			self.update_city(city, self.free_cities_alongside, self.free_cities_on_group)

	def show(self):
		print "STATS:"
		def helper(m, s):
			for k in m:
				print " %s[%s] : %d" % (s, str(k), len(m[k]))
		helper(self.player_pieces_alongside, "self.player_pieces_alongside")
		helper(self.player_pieces_on_group, "self.player_pieces_on_group")
		helper(self.player_cities_alongside, "self.player_cities_alongside")
		helper(self.player_cities_on_group, "self.player_cities_on_group")
		helper(self.enemy_pieces_alongside, "self.enemy_pieces_alongside")
		helper(self.enemy_pieces_on_group, "self.enemy_pieces_on_group")
		helper(self.enemy_cities_alongside, "self.enemy_cities_alongside")
		helper(self.enemy_cities_on_group, "self.enemy_cities_on_group")
		helper(self.free_cities_alongside, "self.free_cities_alongside")
		helper(self.free_cities_on_group, "self.free_cities_on_group")

	def get_nb_player_cities_on_group(self, group):
		if group in self.player_cities_on_group:
			return len(self.player_cities_on_group[group])
		return 0

	def get_nb_enemy_cities_on_group(self, group):
		if group in self.enemy_cities_on_group:
			return len(self.enemy_cities_on_group[group])
		return 0

	def get_nb_free_cities_on_group(self, group):
		if group in self.free_cities_on_group:
			return len(self.free_cities_on_group[group])
		return 0

	def get_nb_enemy_pieces_on_group(self, group):
		count = 0
		for identifier in self.enemy_pieces_on_group:
			if identifier[0] == group:
				count = count + len(self.enemy_pieces_on_group[identifier])
		return count

	def get_nb_player_pieces_on_group(self, group):
		count = 0
		for identifier in self.player_pieces_on_group:
			if identifier[0] == group:
				count = count + len(self.player_pieces_on_group[identifier])
		return count

	def get_player_pieces_on_group(self, group, piece_types):
		result = []
		for piece_type in piece_types:
			if (group, piece_type) in self.player_pieces_on_group:
				result.extend(self.player_pieces_on_group[(group, piece_type)])
		return result

	def get_player_pieces_alongside(self, group, piece_types):
		result = []
		for piece_type in piece_types:
			if (group, piece_type) in self.player_pieces_alongside:
				result.extend(self.player_pieces_alongside[(group, piece_type)])
		return result
