import algos

class Influence:

	def __init__(self, situation):
		self.situation = situation

	def update(self):
		situation = self.situation
		self.height = situation.height
		self.width = situation.width
		self.influence = [[0 for x in range(self.height)] for x in range(self.width)]
		for q in range(self.width):
			for r in range(self.height):
				loc = q, r
				if situation.is_tile_none(loc):
					pass
				elif situation.is_tile_player_piece(loc):
					piece = situation.get_content(loc)
					piece_type = situation.get_piece_type(piece.piece_type_id)
					force = piece_type.speed * 2
					self.lerp(loc, force, 1)
				elif situation.is_tile_player_city(loc):
					self.lerp(loc, 5, 1)
				elif situation.is_tile_enemy_piece(loc):
					situation.check()
					content = situation.get_content(loc)
					piece_type = situation.get_piece_type(content.piece_type_id)
					force = piece_type.speed * 2
					self.lerp(loc, force, -1)
				elif situation.is_tile_enemy_city(loc):
					self.lerp(loc, 5, -1)
				elif situation.is_tile_free_city(loc):
					pass

	def lerp(self, loc, force, factor):
		q, r = loc
		for i in range(force):
			for (qa, ra) in algos.cube_ring(loc, i):
				if self.situation.is_in_map((qa, ra)):
					self.influence[qa][ra] += (force - i) * factor

	def choose_goals(self):
		pass

	def show(self):
		print "===="
		for j in range(self.height):
			line = " " * j + "\\"
			for i in range(self.width):
				if self.influence[i][j] != 0:
					line = line + ("%4d" % self.influence[i][j])
				else:
					line = line + "    "
			print line
