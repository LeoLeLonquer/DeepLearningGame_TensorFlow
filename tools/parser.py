import re
import sys

import situation

class Parser:
	def __init__(self, situation):
		self.situation = situation
		self.re_transport_in_city = re.compile(r"transport in-city (\d+) (\d+)")

		# The order matters: it must be the same of both rexes and handlers (see parse method).
		self.rexes =	[ re.compile(r"set_visible (\d+) (\d+) (\w+) none")
				, re.compile(r"set_visible (\d+) (\d+) (\w+) owned_city (\d+) (\d+)")
				, re.compile(r"set_visible (\d+) (\d+) (\w+) city (\d+)")
				, re.compile(r"set_visible (\d+) (\d+) (\w+) piece (\d+) (\w) (\d+) (\d+)")
				, re.compile(r"set_explored (\d+) (\d+) (\w+)")
				, re.compile(r"width (\d+)")
				, re.compile(r"height (\d+)")
				, re.compile(r"pieces_types (.*)")
				, re.compile(r"player_id (\d+)")
				, re.compile(r"create_piece (\d+) (\d+) (\w) (\d+)")
				, re.compile(r"delete_piece (\d+)")
				, re.compile(r"enter_city (\d+) (\d+)")
				, re.compile(r"leave_city (\d+) (\d+)")
				, re.compile(r"leave_terrain (\d+) (\d+) (\d+)")
				, re.compile(r"move (\d+) (\d+) (\d+)")
				, re.compile(r"invade_city (\d+) (\d+) (\d+)")
				, re.compile(r"lose_city (\d+)")
				, re.compile(r"winner (\d+)")
				]
		self.handlers =	[ self.parse_set_visible_none
				, self.parse_set_visible_owned_city
				, self.parse_set_visible_city
				, self.parse_set_visible_piece
				, self.parse_set_explored
				, self.parse_width
				, self.parse_height
				, self.parse_pieces_types
				, self.parse_player_id
				, self.parse_create_piece
				, self.parse_delete_piece
				, self.parse_enter_city
				, self.parse_leave_city
				, self.parse_leave_terrain
				, self.parse_move
				, self.parse_invade_city
				, self.parse_lose_city
				, self.parse_winner
				]

	def parse_set_visible_none(self, groups):
		location = int(groups.group(1)), int(groups.group(2))
		terrain = situation.GROUND if groups.group(3) == "ground" else situation.WATER
		self.situation.set_visible_none(location, terrain)

	def parse_set_visible_owned_city(self, groups):
		location = int(groups.group(1)), int(groups.group(2))
		terrain = situation.GROUND if groups.group(3) == "ground" else situation.WATER
		city_id = int(groups.group(4))
		owner = int(groups.group(5))
		self.situation.set_visible_owned_city(location, terrain, city_id, owner)

	def parse_set_visible_city(self, groups):
		location = int(groups.group(1)), int(groups.group(2))
		terrain = situation.GROUND if groups.group(3) == "ground" else situation.WATER
		city_id = int(groups.group(4))
		self.situation.set_visible_city(location, terrain, city_id)

	def parse_set_visible_piece(self, groups):
		location = int(groups.group(1)), int(groups.group(2))
		terrain = situation.GROUND if groups.group(3) == "ground" else situation.WATER
		owner = int(groups.group(4))
		piece_symbol = groups.group(5)
		piece_id = int(groups.group(6))
		piece_type = int(groups.group(7))
		self.situation.set_visible_piece(location, terrain, owner, piece_symbol, piece_id, piece_type)

	def parse_set_explored(self, groups):
		location = int(groups.group(1)), int(groups.group(2))
		terrain = situation.GROUND if groups.group(3) == "ground" else situation.WATER
		self.situation.set_explored(location, terrain)

	def parse_width(self, groups):
		self.situation.set_width(int(groups.group(1)))

	def parse_leave_city(self, groups):
		self.situation.leave_city(int(groups.group(1)), int(groups.group(2)))

	def parse_enter_city(self, groups):
		self.situation.enter_city(int(groups.group(1)), int(groups.group(2)))

	def parse_leave_terrain(self, groups):
		location = int(groups.group(2)), int(groups.group(3))
		self.situation.leave_terrain(int(groups.group(1)), location)

	def parse_height(self, groups):
		self.situation.set_height(int(groups.group(1)))

	def parse_player_id(self, groups):
		self.situation.set_player_id(int(groups.group(1)))

	def parse_pieces_types(self, groups):
		pieces_types = groups.group(1).split(";")
		pieces_types = [x.split("#") for x in pieces_types]
		result = {}
		terrain = {"water": situation.WATER, "ground": situation.GROUND}
		for piece_type in pieces_types:
			info = {}
			info["piece_type_id"] = int(piece_type[0])
			info["name"] = piece_type[1]
			info["symbol"] = piece_type[2]
			info["terrains"] = [terrain[x] for x in piece_type[3].split(":")]
			info["build_time"] = int(piece_type[4])
			info["strength"] = int(piece_type[5])
			info["max_hits"] = int(piece_type[6])
			info["speed"] = int(piece_type[7])
			info["capacity"] = int(piece_type[8])
			info["autonomy"] = None if piece_type[9] == "" else int(piece_type[9])
			info["transportable"] = [] if piece_type[10] == "" else [int(x) for x in piece_type[10].split(":")]
			info["visibility"] = int(piece_type[11])
			info["can_invade"] = piece_type[12] in ["true", "True"]
			result[info["piece_type_id"]] = situation.PieceType(info)
		self.situation.set_pieces_types(result)

	def parse_create_piece(self, groups):
		piece_id = int(groups.group(1))
		piece_type = int(groups.group(2))
		piece_symbol = groups.group(3)
		city_id = int(groups.group(4))
		self.situation.create_piece(piece_id, piece_type, piece_symbol, city_id)

	def parse_delete_piece(self, groups):
		piece_id = int(groups.group(1))
		self.situation.delete_piece(piece_id)

	def parse_move(self, groups):
		piece_id = int(groups.group(1))
		location = int(groups.group(2)), int(groups.group(3))
		self.situation.move(piece_id, location)

	def parse_invade_city(self, groups):
		city_id = int(groups.group(1))
		location = int(groups.group(2)), int(groups.group(3))
		self.situation.invade_city(city_id, location)

	def parse_lose_city(self, groups):
		city_id = int(groups.group(1))
		self.situation.lose_city(city_id)

	def parse_winner(self, groups):
		if int(groups.group(1)) == self.situation.get_player_id():
			print "Winner is you"
		else:
			print "Winner is the ennemy"
		sys.exit(0)


	def parse(self, message):
		for i in range(len(self.rexes)):
			groups = self.rexes[i].match(message)
			if groups:
				self.handlers[i](groups)
				return
		self.situation.show()
		raise Exception("error: not handled: " + message)
