import re

GROUND=0
WATER=1

class Tile:
	def __init__(self, terrain, location):
		self.terrain = terrain
		self.explored = False
		self.content = None
		self.visible = False
		self.city = False
		self.location = location
	def get_terrain(self):
		return self.terrain
	def get_parent(self):
		return None
	def get_location(self):
		return self.location
	def get_content(self):
		return self.content
	def is_visible(self):
		return self.visible
	def set_terrain(self, terrain):
		self.terrain = terrain
	def set_content(self, content):
		self.content = content
	def set_visible(self):
		self.visible = True
		self.explored = True
	def set_explored(self):
		self.visible = False
		self.explored = True
	def set_city(self):
		self.city = True
	def string(self):
		if not self.explored:
			return "  "
		if not self.visible:
			if self.city:
				return "O "
			if self.terrain == WATER:
				return ". "
			if self.terrain == GROUND:
				return "+ "
		else:
			if self.content == None:
				if self.terrain == WATER:
					return ". "
				else:
					return "+ "
			else:
				return self.content.string()
		assert False

class Piece:
	def __init__(self, piece_id, piece_type, owner, piece_symbol):
		self.piece_id = piece_id
		self.piece_type = piece_type
		self.owner = owner
		self.piece_symbol = piece_symbol
		self.content = []
		self.parent = None
	def get_parent(self):
		return self.parent
	def string(self):
		return self.piece_symbol + ("%d" % self.owner)
	def get_piece_type(self):
		return self.piece_type
	def get_location(self):
		return self.parent.get_location()
	def get_piece_id(self):
		return self.piece_id
	def get_owner(self):
		return self.owner
	def set_parent(self, parent):
		self.parent = parent
	def __repr__(self):
		if self.parent is None:
			return "piece: id=%d owner=%d type=%d loc=None parent=None" % (self.piece_id, self.owner, self.piece_type)
		q, r = self.get_location()
		return "piece: id=%d owner=%d type=%d loc=(%d,%d) parent=%s" % (self.piece_id, self.owner, self.piece_type, q, r, str(self.parent))

class OwnedCity:
	def __init__(self, city_id, parent, owner):
		self.city_id = city_id
		self.owner = owner
		self.parent = parent
		self.production = None
		self.content = []
	def get_parent(self):
		return self.parent
	def get_owner(self):
		return self.owner
	def get_location(self):
		return self.parent.get_location()
	def string(self):
		return "O%d" % self.owner
	def set_parent(self, parent):
		self.parent = parent
	def __repr__(self):
		q, r = self.get_location()
		return "owned_city: owner=%d loc=(%d,%d)" % (self.owner, q, r)

class City:
	def __init__(self, city_id, parent):
		self.city_id = city_id
		self.parent = parent
	def get_location(self):
		return self.parent.get_location()
	def string(self):
		return "O "
	def set_parent(self, parent):
		self.parent = parent
	def __repr__(self):
		q, r = self.get_location()
		return "city: loc=(%d,%d)" % (q, r)

class PieceType:
	def __init__(self, dictionary):
		self.repr = "PieceType: {"
		for k, v in dictionary.items():
			self.repr += str(k) + ":" + str(v) + ", "
			setattr(self, k, v)
		self.repr += "}"

	def __repr__(self):
		return self.repr

class Situation:
	def __init__(self):
		self.player_cities = {}
		self.player_pieces = {}
		self.other_cities = {}
		self.ennemy_pieces = {}
		self.directions = [ (+1,  0), (+1, -1), ( 0, -1), (-1,  0), (-1, +1), ( 0, +1) ]

	def get_free_cities(self):
		result = []
		for city in self.other_cities.values():
			if isinstance(city, City):
				result.append(city)
		return result

	def get_player_pieces(self):
		return self.player_pieces.values()

	def get_player_cities(self):
		return self.player_cities.values()

	def get_ennemy_pieces(self):
		return self.ennemy_pieces.values()

	def get_ennemy_cities(self):
		result = []
		for city in self.other_cities.values():
			if isinstance(city, OwnedCity):
				assert city.get_owner() != self.player_id
				result.append(city)
		return result

	def check(self):
		err = False
		for piece_id in self.ennemy_pieces:
			piece = self.ennemy_pieces[piece_id]
			if piece.owner == self.player_id:
				print "ERR: other piece_id=%d belongs to player!" % piece_id
				err = True
		for piece_id in self.player_pieces:
			piece = self.player_pieces[piece_id]
			if piece.owner != self.player_id:
				print "ERR: player piece_id=%d doesn't belong to player!" % piece_id
				err = True
		for piece_id in self.ennemy_pieces:
			if piece_id in self.player_pieces:
				print "ERR: piece_id=%d in two!" % piece_id
				err = True
		for piece_id in self.player_pieces:
			if piece_id in self.ennemy_pieces:
				print "ERR: piece_id=%d in two!" % piece_id
				err = True
		for piece_id in self.ennemy_pieces:
			piece = self.ennemy_pieces[piece_id]
			parent = piece.get_parent()
			if isinstance(parent, Tile):
				if piece != parent.get_content():
					q, r = parent.get_location()
					print "ERR: piece_id=%d in tile=(%d,%d) but tile contains=%s" % (piece_id, q, r, str(parent.get_content()))
					err = True
		for piece_id in self.player_pieces:
			piece = self.player_pieces[piece_id]
			parent = piece.get_parent()
			if isinstance(parent, Tile):
				if piece != parent.get_content():
					q, r = parent.get_location()
					print "ERR: piece_id=%d in tile=(%d,%d) but tile contains=%s" % (piece_id, q, r, str(parent.get_content()))
					err = True
		for q in range(self.width):
			for r in range(self.height):
				content = self.view[q][r].content
				if content is None:
					continue
				if isinstance(content, Piece):
					if content.piece_id not in self.player_pieces and content.piece_id not in self.ennemy_pieces:
						print "ERR: piece_id=%d on map not associated to other or player!" % content.piece_id
						err = True
		assert not err

	# SET CONFIGURATION

	def set_height(self, height):
		assert not hasattr(self, "height")
		self.height = height
		if hasattr(self, "width"):
			self.view = [[Tile(None, (q, r)) for r in range(self.height)] for q in range(self.width)] 

	def set_width(self, width):
		assert not hasattr(self, "width")
		self.width = width
		if hasattr(self, "height"):
			self.view = [[Tile(None, (q, r)) for r in range(self.height)] for q in range(self.width)] 

	def set_player_id(self, player_id):
		assert not hasattr(self, "player_id")
		self.player_id = player_id

	def set_pieces_types(self, pieces_types):
		assert not hasattr(self, "pieces_types")
		self.pieces_types = pieces_types

	# SHOW VIEW

	def show(self):
		if not hasattr(self, "width") and not hasattr(self, "height"):
			print "width and height not set"
		else:
			print "width:%d height:%d player_id:%d" % (self.width, self.height, self.player_id)
			print "#cities:%d #pieces:%d" % (len(self.player_cities), len(self.player_pieces))
			for r in range(self.height):
				line = "  " * r
				for q in range(self.width):
					line = line + self.view[q][r].string()
				print line

	def split(self, size):
		assert self.width % size == 0
		assert self.height % size == 0
		result = []
		for qi in range(self.width / size):
			for ri in range(self.height / size):
				chunk = [ self.view[qi * size + q][(ri * size):((ri + 1) * size)] for q in range(size) ]
				result.append(chunk)
		return result

	# GET INFORMATION ON SITUATION

	def get_player_id(self):
		return self.player_id

	def get_player_piece_location(self, piece_id):
		assert piece_id in self.player_pieces
		return self.player_pieces[piece_id].get_location()

	def get_tile(self, location):
		assert self.is_in_map(location)
		q, r = location
		return self.view[q][r]

	def get_content(self, location):
		return self.get_tile(location).get_content()

	def get_piece_type(self, piece_type_id):
		assert piece_type_id in self.pieces_types
		return self.pieces_types[piece_type_id]

	def get_player_piece(self, piece_id):
		assert piece_id in self.player_pieces
		return self.player_pieces[piece_id]

	def get_player_city(self, city_id):
		assert city_id in self.player_cities
		return self.player_cities[city_id]

	def is_player_piece(self, piece_id):
		return piece_id in self.player_pieces

	def get_tiles_distance(self, location_a, location_b):
		qa, ra = location_a
		qb, rb = location_b
		return (abs (qa - qb) + abs (qa + ra - qb - rb) + abs (ra - rb)) / 2

	def is_in_map(self, location):
		q, r = location
		return 0 <= q and q < self.width and 0 <= r and r < self.height

	def is_tile_visible(self, location):
		assert self.is_in_map(location)
		q, r = location
		return self.view[q][r].visible

	def can_player_piece_be_on(self, piece_id, location):
		piece = self.get_player_piece(piece_id)
		piece_type = self.get_piece_type(piece.piece_type)
		if not self.is_in_map(location):
			return False
		tile = self.get_tile(location)
		content = tile.get_content()
		if not tile.is_visible() or tile.get_terrain() not in piece_type.terrains:
			return False
		if isinstance(content, City):
			return piece_type.can_invade
		if isinstance(content, OwnedCity):
			return content.owner == self.player_id or piece_type.can_invade
		if content == None:
			return True
		if content.owner != self.player_id:
			return True
		other_piece_type = self.get_piece_type(content.piece_type)
		return piece_type.piece_type_id in other_piece_type.transportable

	def is_tile_none(self, location):
		return self.get_content(location) is None

	def is_tile_player_piece(self, location):
		content = self.get_content(location)
		return content is not None and isinstance(content, Piece) and content.owner == self.player_id

	def is_tile_player_city(self, location):
		content = self.get_content(location)
		return content is not None and isinstance(content, OwnedCity) and content.owner == self.player_id

	def is_tile_ennemy_city(self, location):
		content = self.get_content(location)
		return content is not None and isinstance(content, OwnedCity) and content.owner != self.player_id

	def is_tile_city(self, location):
		content = self.get_content(location)
		return content is not None and isinstance(content, City)

	def is_tile_ennemy_piece(self, location):
		content = self.get_content(location)
		return content is not None and isinstance(content, Piece) and content.owner != self.player_id

	# CHANGES ON SITUATION

	def set_visible_none(self, location, terrain):
		tile = self.get_tile(location)
		content = tile.get_content()
		if content is not None:
			assert isinstance(content, Piece)
			piece_id = content.get_piece_id()
			assert piece_id not in self.player_pieces
			assert piece_id in self.ennemy_pieces
			del self.ennemy_pieces[piece_id]
		tile.set_visible()
		tile.set_content(None)
		tile.set_terrain(terrain)

	def set_visible_owned_city(self, location, terrain, city_id, owner):
		tile = self.get_tile(location)
		tile.set_visible()
		tile.set_city()
		tile.set_terrain(terrain)
		content = self.get_content(location)
		if content == None:
			city = OwnedCity(city_id, tile, owner)
			tile.set_content(city)
			if owner == self.player_id:
				self.player_cities[city_id] = city
			else:
				self.other_cities[city_id] = city
		elif isinstance(content, City):
			assert city_id in self.other_cities
			del self.other_cities[city_id]
			city = OwnedCity(city_id, tile, owner)
			tile.set_content(city)
			if owner == self.player_id:
				self.player_cities[city_id] = city
			else:
				self.other_cities[city_id] = city
		else:
			if owner == self.player_id:
				if content.get_owner() == owner:
					pass
				else:
					assert city_id in self.other_cities
					del self.other_cities[city_id]
					city = OwnedCity(city_id, tile, owner)
					tile.set_content(city)
					self.player_cities[city_id] = city
			else:
				if city_id in self.other_cities:
					del self.other_cities[city_id]
				if city_id in self.player_cities:
					del self.player_cities[city_id]
				city = OwnedCity(city_id, tile, owner)
				tile.set_content(city)
				if owner == self.player_id:
					self.player_cities[city_id] = city
				else:
					self.other_cities[city_id] = city

	def set_visible_city(self, location, terrain, city_id):
		tile = self.get_tile(location)
		tile.set_visible()
		tile.set_city()
		tile.set_terrain(terrain)
		assert not isinstance(tile.get_content(), Piece)
		city = City(city_id, tile)
		self.other_cities[city_id] = city
		tile.set_content(city)

	def set_visible_piece(self, location, terrain, owner, piece_symbol, piece_id, piece_type):
		tile = self.get_tile(location)
		tile.set_terrain(terrain)
		content = tile.get_content()
		if content != None:
			assert isinstance(content, Piece)
			content.set_parent(None)
			if content.get_owner() != self.player_id:
				del self.ennemy_pieces[content.get_piece_id()]
			else:
				assert piece_id == content.get_piece_id()
		if owner == self.player_id:
			assert piece_id in self.player_pieces
			piece = self.player_pieces[piece_id]
		else:
			piece = Piece(piece_id, piece_type, owner, piece_symbol)
			self.ennemy_pieces[piece_id] = piece
		piece.set_parent(tile)
		tile.set_visible()
		tile.set_content(piece)

	def set_explored(self, location, terrain):
		tile = self.get_tile(location)
		content = self.get_content(location)
		tile.set_content(None)
		tile.set_explored()
		if content != None:
			content.set_parent(None)
			if isinstance(content, City):
				del self.other_cities[content.city_id]
			elif isinstance(content, OwnedCity):
				assert content.owner != self.player_id
				assert content.city_id in self.other_cities
				del self.other_cities[content.city_id]
			else:
				assert isinstance(content, Piece)
				assert content.get_owner() != self.player_id
				assert content.get_piece_id() in self.ennemy_pieces
				del self.ennemy_pieces[content.get_piece_id()]

	def create_piece(self, piece_id, piece_type, piece_symbol, city_id):
		assert piece_id not in self.player_pieces
		assert piece_id not in self.ennemy_pieces
		assert city_id in self.player_cities
		piece = Piece(piece_id, piece_type, self.player_id, piece_symbol)
		piece.set_parent(self.get_player_city(city_id))
		self.player_pieces[piece_id] = piece
		self.player_cities[city_id].production = None

	def leave_city(self, piece_id, city_id):
		assert piece_id in self.player_pieces
		assert piece_id not in self.ennemy_pieces
		assert city_id in self.player_cities
		piece = self.get_player_piece(piece_id)
		piece.set_parent(None)

	def enter_city(self, piece_id, city_id):
		assert piece_id in self.player_pieces
		assert piece_id not in self.ennemy_pieces
		assert city_id in self.player_cities
		piece = self.get_player_piece(piece_id)
		city = self.get_player_city(city_id)
		assert piece.get_parent() == None
		piece.set_parent(city)

	def leave_terrain(self, piece_id, location):
		assert piece_id in self.player_pieces
		assert piece_id not in self.ennemy_pieces
		piece = self.get_player_piece(piece_id)
		tile = self.get_tile(location)
		assert tile.get_content() == piece
		assert piece.get_location() == location
		piece.set_parent(None)
		tile.set_content(None)

	def move(self, piece_id, location):
		assert piece_id in self.player_pieces
		pass

	def invade_city(self, city_id, location):
		tile = self.get_tile(location)
		tile.set_visible()
		tile.set_city()
		content = self.get_content(location)
		assert content != None
		assert not isinstance(content, Piece)
		assert not isinstance(content, OwnedCity) or content.get_owner() != self.player_id
		del self.other_cities[city_id]
		city = OwnedCity(city_id, tile, self.player_id)
		tile.set_content(city)
		self.player_cities[city_id] = city

	def lose_city(self, city_id):
		assert city_id in self.player_cities
		city = self.player_cities[city_id]
		parent = city.get_parent()
		assert isinstance(parent, Tile)
		del self.player_cities[city_id]
		city = City(city_id, parent)
		parent.set_content(city)
		self.other_cities[city_id] = city

	def delete_piece(self, piece_id):
		assert piece_id in self.player_pieces
		piece = self.get_player_piece(piece_id)
		assert piece.get_parent() == None
		del self.player_pieces[piece_id]
