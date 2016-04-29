import random
import math

def choose_relevant_random_production(situation, communication, city_id):
	city = situation.player_cities[city_id]
	if city.production is None:
		q, r = city.get_location()
		terrains = []
		for direction in situation.directions:
			neighbor = q + direction[0], r + direction[1]
			if not situation.is_in_map(neighbor):
				continue
			terrain = situation.get_tile(neighbor).get_terrain()
			if terrain not in terrains:
				terrains.append(terrain)
		relevant_pieces_types_id = []
		for piece_type_id in situation.pieces_types:
			piece_type = situation.pieces_types[piece_type_id]
			if any([x in terrains for x in piece_type.terrains]):
				relevant_pieces_types_id.append(piece_type_id)
		piece_type_id = random.choice(relevant_pieces_types_id)
		city.production = piece_type_id
		communication.action("set_city_production %d %d" % (city_id, city.production))
