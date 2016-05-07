import random

import situation
import algos

# Cette fonction permet de choisir aleatoirement un type d'unite a produire, en
# fonction de la localisation de la ville (pas de bateau en plein milieu d'un
# continent).
# La construction d'un transport doit etre conditionne par le nombre de transports
# pleins sur le group sur lequel se trouve la ville.
def choose_relevant_random_production(situation, communication, city_id):
	city = situation.player_cities[city_id]
	if city.production is None:
		# Recuperation des types de terrain autour pour construire une piece
		# qui peut au moins sortir de la ville (pas de bateau en plein milieu du
		# continent !).
		q, r = city.get_location()
		terrains = []
		for direction in algos.directions:
			neighbor = q + direction[0], r + direction[1]
			if situation.is_in_map(neighbor):
				terrain = situation.get_tile(neighbor).terrain
				if terrain not in terrains:
					terrains.append(terrain)
		# Recuperation des types de pieces pertinents pour la production.
		relevant_piece_types_id = []
		for piece_type_id in situation.piece_types:
			piece_type = situation.piece_types[piece_type_id]
			if any([x in terrains for x in piece_type.terrains]):
				relevant_piece_types_id.append(piece_type_id)

		# TODO
		#relevant_piece_types_id = [x for x in relevant_piece_types_id if x == 0 or x == 2 or x == 3]

		piece_type_id = random.choice(relevant_piece_types_id)
		city.production = piece_type_id
		communication.action("set_city_production %d %d" % (city_id, city.production))
