import handler

class Parameters(handler.Handler):

	def __init__(self, the_situation):
		self.the_situation = the_situation
		self.initialized = False

	def update(self):
		if not self.initialized:
			self.initialized = True
			self.influence_city_intensity=10
			self.influence_city_step_length=1
			self.influence_city_nb_steps=5
			self.influence_unexplored_intensity=1
			self.influence_unexplored_step_length=1
			self.influence_unexplored_nb_steps=60
			self.influence_piece_nb_steps=40
			self.influence_piece_intensity_factor=1
			self.influence_piece_step_length_factor=1
			self.transport_ratio=0.75
			# Parameters for city production decision.
			#   - number of invisible tile in the group: it
			#   - number of enemy cities: ec
			#   - number of player cities: pc
			#   - number of free cities: c
			#   - number of enemy pieces: ep
			#   - number of player pieces: pp
			#   - number of player transportable pieces in the group through adjacent groups: pta
			#   - number of enemy pieces in adjacent groups: aep
			#   - number of player pieces in adjacent groups: app
			#   - number of player not transpoort pieces in adjacent groups: apnt
			#   - number of player TAKE transport pieces in adjacent groups: apt
			#   - number of player LAND transport pieces in adjacent groups: apl

			# - For the ratio ((apt + apl) / (apt + apl + apnt + 1))
			#   This ratio identify the difference between the number of transports
			#   and the number of non-transport piece in WATER. If the number of
			#   transport is larger than the number of non-transport pieces, we should
			#   create more non-transport pieces for defense.
			ratio = dict([(x, 0) for x in self.the_situation.piece_types])
			self.production_ratio_apnt = ratio
			for piece_type in self.the_situation.piece_types.values():
				if self.the_situation.WATER in piece_type.terrains and \
						len(piece_type.transportable) == 0 and \
						piece_type.autonomy is None:
					self.production_ratio_apnt[piece_type.piece_type_id] = 3
			# - For the ratio (it / (it + pp + 1)):
			ratio = dict([(x, 0) for x in self.the_situation.piece_types])
			self.production_ratio_it = ratio
			for piece_type in self.the_situation.piece_types.values():
				if piece_type.can_invade \
						and piece_type.autonomy is None:
					self.production_ratio_it[piece_type.piece_type_id] = 2
			# - For the ratio (ec / (ec + pc + 1)):
			ratio = dict([(x, 0) for x in self.the_situation.piece_types])
			self.production_ratio_ec = ratio
			for piece_type in self.the_situation.piece_types.values():
				if piece_type.can_invade \
						and piece_type.autonomy is None:
					self.production_ratio_ec[piece_type.piece_type_id] = 2
			# - For the ratio (c / (c + pc + 1)):
			ratio = dict([(x, 0) for x in self.the_situation.piece_types])
			self.production_ratio_c = ratio
			for piece_type in self.the_situation.piece_types.values():
				if piece_type.can_invade \
						and piece_type.autonomy is None:
					self.production_ratio_c[piece_type.piece_type_id] = 2
			# - For the ratio (ep / (ep + pp + 1)):
			ratio = dict([(x, 0) for x in self.the_situation.piece_types])
			self.production_ratio_ep = ratio
			for piece_type in self.the_situation.piece_types.values():
				if self.the_situation.GROUND in piece_type.terrains \
						and piece_type.autonomy is None:
					self.production_ratio_ep[piece_type.piece_type_id] = 2
			# - For the ratio (aep / (aep + app + 1)):
			# To defend an adjacent group, we must not be a transport.
			ratio = dict([(x, 0) for x in self.the_situation.piece_types])
			self.production_ratio_aep = ratio
			for piece_type in self.the_situation.piece_types.values():
				if self.the_situation.WATER in piece_type.terrains and len(piece_type.transportable) == 0 \
						and piece_type.autonomy is None:
					self.production_ratio_aep[piece_type.piece_type_id] = 4
			# - For the ratio (apt / (apl + apt + 1)):
			ratio = dict([(x, 0) for x in self.the_situation.piece_types])
			self.production_ratio_apt = ratio
			for piece_type in self.the_situation.piece_types.values():
				can_be_transported = False
				for candidat in self.the_situation.piece_types.values():
					if piece_type.piece_type_id in candidat.transportable:
						can_be_transported = True
						break
				if self.the_situation.GROUND in piece_type.terrains and can_be_transported \
						and piece_type.autonomy is None:
					self.production_ratio_apt[piece_type.piece_type_id] = 2
			# - For the ratio (pta / (pta + apl + apt + 1)):
			ratio = dict([(x, 0) for x in self.the_situation.piece_types])
			self.production_ratio_pta = ratio
			for piece_type in self.the_situation.piece_types.values():
				can_transport_ground_piece = False
				for candidat in self.the_situation.piece_types.values():
					if candidat.piece_type_id in piece_type.transportable:
						can_transport_ground_piece = True
						break
				if self.the_situation.WATER in piece_type.terrains and can_transport_ground_piece \
						and piece_type.autonomy is None:
					self.production_ratio_pta[piece_type.piece_type_id] = 2

	def show(self):
		print "PARAMETERS:"
		print vars(self)

	def end(self):
		self.show()
