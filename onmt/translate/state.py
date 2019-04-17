import torch
from copy import deepcopy
import re
class sequence_state(object):
	def __init__(self):
		self.stack = []
		self.var = {'X':0, 'E':0, 'S':0, 'T':0, 'P':0, 'O':0}
		self.first = True
	def print(self):
		print(self.stack)
		print(self.var)
		#print(self.first)
class BB_sequence_state(object):
	def __init__(self, itos, stoi, mb_device, batch_size, beam_size, eos=3):
		self.states = [sequence_state() for i in range(batch_size*beam_size)] # empty
		self.itos = itos
		self.stoi = stoi
		self.device = mb_device
		self.batch_size = batch_size
		self.beam_size = beam_size
		self.unmask = 1
		self.mask = 0
		self.eos = eos
	def get_mask(self):
		masks = []
		expanded_masks = []
		for state in self.states:
			mask, expanded_mask = self.get_mask_one(state)
			masks.append(mask.unsqueeze(0))
			expanded_masks.append(expanded_mask.unsqueeze(0))
		return torch.cat(masks, 0), torch.cat(expanded_masks, 0)

	def get_mask_one(self, state):
		stack = state.stack
		mask = torch.full([len(self.itos)], self.mask, dtype=torch.float, device=self.device)
		v = self.mask
		if len(stack) == 0: # empty
			if state.first:
				self.allow_drs(mask)
				self.allow_sdrs(mask)
			else:
				self.allow_end(mask)

		elif stack[-1][0] == self.stoi["DRS("]: 
			self.allow_cond(mask, state.var)
			if stack[-1][1] > 0:
				self.allow_close(mask)

		elif stack[-1][0] == self.stoi["SDRS("]:
			if stack[-1][1] < 2:
				self.allow_drs(mask)
				self.allow_sdrs(mask)
			else:
				if stack[-1][1] == 1000: # start predict discourse relations
					self.allow_normal_cond(mask)
				else:
					self.allow_normal_cond(mask)
					self.allow_drs(mask)
					self.allow_sdrs(mask)

		elif self.itos[stack[-1][0]] in ["OR(", "DIS(", "DUP(", "IMP("]:
			if stack[-1][1] < 2:
				self.allow_drs(mask)
				self.allow_sdrs(mask)
			else:
				self.allow_close(mask)

		elif self.itos[stack[-1][0]]in ["NEC(", "POS(", "NOT("]:
			if stack[-1][1] < 1:
				self.allow_drs(mask)
				self.allow_sdrs(mask)
			else:
				self.allow_close(mask)

		elif re.match("^P[0-9]+\($", self.itos[stack[-1][0]]):
			if stack[-1][1] < 1:
				self.allow_drs(mask)
				self.allow_sdrs(mask)
			else:
				self.allow_close(mask)

		elif stack[-1][0] == self.stoi["Ref("]:
			if stack[-1][1] == 0:
				self.allow_ref0(mask, state.var)
			elif stack[-1][1] == 1:
				self.allow_ref1(mask)
			else:
				self.allow_close(mask)

		#elif stack[-1][0] == self.stoi["Pred("]:
		else:
			if stack[-2][0] == self.stoi["SDRS("]:
				if stack[-1][1] < 2:
					self.allow_dv(mask, stack[-2][1])
				else:
					self.allow_close()
			else:
				if stack[-1][1] == 0:
					self.allow_ref0(mask, state.var)
				elif stack[-1][1] == 1000:
					self.allow_close(mask)
				else:
					self.allow_anyv(mask, state.var)
					self.allow_unk(mask)
					self.allow_close(mask)
					v = self.unmask
		#print("expanded_mask", v)
		expanded_mask = torch.full([1], v, dtype=torch.float, device=self.device)
		return mask, expanded_mask
	def index_select(self, select_indices):
		states = []
		for index in select_indices:
			states.append(deepcopy(self.states[index]))
		self.states = states

	def update(self, actions):
		for i, act in enumerate(actions):
			self.states[i] = self.update_one(self.states[i], act)

	def update_beam(self, actions, selects, scores):
		"""
		selects = selects.data.tolist()
		actions = actions.data.tolist()
		scores = scores.exp().data.tolist()
		assert len(selects) == len(actions) == len(self.states) == len(scores)

		states = []
		for i, j, k in zip(selects, actions, scores):
			print(i,j,k)
			self.states[i].print()
			if k <= 0.0:
				states.append(deepcopy(self.states[i]))
			else:
				states.append(self.update_one(self.states[i], j))
		self.states = states
		"""
		assert len(actions) == len(selects)
		if scores:
			assert len(actions) == len(scores)
		states = []
		for i, (sel, act) in enumerate(zip(selects, actions)):
			if scores and scores[i] <= 0:
				states.append(deepcopy(self.states[i]))
			else:
				states.append(self.update_one(self.states[sel], act))
			#self.states[i].print()
		self.states = states

	def update_one(self, state, act):
		nstate = deepcopy(state)
		nstate.first = False
		stack = nstate.stack
		if act == self.eos:
			pass
		elif act == self.stoi[")"]:
			if (self.itos[stack[-1][0]] not in ["DRS(", "SDRS("]) and stack[-2][0] == self.stoi["SDRS("]:
				stack.pop()
				stack[-1][1] = 1000
			else:
				stack.pop()
				if len(stack) != 0:
					stack[-1][1] += 1
		elif act < len(self.itos) and self.itos[act][-1] == "(" :
			stack.append([act, 0])
		else:
			if act < len(self.itos) and re.match("^[anvr]\.[0-9][0-9]$", self.itos[act]):
				stack[-1][1] = 1000
			else:
				stack[-1][1] += 1

			if stack[-1][0] == self.stoi["Ref("]:
				act_name = self.itos[act]
				if act_name in list("XESTPO"):
					nstate.var[act_name] += 1
		return nstate

	def allow_drs(self, mask):
		mask[self.stoi["DRS("]] = self.unmask
	def allow_sdrs(self, mask):
		mask[self.stoi["SDRS("]] = self.unmask
	def allow_end(self, mask):
		mask[self.eos] = self.unmask
	def allow_cond(self, mask, d):
		for act, act_name in enumerate(self.itos):
			if act_name[-1] == "(" and act_name not in ["DRS(", "SDRS("]:
				mask[act] = self.unmask
			if re.match("^P[0-9]\($", act_name) and int(act_name[1:-1]) < d["P"]:
				mask[act] = self.unmask
	def allow_close(self, mask):
		mask[self.stoi[")"]] = self.unmask
	def allow_unk(self, mask):
		mask[self.stoi["<unk>"]] = self.unmask
	def allow_normal_cond(self, mask):
		for act, act_name in enumerate(self.itos):
			if act_name[-1] == "(" and act_name not in ["DRS(", "SDRS(", "OR(", "DIS(", "DUP(", "IMP(","NEC(", "POS(", "NOT("]:
				mask[act] = self.unmask
	def allow_ref0(self, mask, d):
		for act, act_name in enumerate(self.itos):
			if re.match("^B[0-9]*$", act_name):
				mask[act] = self.unmask
			if act_name == "O":
				mask[act] = self.unmask
			if re.match("^O[0-9]+$", act_name) and int(act_name[1:]) < d["O"]:
				mask[act] = self.unmask
	def allow_ref1(self, mask):
		for act_name in list("XESTP"):
			mask[self.stoi[act_name]] = self.unmask
	def allow_dv(self, mask, ndrs):
		for act, act_name in enumerate(self.itos):
			if re.match("^K[0-9]+$", act_name) and int(act_name[1:]) < ndrs:
				mask[act] = self.unmask
	def allow_anyv(self, mask, d):
		for act, act_name in enumerate(self.itos):
			if act < 4:
				continue
			if act_name == ")":
				continue
			if act_name[-1] == "(":
				continue
			if act_name in list("XESTPBO"):
				continue
			if re.match("^[BO][0-9]*$",act_name):
				continue
			if re.match("^[XESTP][0-9]+$",act_name):
				if int(act_name[1:]) < d[act_name[0]]:
					mask[act] = self.unmask
				continue
			mask[act] = self.unmask









