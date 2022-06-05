import numpy as np
import time
import bisect
import sys
import random



class Score:

	def __init__(self, X, penalty_discount=2.0):

		self.n, self.p = X.shape
		self.c = penalty_discount / 2.0
		self.cov = np.corrcoef(X.T, dtype=np.double)
		self.cache = {}

		# for i in range(self.p):
		# 	self.cache[i] = {}

	def get_n(self):
		return self.n

	def get_p(self):
		return self.p

	def set_penalty_discount(self, penalty_discount):
		self.c = penalty_discount / 2.0

	def score(self, y, X):

		S = sorted(X)
		key = tuple(S)

		if key not in self.cache:
			self.cache[key] = np.linalg.slogdet(self.cov[np.ix_(S, S)])[1]
		log_prob = self.cache[key]
		bisect.insort(S,y)
		key = tuple(S)
		if key not in self.cache:
			self.cache[key] = np.linalg.slogdet(self.cov[np.ix_(S, S)])[1]
		log_prob -= self.cache[key]

		# if key not in self.cache[y]:
		# 	self.cache[y][key] = - np.linalg.slogdet(1 - np.dot(np.dot(self.cov[np.ix_([y], X)], np.linalg.inv(self.cov[np.ix_(X, X)])), self.cov[np.ix_(X, [y])]))[1]
		# log_prob = self.cache[y][key]

		return self.n/2 * log_prob - self.c * len(X) * np.log(self.n)



class Order:

	def __init__(self, score):

		p = score.get_p()

		self.order = list(range(p))
		self.parents = {}
		self.local_scores = {}
		self.edges = 0

		random.shuffle(self.order)

		for i in range(p):
			y = self.order[i]
			self.parents[y] = []
			self.local_scores[y] = score.score(y, [])

	def get(self, i):
		return self.order[i]

	def set(self, i, y):
		self.order[i] = y

	def index(self, y):
		return self.order.index(y)

	def insert(self, i, y):
		self.order.insert(i, y)

	def pop(self, i=-1):
		return self.order.pop(i)

	def get_parents(self, y):
		return self.parents[y]

	def set_parents(self, y, y_parents):
		self.parents[y] = y_parents

	def get_local_score(self, y):
		return self.local_scores[y]

	def set_local_score(self, y, local_score):
		self.local_scores[y] = local_score

	def get_edges(self):
		return self.edges

	def set_edges(self, edges):
		self.edges = edges

	def bump_edges(self, bump):
		self.edges += bump



def grasp(X, depth):

	runtime = time.perf_counter()

	n, p = X.shape
	score = Score(X)
	order = Order(score)

	for i in range(p):
		y = order.get(i)
		y_parents = order.get_parents(y)

		candidates = [order.get(j) for j in range(0, i)]
		grow(y, y_parents, candidates, score)
		local_score = shrink(y, y_parents, score)
		order.set_local_score(y, local_score)
		order.bump_edges(len(y_parents))

	while dfs(depth - 1, set(), [], order, score):
		sys.stdout.write('\rGRaSP edge count: %i   ' % order.get_edges())
		sys.stdout.flush()

	runtime = time.perf_counter() - runtime

	sys.stdout.write(' \nGRaSP completed in: %.2fs \n' % runtime)
	sys.stdout.flush()

	graph = np.zeros([p,p], 'uint8')
	for y in range(p):
		for x in order.get_parents(y):
			graph[y,x] = 1

	return graph



def dfs(depth, flipped, history, order, score):

	cache = [{}, {}, {}, 0]

	indices = list(range(score.get_p()))
	random.shuffle(indices)

	for i in indices:
		y = order.get(i)
		y_parents = order.get_parents(y)
		random.shuffle(y_parents)

		for x in y_parents:
			j = order.index(x)

			covered = set([x] + order.get_parents(x)) == set(y_parents)
			if len(history) > 0 and not covered:
				continue

			for k in range(j, i + 1):
				z = order.get(k)
				cache[0][k] = z
				cache[1][k] = order.get_parents(z)[:]
				cache[2][k] = order.get_local_score(z)
			cache[3] = order.get_edges()

			tuck(i, j, order)
			edge_bump, score_bump = update(i, j, order, score)

			if score_bump > 0:
				order.bump_edges(edge_bump)
				# print(history)
				return True

			if score_bump == 0:
				flipped = flipped ^ set([tuple(sorted([x, z])) for z in order.get_parents(x) if order.index(z) < i])

				if len(flipped) > 0 and flipped not in history:

					history.append(flipped)

					if depth > 0 and dfs(depth - 1, flipped, history, order, score):
						return True
					
					del history[-1]



			for k in range(j, i + 1):
				z = cache[0][k]
				order.set(k, z)
				order.set_parents(z, cache[1][k])
				order.set_local_score(z, cache[2][k])
			order.set_edges(cache[3])

	return False



def update(i, j, order, score):

	edge_bump = 0
	old_score = 0
	new_score = 0

	for k in range(j, i + 1):
		z = order.get(k)
		z_parents = order.get_parents(z)

		edge_bump -= len(z_parents)
		old_score += order.get_local_score(z)

		candidates = [order.get(l) for l in range(0, k)]

		# z_parents.clear()

		# for w in [w for w in candidates]:
			# z_parents.append(w)

		for w in [w for w in z_parents if w not in candidates]:
			z_parents.remove(w)
		shrink(z, z_parents, score)

		for w in z_parents:
			candidates.remove(w)

		grow(z, z_parents, candidates, score)

		local_score = shrink(z, z_parents, score)
		order.set_local_score(z, local_score)

		edge_bump += len(z_parents)
		new_score += local_score

	return edge_bump, new_score - old_score



def grow(y, y_parents, candidates, score):

	best = score.score(y, y_parents)

	add = None
	checked = []
	while add != None or len(candidates) > 0:

		if add != None:
			checked.remove(add)
			y_parents.append(add)
			candidates = checked
			checked = []
			add = None

		while len(candidates) > 0:

			x = candidates.pop()
			y_parents.append(x)
			current = score.score(y, y_parents)
			y_parents.remove(x)
			checked.append(x)

			if current > best:
				best = current
				add = x

	return best



def shrink(y, y_parents, score):

	best = score.score(y, y_parents)

	remove = None
	checked = 0
	while remove != None or checked < len(y_parents):

		if remove != None:
			y_parents.remove(remove)
			checked = 0
			remove = None

		while checked < len(y_parents):
			x = y_parents.pop(0)
			current = score.score(y, y_parents)
			y_parents.append(x)
			checked += 1

			if current > best:
				best = current
				remove = x

	return best



def tuck(i, j, order):

	ancestors = []
	get_ancestors(order.get(i), ancestors, order)

	shift = 0

	for k in range(j + 1, i + 1):
		if order.get(k) in ancestors:
			order.insert(j + shift, order.pop(k))
			shift += 1



def get_ancestors(y, ancestors, order):

	ancestors.append(y)

	for x in order.get_parents(y):
		if x not in ancestors:
			get_ancestors(x, ancestors, order)
