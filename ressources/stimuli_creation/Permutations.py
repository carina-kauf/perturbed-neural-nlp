

from scipy.stats import kendalltau
from random import randint

#add seeds for reproducibility
import numpy as np
import random
np.random.seed(42)
random.seed(42)

def kendall_distance(x,y):
	"""
		Use kendalltau to compute kendall distance
		http://en.wikipedia.org/wiki/Kendall_tau_distance
	"""
	assert len(x) == len(y)
	n = len(x)
	
	tau, pv = kendalltau(x,y)
	
	# concordant_minus_discordant
	concordant_minus_discordant = tau * (0.5)*n*(n-1)
	
	# c + d = n(n-1)/2
	# so c+d-(c-d) = 2d = n(n-1)/2 - concordant_minus_discordant
	d = (n*(n-1)/2 - concordant_minus_discordant)/2
	
	return round(d) # round to fix numerical precision errors

def make_permutation_with_distance(d, n):
	"""
		Make a permutation on n elements whose distance to 0,1,2,...,n
		is AT LEAST d
		
		Note: we sometimes may be more than d, as we return the first time
		a swap gets us above or equal to d. Sometimes, you can't get a given number...		
	"""
	np.random.seed(42)
	random.seed(42)

	assert n >= 2
	assert d <= n*(n-1)/2
	
	nar = list(range(n))
	xar = list(range(n))
	
	## TODO: We could make this faster by running at least d swaps first
	while kendall_distance(nar, xar) < d:
		#print kendall_distance(nar, xar)
		# swap two elements
		i = randint(0,n-2)
		xar[i], xar[i+1] = xar[i+1], xar[i]
		
	return xar

if __name__ == "__main__":

	N = 11
	
	for _ in range(25):
		ar = make_permutation_with_distance(8, N)
		print(kendall_distance( ar, range(N) ), ar)
