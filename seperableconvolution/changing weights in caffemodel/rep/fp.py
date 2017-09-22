import numpy as np

r = np.zeros((96,3,7,7))
i=0
w=0
while (i<96):
	j=0
	while (j<3):
		r[i][j] = 1
		w=w+1
		j=j+1

	i=i+1

