import numpy as np
f = open('seperable.txt')
lines = f.readlines()
for ln in lines[0:len(lines)]:
	line=ln.strip().split()
b = line
a=np.zeros((285,7,7))
i=0
l=0
while (i<285):
	j=0
	while (j<7):
		k=0
		while (k<7):
			a[i][j][k] = b[l]			
			k=k+1
			l=l+1
		j=j+1
	i=i+1
t=np.zeros((285,7,7))
i=0
while (i<285):
	j=0
	while (j<7):
		k=0
		while (k<7):
			t[i][j][k] = t[i][k][j] = (a[i][j][k]+a[i][k][j])/2			
			k=k+1
			l=l+1
		j=j+1
	i=i+1
while (i<285):
	t[i][1][1]=t[i][7][7]=(t[i][7][7]+t[i][1][1])/2
	t[i][2][2]=t[i][6][6]=(t[i][6][6]+t[i][2][2])/2
	t[i][3][3]=t[i][5][5]=(t[i][5][5]+t[i][3][3])/2
	i=i+1
m=0
while (m<285):
	print(a[m])
	m=m+1
#file = open('outtt.txt','w')
#file.write(t[0])
#file.close()
f.close()

