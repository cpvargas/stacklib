''' 
Does N times random stacks of X maps of large L in pixels.

At each stacks it gets the central temperature, makes a histogram for all 
stacks, then fits a normal distribution for the histogram. 
'''

N = 100000
X = 10
L = 16

import stacklib as sl
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
path = os.environ["HOME"] + '/FILES/' 

m = path + 'ACT_148_equ_season_3_1way_v3_src_free.fits'
w = path + 'ACT_148_equ_season_3_1way_calgc_strictcuts2_weights.fits'
b = path + 'profile_AR1_2009_pixwin_130224.txt'
s = path + 'Equa_mask_15mJy.fits'

RA0 = 55.
RA1 = 324.
DEC0 = -1.5
DEC1 = 1.5

M = sl.StackMap(m,w,b,s,RA0,RA1,DEC0,DEC1)
M.squeezefullmap()
M.filterfullmap() 
M.unsqueezefullmap()

DeltaTs = []

def onestack(X,L):
    cat = sl.fakecatalog(X)
    M.setsubmapL(L)
    M.setstackmap()
    for item in cat:
        M.setsubmap(item[0],item[1])
        M.stacksubmap()
    M.finishstack()
    return DeltaTs.append(M.stackmap[L/2,L/2])

for i in range(N):
    onestack(X,L)

# histogram
n, bins, patches = plt.hist(DeltaTs,bins=50,normed = 1, facecolor = 'blue')

# best fit of data
(mu, sigma) = norm.fit(DeltaTs)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)


plt.xlabel('Temperature (microKelvin)')
plt.ylabel('Probability Density')

plt.show()