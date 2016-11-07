'''
Script to test stacking after filtering

Creates a fake catalog of 1000 sources
Add beams with -150 amplitude at catalog positions
pixel-wise multiplication
filters
reverse pixel-wise multiplication
stacks the catalog
'''

from datetime import datetime
startTime = datetime.now()

import stacklib as sl
import numpy as np

#Assuming files are at the folder FILES in the home directory
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

cat = sl.fakecatalog(1000)

psize = np.abs(M.maphdr['CDELT1'])

Beam = -150*sl.beam(b,psize,15)

for item in cat:
    loc = M.getpix(item[0],item[1])
    M.fullmap = sl.pastemap(M.fullmap, Beam, loc)

M.squeezefullmap()

M.filterfullmap() 

M.unsqueezefullmap()

M.setsubmapL(16)

M.setstackmap()

for item in cat:
    M.setsubmap(item[0],item[1])    
    M.stacksubmap()
    
M.finishstack()

print datetime.now() - startTime