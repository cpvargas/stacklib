'''
Simple script to test some functions and methods

Creates a fake catalog of 100 sources at [RA,DEC] inside fullmap area
at each position pastes beammaps of amplitude -150 on a zero fullmap, 
then it performs a stack of all beams.
'''

from datetime import datetime
startTime = datetime.now()

import sys
import os
sys.path.append(os.path.abspath("../"))

import stacklib as sl
import numpy as np

path = os.environ["HOME"] + '/FILES/' 

m = path + 'ACT_148_equ_season_3_1way_v3_src_free.fits'
w = path + 'ACT_148_equ_season_3_1way_calgc_strictcuts2_weights.fits'
b = path + 'profile_AR1_2009_pixwin_130224.txt'
s = path + 'Equa_mask_15mJy.fits'

RA0 = 57.5
RA1 = 308.5
DEC0 = -1.5
DEC1 = 1.5

M = sl.StackMap(m,w,b,s,RA0,RA1,DEC0,DEC1)

M.setfullmap(boostFT = 'False')

M.fullmap = np.abs(M.fullmap*0)

cat = sl.fakecatalog(100,RA0,RA1,DEC0,DEC1,0.2)

psize = np.abs(M.maphdr['CDELT1'])

Beam = -150*sl.beam(b,psize,10)
              
for item in cat:
    loc = M.getpix(item[0],item[1])
    M.fullmap = sl.pastemap(M.fullmap, Beam, loc)
    
M.setsubmapL(16)

M.setstackmap()

for item in cat:
    M.setsubmap(item[0],item[1])    
    M.stacksubmap()
    
M.finishstack()

print datetime.now() - startTime