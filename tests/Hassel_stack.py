"""
Stack of Hasselfield clusters
"""

from datetime import datetime
startTime = datetime.now()

import stacklib as sl

from astropy.io import fits

import os
path = os.environ["HOME"] + '/FILES/' 

hdulist = fits.open(path +'Hassel_inside.fits')
tbdata = hdulist[1].data

m = path +'ACT_148_equ_season_3_1way_v3_src_free.fits'
w = path +'ACT_148_equ_season_3_1way_calgc_strictcuts2_weights.fits'
b = path +'profile_AR1_2009_pixwin_130224.txt'
s = path +'Equa_mask_15mJy.fits'

RA0 = 55.
RA1 = 324.
DEC0 = -1.5
DEC1 = 1.5

M = sl.StackMap(m,w,b,s,RA0,RA1,DEC0,DEC1)

M.squeezefullmap()

M.filterfullmap() 

M.unsqueezefullmap()

M.setsubmapL(16)

M.setstackmap()

for i in range(tbdata.shape[0]):
    M.setsubmap(tbdata["R.A (deg)"][i],tbdata["Dec (deg)"][i])    
    M.stacksubmap()
    
M.finishstack()

print datetime.now() - startTime




    