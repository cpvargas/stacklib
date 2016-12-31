import stacklib as sl
from astropy.io import fits
import numpy as np

from datetime import datetime

#import os
#path = os.environ["HOME"] + '/FILES/' 

path = u'C:\\FILES\\'

m = path + 'ACT_148_equ_season_3_1way_v3_summed.fits'
w = path + 'ACT_148_equ_season_3_1way_calgc_strictcuts2_weights.fits'
b = path + 'profile_AR1_2009_pixwin_130224.txt'
s = path + 'Equa_mask_15mJy.fits'

RA0 = 57.5
RA1 = 308.5
DEC0 = -1.5
DEC1 = 1.5

M = sl.StackMap(m,w,b,s,RA0,RA1,DEC0,DEC1)
M.setsubmapL(32)
M.setstackmap()

t = fits.open(path + 'radio_quiet.fit')

tbdata = t[1].data

startTime = datetime.now() 

def stack(ClustersRange,Binname):
    Ms = []
    Ns = []

    for i in ClustersRange:
        Ngals = tbdata['GM_SCALED_NGALS'][i]
        Ns.append(Ngals)
        
        z = tbdata['PHOTOZ'][i]
        RA = tbdata['RA'][i]
        DEC = tbdata['DEC'][i]
        
        M.setfullmap()
        
        M.getbeammap()
        
        Mass = sl.M200toM500*sl.M_200_from_N_gals(Ngals)
        Ms.append(Mass)
        
        M.gettaumap(Mass,z)
        M.getnormconv()
    
        M.setsubmap(RA,DEC)
        
        M.squeezefullmap()
    
        M.filterfullmap() 
        
        M.unsqueezefullmap() 
        
        M.getsubmap()
        M.getsubmapSZ()
        
        M.stacksubmap()
        
    M.finishstack()
    
    hdu = fits.PrimaryHDU(M.stackmap)
    hdu.writeto('{}.fits'.format(Binname))
    print '{}'.format(Binname)
    print 'Y_SZ = ' + str(M.stackmap[16,16])
    print 'M.SN = ' + str(M.SN)
    print 'Ngals =' + str(Ns[0]) + '-' + str(Ns[len(Ns)-1])
    print '<M> = ' + str(np.mean(Ms))
    print datetime.now() - startTime
    

#stack(range(0,5),'Bin_0_5')
#stack(range(35,125),'Bin_3')
#stack(range(35,150),'Bin_35_150')
#stack(range(35,175),'Bin_35_175')
#stack(range(35,200),'Bin_35_200')
#stack(range(35,225),'Bin_35_225')
#stack(range(35,250),'Bin_35_250')

#stack(range(200,1371),'Bin_200_1371')
#stack(range(250,1371),'Bin_250_1371')
#stack(range(150,1371),'Bin_150_1371')

#stack(range(0,5),'Bin_0_5')
#stack(range(5,35),'Bin_5_35')
#stack(range(35,125),'Bin_35_125')
#stack(range(125,1371),'Bin_125_1371')
#stack(range(35,150),'Bin_35_150')

#pendiente
#stack(range(35,200),'Bin_35_200')
#stack(range(35,250),'Bin_35_250')
#stack(range(35,300),'Bin_35_300')
#stack(range(35,350),'Bin_35_350')


stack(range(0,5),'Bin_0_5')