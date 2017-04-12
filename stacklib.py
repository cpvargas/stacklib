# -*- coding: utf-8 -*-
from astropy.io import fits
from astropy.wcs import wcs
import numpy as np
from astropy.cosmology import FlatLambdaCDM



#Physical Constants
h = 6.626e-34 #Planck Constant, J*s
k_B = 1.381e-23 #Boltzmann Constant, J/K

T_CMB = 2.725 #CMB Temperatura, K
sigma_T = 6.6524e-25 #Thomson Cross Section, cm^2
m_e = 0.511e3 #electron mass, keV/c^2

H_0 = 68. #Hubble Constant, km/s/Mpc
h_70 = H_0/70. 

Omega_m = 0.31 #Matter density parameter
Omega_Lambda = 0.69 #Dark energy density parameter

#Universal Pressure Profile constants
[P_0,c_500,gamma,alpha,beta] = [8.403*h_70**(-3./2.), 1.177, 
                                0.3081, 1.0510, 5.4905]

#setting the cosmology
cosmo = FlatLambdaCDM(H0 = H_0, Om0 = Omega_m)


###############################################################################

class StackMap(object):
    #initializations###########################################################
    def __init__(self, MapFile, WeightsFile, BeamFile, 
                 RA0, RA1, DEC0, DEC1, SourcesFile = None):
        
        #getting headers, data and projections from mapfile
        
        #we will use the 0 representation in which the inferior left corner 
        #pixel is the [0,0] and also the element [0,0] on the array
        #the other representation is the 1 representation in which the 
        #inferior left corner pixel is [1,1]
        
        maphdr = fits.getheader(MapFile, 0)
        
        datamap = fits.getdata(MapFile, 0)
        w = wcs.WCS(maphdr)

        #weights has the same header and projection so we don't load them 
        weightsmap = fits.getdata(WeightsFile, 0)           
        
        #cutting the working area from the full data
        
        #First we find the pixel corresponding to the corners
        verticesworld = np.array([[RA0,DEC0],[RA1,DEC1]])
        rawpix = w.wcs_world2pix(verticesworld, 0) #0 for the representation
 
        #rawpix is the pixel as a float, we want the element on the array
        #for this pixels, to get this we have to round the floats, this is
        #because each pixel is a rectangular area with width and eight
        #the center is the pixel so for example each pixel inside 10.5 to 11.5
        #on width and eight are part of the pixel [10,10]
                            
        pixcoords = np.array(np.round(rawpix,0), dtype=np.int)
        
        #pixcoords = numpy.array([[x0,y0],[x1,y1]])
    
        #mapdata is a numpy array [y,x]
        #crop has to be [y0:y1+1, x0:x1+1] 
        #the +1 is to include up to the x1 and y1 value
        
        x0 = pixcoords[0][0]
        y0 = pixcoords[0][1]
        x1 = pixcoords[1][0]
        y1 = pixcoords[1][1]
        
        self.x0 = x0
        self.y0 = y0
        
        self.datamap = datamap[y0:y1+1,x0:x1+1]
        self.weightsmap = weightsmap[y0:y1+1,x0:x1+1]
        
        #setting fullmap as a copy of datamap (to preserve original data)
        self.fullmap = np.copy(self.datamap)
        
        #if sourcesmap
        if SourcesFile != None:
            sourcesmap = fits.getdata(SourcesFile,0)
            sourcesmaphdr = fits.getheader(SourcesFile,0)
            sourcesw = wcs.WCS(sourcesmaphdr)
            sourcesmap = fits.getdata(SourcesFile, 0)
            srawpix = sourcesw.wcs_world2pix(verticesworld, 0)
            spixcoords = np.array(np.round(srawpix,0), dtype=np.int)
            sx0 = spixcoords[0][0]
            sy0 = spixcoords[0][1]
            sx1 = spixcoords[1][0]
            sy1 = spixcoords[1][1]        
            self.sourcesmap = sourcesmap[sy0:sy1+1,sx0:sx1+1]
            self.fullmap *= self.sourcesmap
        
        #updating in the header the map dimensions and reference pixel
        self.maphdr = maphdr.copy()
        self.maphdr['NAXIS1'] = self.datamap.shape[1]
        self.maphdr['NAXIS2'] = self.datamap.shape[0]
        self.maphdr['CRPIX1'] -= x0
        self.maphdr['CRPIX2'] -= y0
        
        self.w= wcs.WCS(self.maphdr)
        
                                                                                                                     
    ###########################################################################

 
    '''
    About pixels.
    
    In the map array each pixel is represented by an int [x,y] pair
    for example, self.datamap[y,x] gives some value, the value of the [x,y]
    pixel
    
    BUT a pixel in WCS is defined as a region with -0.5 and +0.5 width and
    height, and the center is the int pair value

    So we can imagine for example that the pixel [5,5] containts a region 
    from 4.5 to 5.5 in x and y or we can imagine the same pixel as a region
    from 5 to 6 excluding 6
    
    Here we will make use of the second definition in some parts, for the
    rest we use the first definition
    '''
    
    def getpix(self,RA,DEC):
        coord = np.array([[RA,DEC]], dtype = np.float_)
        px, py = np.array(np.round(self.w.wcs_world2pix(coord, 0),0),
                          dtype=np.int_)[0]
        return px,py    
    
        
    #submaps methods###########################################################
    def setsubmapL(self, L):
        '''
        sets the submap lenght L in pixels
        
        submaps are squares of even lenghts
        '''
        if L%2==0:
            self.submapL = L
        else:
            print("L must be even")
    

    def getsubmap(self,RA,DEC):
        L = self.submapL
        cx,cy = self.getpix(RA,DEC)
        self.submap = np.copy(self.fullmap[cy-L/2:cy+L/2,cx-L/2:cx+L/2])
        self.submapw = np.copy(self.fullmapweights[cy-L/2:cy+L/2,cx-L/2:cx+L/2])
        
    ###########################################################################
    
    def squeezefullmap(self):
        '''
        Multiplies the fullmap pixel-wise, by the number of observations in 
        any single pixel: sqrt(N_obs(x)/N_obs,max)
        '''
        self.N_max = np.amax(self.fullmapweights)
        self.fullmap *= np.sqrt(self.fullmapweights/self.N_max)
        
    #filter methods############################################################
        
        
    ###########################################################################
    def unsqueezefullmap(self):
        '''
        Reverse the pixel-wise multiplication
        '''
        self.fullmap *= np.sqrt(self.N_max/self.fullmapweights)    

    #stacking methods##########################################################
    def setstackmap(self):
        '''
        sets the stackmap: the map in which we will stack submaps
        
        L has to be setted before calling this method
        '''
        
        L = self.submapL

        self.stackmap = np.zeros((L,L))
        self.stackmapw = np.zeros((L,L))
        self.nstack = 0
    
    def stacksubmap(self):
        '''
        stacks the current submap
        
        A submap has to be setted before calling this method
        '''
        self.stackmap += self.submap
        self.nstack += 1
        
    def finishstack(self):
        '''
        Finishes the stacking process dividing the stackmap by the stack of
        weights
        
        Needs at least one call on stacksubmap
        '''
        self.stackmap = self.stackmap/self.nstack
        
        

    ###########################################################################
    
    #plot and saving methods###################################################
    
    def writefullmapfits(self,filename):
        return fits.writeto(filename, self.fullmap, self.maphdr)
    
    def writesubmapfits(self,filename):
        return fits.writeto(filename, self.submap)
        
    ###########################################################################
    
    