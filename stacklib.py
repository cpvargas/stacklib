# -*- coding: utf-8 -*-
from astropy.io import fits
from astropy.wcs import wcs
import numpy as np
from matplotlib import pyplot as plt #needs wcsaxes package for plots
from scipy import interpolate
from scipy import ndimage
from scipy.misc import imresize
from matplotlib.patches import Circle

def beam(beamfile, pixsize, radius):
    '''
    generates a 2D beam profile: 2Darray of size (2*radius+1, 2*radius+1)
    
    beamfile: file containing the unit normalized beam radial profile
    pixsize: size of the pixel in degrees
    radius: desired extension of the profile in pixels  
    '''
    
    x, y = np.loadtxt(beamfile, unpack = True)
    
    #adding 0. at x=1. and 0. at x=3 
    #for cases when we need values at x>1
    x = np.append(x,[1.,3.])
    y = np.append(y,[0.,0.])
    
    #interpolation
    Beamfunction = interpolate.interp1d(x, y)
    
    #we want a map like
    #
    #         +radius
    #            .
    #            .
    #            .
    #-radius ... 0 ... +radius
    #            .
    #            .
    #            .
    #        -radius
    #    

    r = radius
    y,x = np.ogrid[-r:r+1,-r:r+1]

    #we evaluate the function at each pixel since the function is radial
    #we use the typical r = sqrt(x**2+y**2). But a scale change has to 
    #be made since pixels have certain width and height in ACT equatorial 
    #maps, to change the scale we only need to multiply x and y by the
    #pixel size

    Beam = Beamfunction(np.sqrt((pixsize*x)**2+(pixsize*y)**2))
    return Beam

def beammap(beamfile, mapheader):
    '''
    Generates a Beam Map: an array of the size of the map with the full 
    Beam 2D profile of the map at his center

    beamfile: file containing the unit normalized beam radial profile
    mapheader: header of the map
    '''
    pixsize = abs(mapheader['CDELT1'])
    radius = int(1./pixsize)
    
    Beam = beam(beamfile, pixsize, radius)

    #we create a zero map with datamap size
    shape_x = mapheader['NAXIS1']
    shape_y = mapheader['NAXIS2']

    Beammap = np.zeros((shape_y,shape_x))
    cx = int(shape_x/2)
    cy = int(shape_y/2)

    #at the center of it we paste the beammap
    Beammap[cy-radius:cy+radius+1,cx-radius:cx+radius+1] = Beam
    return Beammap

def matchedfilter(inmap,inmap_beam):
    mfft = np.abs(np.fft.fft2(inmap))
    mfft2_inv = 1./mfft**2
        
    median = np.median(np.copy(mfft2_inv))
        
    mfft2_inv[mfft2_inv>median/10] = median
    mfft2_inv[mfft2_inv<median*10] = median
            
    mfft2_inv = ndimage.gaussian_filter(mfft2_inv, sigma=5)   
    
    #pixel size in radians!!!
    tpix = 0.00825*2*np.pi/360.
    
    ly = np.fft.fftfreq(inmap.shape[0], tpix)
    lx = np.fft.fftfreq(inmap.shape[1], tpix)

    #These arrays are like
    #0,delta_l,2delta_l,..., nyquist,nyquist-delta_l,...,-delta_l 

    #Constructing the taper function, which is sin^5(pi*l/2400) up to 1200 
    #and then 1 for other values
    xs = np.linspace(0.,1200.,10000)
    ys = (np.sin(np.pi*xs/2400.))**5
    #adding 1s after 1200
    xs = np.append(xs,500000.)
    ys = np.append(ys,1.0)
    #1D interpolation
    sin5 = interpolate.interp1d(xs,ys)
    
    #For the 2D map we use lx and ly, but ly must be reshaped
    ly = ly.reshape((ly.shape[0],1))
    Sin5Map = sin5(np.sqrt(lx**2+ly**2))
    
    #applying the taper
    mfft2_inv = mfft2_inv*Sin5Map
    
    beam_fft = np.fft.fft2(inmap_beam)
    beam_fft_c = np.conjugate(beam_fft)
    
    mfilt = (beam_fft_c*mfft2_inv)
    mfilt_norm = np.sum(np.abs(beam_fft)**2*mfft2_inv)/mfilt.size
    
    mfilt /= mfilt_norm
    return mfilt  

def filtermap(data,filt):
    map_fft = np.fft.fft2(data)
    filtmap_fft = map_fft * filt
    filtmap = np.fft.ifft2(filtmap_fft)    
    return np.fft.fftshift(np.real(filtmap))


def pastemap(inmap, minimap, location):
    '''
    Pastes a minimap centered at location in inmap .
    
    minimap must be smaller than inmap and it has to fit inside it centered 
    at location. 
    
    Also we use an odd definition of minimaps, shape should be (2*r+1,2*r+1)
    
    location is in pixels of inmap as np.array([x,y], dtype = np.int_)
    '''
    r = int(minimap.shape[0]//2)
    cx = location[0]
    cy = location[1]
    inmap[cy-r:cy+r+1,cx-r:cx+r+1] += minimap
    return inmap
    
def fakecatalog(num):
    '''
    Generates a fake catalog of sources at [RA,DEC]
    NEED TO FIX THE RANGES TO BE AUTOMATIC
    '''
    fakecat = []
    for i in range(num):
        RA = np.random.uniform(324.+1.,360.+55.-1.)
        if RA>=360:
            RA -= 360
        DEC = np.random.uniform(-1.,1.)
        fakecat.append([RA,DEC])
    return fakecat
    
def zeroPadMap(inmap, newLength):
    '''
    Enlarges map to newLenght preserving values
    '''
    
    mNx_old, mNy_old = inmap.shape
    inMap = inmap[:2*(mNx_old/2), :2*(mNy_old/2)]
        
    dx = newLength-mNx_old
    dy = newLength-mNy_old
    assert dx > 0
    assert dy > 0

    # map_fft
    # |1|2|
    # |3|4|
    map_fft = np.fft.fft2(inMap)
    mfNx, mfNy = map_fft.shape

    mfSampFreq_x = np.fft.fftfreq(mfNx, 1)
    mfSampFreq_y = np.fft.fftfreq(mfNy, 1)

    mfNxx = np.argmax(mfSampFreq_x) + 1
    mfNyy = np.argmax(mfSampFreq_y) + 1

    # empty matrix
    zeroPaddedMap_fft = np.zeros([mfNx+dx, mfNy+dy], dtype=complex)
    # filling it
    zeroPaddedMap_fft[:mfNxx, :mfNyy] = map_fft[:mfNxx, :mfNyy]
    zeroPaddedMap_fft[:mfNxx, mfNyy+dy:] = map_fft[:mfNxx, mfNyy:]
    zeroPaddedMap_fft[mfNxx+dx:, :mfNyy] = map_fft[mfNxx:, :mfNyy]
    zeroPaddedMap_fft[mfNxx+dx:, mfNyy+dy:] = map_fft[mfNxx:, mfNyy:]
    # symmetrizing map
    zeroPaddedMap_fft[mfNxx+dx, :] /= np.sqrt(2)
    zeroPaddedMap_fft[mfNxx, :] = zeroPaddedMap_fft[mfNxx+dx, :]
    zeroPaddedMap_fft[:, mfNyy+dy] /= np.sqrt(2)
    zeroPaddedMap_fft[:, mfNyy] = zeroPaddedMap_fft[:, mfNyy+dy]

    factor = float(inMap.size)/float(zeroPaddedMap_fft.size)

    zeroPaddedMap = np.real(np.fft.ifft2(zeroPaddedMap_fft))/factor
    
    return zeroPaddedMap
        
###############################################################################

class StackMap(object):
    #initializations###########################################################
    def __init__(self, MapFile, WeightsFile, BeamFile, SourcesFile, 
                 RA0, RA1, DEC0, DEC1):
        
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
        
        #the same for sourcesfile
        sourcesmap = fits.getdata(SourcesFile, 0)
        
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
        self.sourcesmap = sourcesmap[y0:y1+1,x0:x1+1]
        
        
        #updating in the header the map dimensions and reference pixel

        self.maphdr = maphdr.copy()
        self.maphdr['NAXIS1'] = self.datamap.shape[1]
        self.maphdr['NAXIS2'] = self.datamap.shape[0]
        self.maphdr['CRPIX1'] -= x0
        self.maphdr['CRPIX2'] -= y0
        
        self.w= wcs.WCS(self.maphdr)
                                        
        #Beam map initializations
        
        #BeamMapFile is a file that contains the BeamMap unit normalized
        #radial profile B(r). unit normalized means that B(0) = 1.
        #the profile contains values from 0deg to near 1deg.
        
        self.beammap = beammap(BeamFile, self.maphdr)
                
        #fullmap initialization
        
        #this is the core map. first we create it as the original map with 
        #sources masked multiplying datamap with sourcesmap
        
        self.fullmap = np.copy(self.datamap)  #* self.sourcesmap
        
        
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

    def squeezefullmap(self):
        '''
        Multiplies the fullmap pixel-wise, by the number of observations in 
        any single pixel: sqrt(N_obs(x)/N_obs,max)
        '''
        self.N_max = np.amax(self.weightsmap)
        self.fullmap *= np.sqrt(self.weightsmap/self.N_max)

        
    #filter methods############################################################
    def filterfullmap(self):
        self.filt = matchedfilter(self.fullmap,self.beammap)
        self.fullmap = filtermap(self.fullmap,self.filt)
        
    ###########################################################################
    def unsqueezefullmap(self):
        '''
        Reverse the pixel-wise multiplication
        '''
        self.fullmap *= np.sqrt(self.N_max/self.weightsmap)    


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
    
    def setsubmap(self,RA,DEC):
        '''
        sets the current submap centered at RA,DEC
        
        The center is at L/2,L/2.
        
        if the quadrants are 2|1
                             3|4
        the center is at the inferior left corner of 1
        
        L has to be setted before calling this method
        '''
        
        cx, cy = self.getpix(RA,DEC)
        
        L = self.submapL
        
        self.submap = self.fullmap[cy-L/2:cy+L/2,cx-L/2:cx+L/2]
        self.submapw = self.weightsmap[cy-L/2:cy+L/2,cx-L/2:cx+L/2]
        
    def zpadsubmap(self, newLength):
        '''
        Decreases submap pixels size via zero padding
        For the weights it does a zero order interpolation
        
        newLength: new large in pixel
        '''
        self.newLength = newLength
        self.factor = self.newLength/self.submapL
        
        self.submapzpad = zeroPadMap(self.submap, newLength)

        self.submapzpadw = imresize(self.submapw, (newLength,newLength),
                                    interp="nearest")              
    
    def recenter(self):
        '''
        Recenters the zpadded submap
        '''
        #we add 1 big-old pixel at the left and down borders to simplify the
        #recentering
        N = (self.submapL + 1)*self.factor
        
        self.submapzpad2big = np.zeros((N,N))
        self.submapzpad2 = np.zeros(self.submapzpad.shape)
             
        #Here we make use of the second definition of pixel
        #we get the float, and then substract 0.5   
        cx, cy = self.w.wcs_world2pix(self.coord, 0)[0]
        cx -= 0.5
        cy -= 0.5
                
        #Then we get the position of the new center respect the previous
        #center, which was the inferior left corner of the big-old pixel
        
        f = self.factor
        
        posx = int(f*(cx - int(cx)))
        posy = int(f*(cy - int(cy)))
        
        self.submapzpad2big[f-posy:N-posy, f-posx:N-posx] = self.submapzpad
        self.submapzpad2 = self.submapzpad2big[f:N,f:N]
        
    def cvalsubmap(self):
        mid = self.submapL/2
        return self.submap[mid,mid]
        
    def cvalsubmapzpad(self):
        mid = self.newLength/2
        return self.submapzpad[mid,mid]
    
    def cvalsubmapzpad2(self):
        mid = self.submapzpad2.shape[0]/2
        return self.submapzpad2[mid,mid]  
    ###########################################################################
    
    #stacking methods##########################################################
    def setstackmap(self):
        '''
        sets the stackmap: the map in which we will stack submaps
        
        L has to be setted before calling this method
        '''
        
        L = self.submapL

        self.stackmap = np.zeros((L,L))
        self.stackmapw = np.zeros((L,L))                
    
    def stacksubmap(self):
        '''
        stacks the current submap
        
        A submap has to be setted before calling this method
        '''
        self.stackmap += self.submap*self.submapw
        self.stackmapw += self.submapw
        
    def finishstack(self):
        '''
        Finishes the stacking process dividing the stackmap by the stack of
        weights
        
        Needs at least one call on stacksubmap
        '''
        self.stackmap = self.stackmap/self.stackmapw

    ###########################################################################
    
    #plot and saving methods###################################################
    def plotfullmap(self):
        return plt.imshow(self.fullmap, origin = 'lower',
                          interpolation = 'none')
         
    def plotsubmap(self):
        return plt.imshow(self.submap, origin = 'lower',
                          interpolation = 'none')
    
    def plotsubmapw(self):
        return plt.imshow(self.submapw, origin = 'lower',
                          interpolation = 'none')
        
    def savefullmapfits(self,filename):
        return fits.writeto(filename, self.fullmap, self.maphdr)
    
    def savesubmapfits(self,filename):
        return fits.writeto(filename, self.submap)
        
    def plotsubmapzpad(self):
        return plt.imshow(self.submapzpad, origin = 'lower',
                          interpolation = 'none')
    
    def plotsubmapzpad2(self):
        return plt.imshow(self.submapzpad2, origin = 'lower',
                          interpolation = 'none')

    def plot(self, Type, circle = False):
        
        fig,ax = plt.subplots(1)
        
        if Type == "fullmap":
            img = self.fullmap
        if Type == "submap":
            img = self.submap
        if Type == "submapzpad":
            img = self.submapzpad
        if Type == "submapzpad2":
            img = self.submapzpad2
        if Type == "stackmap":
            img == self.stackmap
               
        ax.imshow(img, interpolation = "none", origin = "lower")
        
        if circle:
            circ = Circle((img.shape[1]/2,img.shape[0]/2), 0.25, color = "white")   
            ax.add_patch(circ) 
        
        return plt.show()
        
    def splot(self, Type, filename):
        
        fig,ax = plt.subplots(1)
        
        if Type == "fullmap":
            img = self.fullmap
        if Type == "submap":
            img = self.submap
        if Type == "submapzpad":
            img = self.submapzpad
        if Type == "submapzpad2":
            img = self.submapzpad2
               
        ax.imshow(img, interpolation = "none", origin = "lower")
        circ = Circle((img.shape[1]/2,img.shape[0]/2), 0.25, color = "white")   
        ax.add_patch(circ) 
        
        return plt.savefig(filename)
    ###########################################################################