# -*- coding: utf-8 -*-
from astropy.io import fits
from astropy.wcs import wcs
import numpy as np
from matplotlib import pyplot as plt #needs wcsaxes package for plots
from scipy import interpolate
from scipy import ndimage
from scipy.misc import imresize
from matplotlib.patches import Circle
from astropy.cosmology import FlatLambdaCDM
from scipy import integrate
from types import *

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

#conversion from M200 to M500
M200toM500 = 0.765

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

def Tau_minimap(R_500, z, pixsize):
    '''
    Minimap of the unit normalized model up to 5*theta_500
    '''
    c_500 = 1.177
    alpha = 1.0510
    beta = 5.4905
    gamma = 0.3081
    r_s = R_500/c_500
    
    def D_A(z):
        return cosmo.angular_diameter_distance(z).value 
    
    def P_3D(r):
        x = r/r_s
        return 1./((x**gamma)*(1+x**alpha)**((beta-gamma)/alpha))
    
    r = np.ceil((5*R_500/D_A(z))*(360/(2*np.pi))/pixsize)
    
    DA = D_A(z)
    l_max = 5*R_500
    
    P_0 = integrate.quad(P_3D,0.03,l_max)[0]

    projection = lambda s,theta : P_3D(np.sqrt(s**2+theta**2*DA**2))

    def P_2D(x,y):
        d = np.sqrt(x**2+y**2)
        return integrate.quad(projection,0.03,l_max,args = (d,))[0]/P_0
    
    f = np.vectorize(P_2D)

    xaxis = np.linspace(-r, r, 2*r+1)*pixsize*(2*np.pi)/360.
    yaxis = np.linspace(-r, r, 2*r+1)*pixsize*(2*np.pi)/360.
    result = f(xaxis[:,None], yaxis[None,:])    
    
    return result

def Tau_map(R_500,z,mapheader):
    pixsize = abs(mapheader['CDELT1'])
    
    minitau = Tau_minimap(R_500,z,pixsize)
    radius = minitau.shape[0]/2
    
    shape_x = mapheader['NAXIS1']
    shape_y = mapheader['NAXIS2']

    taumap = np.zeros((shape_y,shape_x))
    
    cx = int(shape_x/2)
    cy = int(shape_y/2)

    taumap[cy-radius:cy+radius+1,cx-radius:cx+radius+1] = minitau
    return taumap
    
def normalized_convolution(s1,s2):
    '''
    Normalized convolution of two signals
    '''
    conv = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(s1)*np.fft.fft2(s2))))
    ymid = conv.shape[0]/2
    xmid = conv.shape[1]/2
    norm = conv[ymid,xmid]
    conv /= norm
    return conv
    
def matchedfilter_one(inmap,signal):
    mfft = np.abs(np.fft.fft2(inmap))
    mfft2_inv = 1./mfft**2
        
    median = np.median(np.copy(mfft2_inv))
        
    mfft2_inv[mfft2_inv>median/3] = median
    mfft2_inv[mfft2_inv<median*3] = median
            
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
    
    signal_fft = np.fft.fft2(signal)
    signal_fft_c = np.conjugate(signal_fft)
    
    mfilt = (signal_fft_c*mfft2_inv)
    mfilt_norm = np.sum(np.abs(signal_fft)**2*mfft2_inv)/mfilt.size
    
    mfilt /= mfilt_norm
    return mfilt  


def filtermap(data,filt):
    map_fft = np.fft.fft2(data)
    filtmap_fft = map_fft * filt
    filtmap = np.fft.ifft2(filtmap_fft)    
    return np.real(np.fft.fftshift(filtmap))


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
    
def fakecatalog(num, RA0, RA1, DEC0, DEC1, A):
    '''
    Generates a fake catalog of num sources at random [RA,DEC]
    confined on a region of RA1, RA0, DEC1, DEC0
    
    A degrees are cropped to leave some space for submaps
    '''
    assert type(num) is IntType
    A = np.float(A)
    RA0 = np.float(RA0)
    RA1 = np.float(RA1)
    DEC0 = np.float(DEC0)
    DEC1 = np.float(DEC1)
    
    fakecat = []
    for i in range(num):
        RA = np.random.uniform(RA1+A,360.+RA0-A)
        if RA>=360:
            RA -= 360
        DEC = np.random.uniform(DEC0+A,DEC1-A)
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

def f_sz(freq):
    '''
    frequency component factor of sunyaev zel'dovich effect
    freq in GHz
    '''
    x = 10.**9*freq*h/(k_B*T_CMB)
    return x*(np.exp(x)+1)/(np.expm1(x))-4.

def f_sz_rel(freq,T_e):
    '''
    relativistic frequency component factor of Sunyaev Zel'dovich effect
    freq en GHz
    
    needs and electron temperature T_e in keV/k_B (a 5 value means K_B*T_e=5keV)
    
    Typical values ranges 1 to 15
    '''
    theta_e = float(T_e)/0.511e3 #K_B*T_e in keV, since m_e*c^2 = 0.511 MeV
    x = 10.**9*float(freq)*h/(k_B*T_CMB) 
    X = x/np.tanh(x/2)
    S = x/np.sinh(x/2)
    Y_0 = -4.0+X
    Y_1 = -10.0+(47.0/2.0)*X-(42.0/5.0)*X**2+(7.0/10.0)*X**3+(S**2)*(-21.0/5.0
          +(7.0/5.0)*X)
    Y_2 = (-15.0/2.0+(1023.0/8.0)*X-(868.0/5.0)*X**2+(329.0/5.0)*X**3
          -(44.0/5.0)*X**4+(11.0/30.0)*X**5+(S**2)*(-434.0/5.0+(658.0/5.0)*X
          -(242.0/5.0)*X**2+(143.0/30.0)*X**3)+(S**4)*(-44.0/5.0+(187.0/60.0)*X))
    return (Y_0+Y_1*theta_e+Y_2*theta_e**2)

def R_500_from_M_500(M_500,z):
    M_500_kg = 1.989e30*M_500
    
    #first we pass all to mks
    G = 6.67408e-11 # m^3 kg^-1 s^-2
    
    #for the Hubble constant we divide km/Mpc
    #1 pc = 3.0857 × 10^16 m
    H_0_s = H_0*(1.e3/(1.e6*3.0857e16)) #s^-1
    
    def H(z):
        return H_0_s*np.sqrt(Omega_m*(1+z)**3+Omega_Lambda)
    
    def rho_c(z):
        '''
        critical density in kg m^-3
        '''
        return (3*(H(z))**2)/(8*np.pi*G)
        
    R_500 = (3*M_500_kg/(4*np.pi*500*rho_c(z)))**(1./3.) #in m
    return R_500/3.0857e22 #in Mpc

def T_e_from_M_500(M_500, z):
    '''
    Temperature of a singular isothermal sphere with mass M_500 at redshift z
    
    M_500 in M_sun
    '''
    R_500 = R_500_from_M_500(M_500,z) #in Mpc
    m_p = 0.938 #Gev / c^2 #proton mass
    mu = 0.59 #mean molecular weight per free electron. From Nagai et al 2007
    G = 4.302e-3 #pc M_sun^-1 (km/s)^2 #gravitational constant in useful units
    c = 299792 #km/s
    
    return mu*m_p*G*M_500/(2*R_500*c**2) #in keV      
  
def M_200_from_N_gals(Ngals):
    '''
    From Hao, in solar masses
    '''
    return 8.8e13*((0.66*Ngals+9.567)/20.)**1.28
    
def Y_sz_theoretical(M_500,z):
    alpha = 1.78
    I5 = 1.1037
    B_x = I5*2.925e-5
    
    EZ = np.sqrt(Omega_m*(1+z)**3+Omega_Lambda)
    
    DAZ = cosmo.angular_diameter_distance(z).value
    
    return B_x*((M_500/(3.e14*h_70))**alpha)#*(EZ**(3./2.)*DAZ**2)
    
    
def DeltaT_0_theoretical(M_500,z):
    tau = Tau_minimap(R_500_from_M_500(M_500,z),z,0.00825)
    integ = np.sum(tau)*(0.00825*2*np.pi/360.)**2
    Te = T_e_from_M_500(M_500, z)
    Ysz = Y_sz_theoretical(M_500,z)
    fsz = f_sz_rel(148,Te)
    return (Ysz*fsz*T_CMB/integ)*1.e6
    
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
        
        self.datamap *= self.sourcesmap
        
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
        
        #self.beammap = beammap(BeamFile, self.maphdr)
        
        self.beamfile = BeamFile        
                                
    ###########################################################################
    '''
    def setfullmap(self):
        #fullmap initialization
        
        #this is the core map. first we create it as the original map with 
        #sources masked multiplying datamap with sourcesmap
        self.fullmap = np.copy(self.datamap)  #* self.sourcesmap
    '''
    def setfullmap(self):
        self.fullmapheader = self.maphdr.copy()
        self.fullmapheader["NAXIS2"] = 256
        self.fullmapheader["NAXIS1"] = 8192
        #CRPIX is changed later
    
    def getbeammap(self):
        self.beammap = beammap(self.beamfile, self.fullmapheader)
    
    def gettaumap(self, M_500, z):
        self.M_500 = M_500
        self.z = z
        R_500 = R_500_from_M_500(M_500,z)
        self.R_500 = R_500
        self.taumap = Tau_map(R_500,z,self.fullmapheader)
            
    def getnormconv(self):
        self.nc = normalized_convolution(self.taumap,self.beammap)
        
    def fakeclus(self,RA,DEC,R_500,z,radius):
        '''
        Adds a fake cluster in fullmap
        radius in pixels
        '''
        taumap = Tau_map(R_500,z,self.fullmapheader)
        beammap = self.beammap
        conv = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(taumap)*np.fft.fft2(beammap))))
        ymid = conv.shape[0]/2
        xmid = conv.shape[1]/2
        norm = conv[ymid,xmid]
        conv /= norm
        
        conv *= -200
        
        v = np.array([[RA,DEC]])
        rp = self.w.wcs_world2pix(v,0)
        p = np.array(np.round(rp,0), dtype = np.int)
        px = p[0][0]
        py = p[0][1]
        r = radius
        self.datamap[py-r:py+r+1,px-r:px+r+1] += conv[ymid-r:ymid+r+1,xmid-r:xmid+r+1]     
        
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
    
    def getpix_fullmap(self,RA,DEC):
        coord = np.array([[RA,DEC]], dtype = np.float_)
        px, py = np.array(np.round(self.fullmapw.wcs_world2pix(coord, 0),0),
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
        (using origin = "lower")
        
        L has to be setted before calling this method
        '''
        self.RA,self.DEC = RA,DEC
        
        cx, cy = self.getpix(RA,DEC)
        
        L = self.submapL
        
        Lx = self.maphdr["NAXIS1"]
        Ly = self.maphdr["NAXIS2"]
        
        #3 4
        #1 2
        
        #1
        if cx<=Lx/2 and cy<=Ly/2:
            x0 = 0
            x1 = 8192
            y0 = 0
            y1 = 256    
            
        #2
        if cx>Lx/2 and cy<=Ly/2:
            x0 = Lx-8192
            x1 = Lx
            y0 = 0
            y1 = 256
            
        #3
        if cx<=Lx/2 and cy>Ly/2:
            x0 = 0
            x1 = 8192
            y0 = Ly-256
            y1 = Ly
            
        #4
        if cx>Lx/2 and cy>Ly/2:
            x0 = Lx-8192
            x1 = Lx
            y0 = Ly-256
            y1 = Ly
        
        self.fullmap = np.copy(self.datamap)[y0:y1,x0:x1]
        self.fullmapweights = self.weightsmap[y0:y1,x0:x1]
        self.fullmapheader["CRPIX1"] -= x0
        self.fullmapheader["CRPIX2"] -= y0
        
        self.fullmapw= wcs.WCS(self.fullmapheader)
        
        cx,cy = self.getpix_fullmap(RA,DEC)
        
        self.submap = self.fullmap[cy-L/2:cy+L/2,cx-L/2:cx+L/2]
        self.submapw = self.weightsmap[cy-L/2:cy+L/2,cx-L/2:cx+L/2]
    
    def getsubmap(self):
        
        L = self.submapL
        
        cx,cy = self.getpix_fullmap(self.RA,self.DEC)
        
        self.submap = self.fullmap[cy-L/2:cy+L/2,cx-L/2:cx+L/2]
        self.submapw = self.fullmapweights[cy-L/2:cy+L/2,cx-L/2:cx+L/2]
    
    def getsubmapSZ(self):
        I = (np.sum(self.taumap))*(0.00825*2*np.pi/360.)**2
        Te = T_e_from_M_500(self.M_500, self.z)
        fsz = f_sz_rel(148,Te)
        EZ = np.sqrt(Omega_m*(1+self.z)**3+Omega_Lambda)
        DAZ = cosmo.angular_diameter_distance(self.z).value
        #needs a factor of 1.e-6 because map temperature is in
        #microKelvins and T_cmb is in Kelvins
        T_to_SZ = I*(DAZ**2)*(EZ**(-2./3.))/(fsz*T_CMB)*1.e-6
        self.submap *= T_to_SZ
        self.sigma = np.std(self.fullmap)*np.abs(T_to_SZ)
    
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
    
    def squeezefullmap(self):
        '''
        Multiplies the fullmap pixel-wise, by the number of observations in 
        any single pixel: sqrt(N_obs(x)/N_obs,max)
        '''
        self.N_max = np.amax(self.fullmapweights)
        self.fullmap *= np.sqrt(self.fullmapweights/self.N_max)


    #filter methods############################################################
    def filterfullmap_clus(self,R_500,z):
        self.normconv(R_500,z)
        self.filt = matchedfilter_one(self.fullmap,self.nc)
        self.fullmap = filtermap(self.fullmap,self.filt)
        
        #apodizing 10 border pixels
        Lx = self.fullmap.shape[1]
        Ly = self.fullmap.shape[0]
        self.fullmap = self.fullmap[10:Ly-10,10:Lx-10]
        self.fullmapweights = self.fullmapweights[10:Ly-10,10:Lx-10]
        self.fullmapheader["CRPIX1"] -= 10
        self.fullmapheader["CRPIX2"] -= 10
        self.fullmapw= wcs.WCS(self.fullmapheader)
        
    
    def filterfullmap(self):
        self.filt = matchedfilter_one(self.fullmap,self.nc)
        self.fullmap = filtermap(self.fullmap,self.filt)
        
        #apodizing 10 border pixels
        Lx = self.fullmap.shape[1]
        Ly = self.fullmap.shape[0]
        self.fullmap = self.fullmap[10:Ly-10,10:Lx-10]
        self.fullmapweights = self.fullmapweights[10:Ly-10,10:Lx-10]
        self.fullmapheader["CRPIX1"] -= 10
        self.fullmapheader["CRPIX2"] -= 10
        self.fullmapw= wcs.WCS(self.fullmapheader)        
        
        
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
        #self.stackmapw = np.zeros((L,L))
        self.nstack = 0
        self.sigmas = []
    
    def stacksubmap(self):
        '''
        stacks the current submap
        
        A submap has to be setted before calling this method
        '''
        self.stackmap += self.submap
        self.nstack += 1
        self.sigmas.append(self.sigma)
        L = self.stackmap.shape[0]
        self.SN = np.sqrt(self.nstack)*self.stackmap[L/2,L/2]/(self.nstack*np.mean(self.sigmas))
        #self.stackmapw += self.submapw
        
    def finishstack(self):
        '''
        Finishes the stacking process dividing the stackmap by the stack of
        weights
        
        Needs at least one call on stacksubmap
        '''
        self.stackmap = self.stackmap/self.nstack
        self.sigmamean = np.mean(self.sigmas)/np.sqrt(self.nstack)

    ###########################################################################
    
    #plot and saving methods###################################################
    def savefullmapfits(self,filename):
        return fits.writeto(filename, self.fullmap, self.maphdr)
    
    def savesubmapfits(self,filename):
        return fits.writeto(filename, self.submap)
        
    def plot(self, Type, circle = False):
        
        fig,ax = plt.subplots(1)
        
        if Type == "fullmap":
            img = self.fullmap
        if Type == "submap":
            img = self.submap
        if Type == "submapw":
            img = self.submapw
        if Type == "submapzpad":
            img = self.submapzpad
        if Type == "submapzpad2":
            img = self.submapzpad2
        if Type == "stackmap":
            img = self.stackmap
               
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