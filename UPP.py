# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

h = 6.626e-34 # J*s
k_B = 1.381e-23 #J/K

T_CMB = 2.725 #K
sigma_T = 6.6524e-25 #cm^2
m_e = 0.511e3 #keV/c^2

H_0 = 68. #km/s/Mpc
h_70 = H_0/70.

Omega_m = 0.31
Omega_Lambda = 0.69

[P_0,c_500,gamma,alpha,beta] = [8.403*h_70**(-3./2.), 1.177, 
                                0.3081, 1.0510, 5.4905]

def lp(x):
    '''
    lp: dimensionless universal profile
    
    Returns lp(x), x = r/R_500 
    '''
    return P_0/((c_500*x)*(1+(c_500*x)**alpha)**((beta-gamma)/alpha))

def plot_lp():
    x = np.linspace(0.03,1,100)
    y = lp(x)/P_0

    plt.plot(x,y)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def f_sz(freq):
    '''
    frequency component factor of sunyaev zel'dovich effect
    freq in GHz
    '''
    x = 10.**9*freq*h/(k_B*T_CMB)
    return x*(np.exp(x)+1)/(np.expm1(x))-4.

def plot_f_sz():
    x = np.linspace(1.,500.,1000)
    y = [f_sz(f) for f in x]  
    plt.plot(x,y)
    plt.show()


def f_sz_rel(freq,T_e):
    '''
    relativistic frequency component factor of Sunyaev Zel'dovich effect
    freq en GHz
    
    needs and electron temperature T_e in keV/k_B (a 5 value means K_B*T_e=5keV)
    
    Typical values ranges 1 to 5
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
 
def plot_f_sz_rel():
    x = np.linspace(100,300.,10000) 
    y = [f_sz(f) for f in x]  
    y_rel_5 = [f_sz_rel(f,5.) for f in x]
    y_rel_1 = [f_sz_rel(f,1.) for f in x]
    plt.plot(x,y_rel_5, color = 'Blue')
    plt.plot(x,y_rel_1, color = 'Red')
    plt.plot(x,y, color = 'Black')
    plt.show()

def h_(z):
    '''
    Hubble scaled redshift function
    h(z) = H(z)/H_0 = (Omega_m*(1+z)^3+Omega_Lambda)^1/2
    '''
    return np.sqrt(Omega_m*(1+z)**3+Omega_Lambda)

def P(r,z,M_500, R_500):
    '''
    Universal Pressure Profile
    
    M_500 in 10^14 M_sun. typical from 0.1 to 10
    R_500 in Mpc. typical from 0.1 to 1
    
    P(r) = 1.65x10^-3 h(z)^8/3 [M_500 / 3x10^14 h_70^-1]^(2/3+α_p+α'_p(x))
           x lp(x) h_70^2 keV cm^-3
    '''
    x = r/R_500
    alpha_p = 0.12
    def alphap_p(x):
        #not using this correction for now, because it's bugged
        #return (0.10-(alpha_p+0.10)*(  ((x/0.5)**3)/(1.+ ((x/0.5)**3) )  ))
        return 0
    
    P_500 = (1.65e-3)*(h_(z)**(8./3.))*((M_500/(3.e14*h_70**(-1)))**(2./3.))*h_70**2.
    P = P_500*((M_500/(3.e14*h_70**(-1)))**(alpha_p+alphap_p(x)))*lp(x)
    return P
    
  
def plot_P():  
    x = np.linspace(0.01,1,1000)
    y = [P(r,1.,1.,1.)[0]/P(r,1.,1.,1.)[1] for r in x]
    plt.plot(x,y)
    plt.xlim([0.01,1.])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def P_ss(r,z,M_500):
    '''
    Self similar universal pressure profile
    
    r in Mpc, typical values are 0.01 to 1
    M_500 in solar masses
    '''
    #first we pass all to mks
    G = 6.67408e-11 # m^3 kg^-1 s^-2
    
    #for the Hubble constant we divide km/Mpc
    #1 pc = 3.0857 × 10^16 m
    H_0_s = H_0*(1.e3/(1.e6*3.0857e16)) #s^-1
    def H(z):
        return H_0_s*np.sqrt(Omega_m*(1+z)**3+Omega_Lambda)
        
    f_B = 0.175
    mu = 0.59
    mu_e = 1.14
    
    #passing M_500 to kg
    M_500_kg = 1.989e30*M_500
    
    P_500 = (3./(8.*np.pi))*((500.*G**(-1./4)*H(z)**2)/2.)**(4./3.)*(mu/mu_e)*f_B*((M_500_kg)**(2./3.))
    #P_500 is in kg m^-1 s^-2 = kg m^2/s^2 m^-3 = J m^-3
    #we pass this to keV cm^-3 
    #keV = 1000*1.602176565e-19 J => J = (1/1.602176565e-16) keV
    #1m = 100 cm -> m^-3 = 10^-6 cm
    #=> J m^-3 = 1.e10/1.602176565
    P_500 *= 1.e10/1.602176565
    
    def rho_c(z):
        '''
        critical density in kg m^-3
        '''
        return (3*(H(z))**2)/(8*np.pi*G)
    
    R_500 = (3*M_500_kg/(4*np.pi*500*rho_c(z)))**(1./3.) #in m
    
    #1 m = 1/(3.0857e16) pc
    R_500 = R_500/3.0857e22 #in Mpc
    x = r/R_500
    P = P_500*lp(x)
    return P

def plot_bothP():
    x = np.linspace(0.01,1,1000)
    y = [P(r,0.1,1.e13,0.3232) for r in x]
    y_ss = [P_ss(r,0.1,1.e13) for r in x]
    plt.plot(x,y, color = 'Blue')
    plt.plot(x,y_ss, color = 'Red')
    #plt.xlim([0.01,1.])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()