"""
This is a script that incorporates metabolic theory
of ecology (MTE) and thermal performance curves (TPC)
or temperature response into reproduction and death
probabilities.  The theory is mostly taken from
Amarasekare and Savage (2011), Brown et al (2004),
and some ideas from Arroyo et al (2022).

Created: 2022? by Camille Febvre
Last modified: Feb 4, 2024 by Camille Febvre
"""

# import modules
import numpy as np
from scipy.stats import skewnorm
from scipy.special import lambertw
import TNM_constants as const # 

# TWO OPTIONS: 
#    1. Amarasekare and Savage equation
#    2. Scipy skewnorm
# skewnorm should match either amarasekare roff or poff

# skewnorm parameters
scaling = 5.5
Tref = 303 #308
width = 11
skew = -3

# scale MTE
Mscale = 1.95 # default 1

# Amarasekare defines thermal performance curves based on 
# life history traits
def roff_Amarasekare(T,Topt=298):
    """ Original Amarasekare and Savage equation (reproduced here for legacy reasons and for comparison if desired) """
    # Constants
    alphaTr = 60
    # Topt = 298
    dTr_ = 0.03
    dTr = 0.05
    Tr = 294
    bTr = 50
    Ad_ = 7500
    Ad = 6000 # 14000
    bTr_ = 295
    Aa = -8000
    s = 2.5

    # Functions
    def W(x):
        return lambertw(x)
    def TD(T):
        return 1/Tr - 1/T
    def roff(T): #fecundity development and mortality
        factor = 1/(alphaTr*np.exp(Aa*TD(T)))
        last_bit = dTr*np.exp(Ad*TD(T)) + dTr_*np.exp(Ad_*TD(T)) 
        expon = Aa*TD(T) - (T-Topt)**2/(2*s**2) + alphaTr*np.exp(Aa*TD(T))*last_bit
        parentheses = bTr_*alphaTr*np.exp(expon)
        return np.real(factor * W(parentheses))
    
    return roff(T) # just return roff


def poff_Amarasekare(T,Topt=298):
    # poff = 1-e^(-g)
    #return 1 - np.exp(-roff_Amarasekare(T))
    # poff = roff
    return roff_Amarasekare(T,Topt)

# define temperature-dependent equations
def poff_T(T,Tresponse_i=False):
    """Impact of temperature on probability of reproduction.
    Designed to match positive term in rm equation of 
    Amarasekare et al.

    IN: 
        T (K)
        Tresponse: False or list of 3 Tresponses of this species
    OUT: 
        poff(T)
    """
    # parameters set to so maximum is 1
    #Tref = 310.5 # shifts curve along T-axis
    #width = 11 # defines width of curve
    #skew = -3 # defines skewness
    #scaling = 6 # scaled to match Amarasekare 
    
    if type(Tresponse_i) != bool: # variable Tresponse by species
        Tref_i = Tresponse_i[0]
        width_i = width #Tresponse_i[1]
        skew_i = skew #Tresponse_i[2]
    else: # otherwise all species have same parameters 
        Tref_i,width_i,skew_i = Tref,width,skew
        #2.65*6 # scaled up so that maximum is one
    # NOTE that this means the integral = scaling, not 1!
    # 3/30/23: scale up by 1.0134 so peak is closer to 1 (at T = 307.8K)
    return 1.0134*3*scaling*skewnorm.pdf(T,skew_i,loc=Tref_i+5,scale=width_i) # shift Topt by 5 to get Tref 3/4/23


def poff_i(f):
    """TNM poff = sigmoidal function of fitness.

    IN: f- fitness, calculated from interaction matrix
    OUT: poff_i(f)
    """
    return 1/(1+np.exp(-f))

def poff_total(f,T,Tresponse_i=False,TRC=False):
    """Calculate probability of reproduction
    depending on fitness and temperature by
        p(f,T) = p(f)*p(T).
    
    IN: 
        f- fitness, calculated from interaction matrix
        T (K)
        Tresponse: False or list of 3 Tresponses of this species
    OUT: 
        poff(f,T) = poff(f)*poff(T)
    """
    #return poff_T(T,Tresponse_i)*poff_i(f)
    if type(Tresponse_i) != bool: # Topt, Twidth, skew
    # coudn't I just say, "if TRC:" here? ***
        Topt = Tresponse_i[0]
    else: Topt = 298
    if TRC == "MTE-env":
        # real math: 
        #return Mscale*MTE(T)*poff_T(T,Tresponse_i)*poff_i(f)
        # cop out:
        return Mscale*poff_Amarasekare(T,Topt)*poff_i(f) # shortcut for MTE-TPC combo is just to use Amarasekare equation
    else:
        return poff_T(T,Tresponse_i)*poff_i(f)

def pdeath(T,option='rate'):
    """Calculate probability of death depending on T.
    Designed to match negative term (d(T)) of rm equation 
    in Amarasekare et al.

    IN: T (K)
    OUT: pdeath
    """
    Tr = 294 # K # reference T (usually between 20 and 30 deg C)
    #if not T:
    #    return const.pkill
    dTr = 1.4*0.05*1.6 # adult moratlity rate at Tref, scaled so pdeath(T=307.8) = 0.2 3/30/23
    Ad = 6000 # 14000
    TD = 1/Tr - 1/T
    death_rate = dTr*np.exp(Ad*TD)
    if option == 'rate':
        return death_rate
    else: # option = prob
        return 1 - np.exp(-death_rate)

def pmut(T,varies=False):
    """Calculate probability of mutation depending on T.
    IN: T (K)
    OUT: pmut
    """
    if varies:
        Tr = 294 # K # reference T (usually between 20 and 30 deg C)
        #if not T:
        #    return const.pkill
        #mTr = 0.004 # mutation rate at Tref: scaled to be 0.01 at T=307.8K (T at which poff_T=1)
        #Am = 6000 # I assume this to be the same as for death-- but this is Ea=.516 eV
        mTr = 1.4*0.0035 # mutation rate at Tref: scaled to be 0.01 at T=307.8
        Ea = .6 # eV, from MTE theory
        k = 8.6e-5 # ev/K Boltzmann's constant
        Am = Ea/k
        TD = 1/Tr - 1/T
        return mTr*np.exp(Am*TD)
    else: return const.pmut

def MTE(T):
    B0 = .2*1.9e10 #*2.5 modified again 3/4/23, removing 2.5 multiplier #modified Jan 31, 2023 to match the 
    # scaled TRC # then mult by 2.5 on Feb 8 to scale up TRCs
    Ea = 0.6 # eV
    k = 8.6E-5 # ev/K Boltzmann's constant
    B = B0 * np.exp(-Ea/k/T)
    return B
