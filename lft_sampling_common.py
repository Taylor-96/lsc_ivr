import numpy as np
import random as rand
import math
import matplotlib
import matplotlib.pyplot as plt
import sys 
#matplotlib.rcParams['text.usetex'] = True
np.set_printoptions(threshold=sys.maxsize)
"""
Steps:

propose a number u which is distributed according to a uniform distro between -h and h. The value of h is adjusted to meet a predefined acceptance ratio.


Working with hbar=c=m_e=1

"""


#CM_INV_TO_AU = 7.2516e-7
CM_INV_TO_AU = 1./219471.52 # cm-1 to hartree
BOLTZMANN_AU = 3.1668114E-06      #Hartree/K
KCAL_MOL_ANG_CUB_TO_AU = 0.00023614758

# initialize (thermalize) the system (cold first). The system is an array of N_tau position values corresponding to N_tau values of time.
T=300 # temp in kelvin
k = 1.0
kT=BOLTZMANN_AU*T
beta=1/kT
N_tau =100  # number of timeslices
d_tau = beta / N_tau
N_sep = 10 # Every 1000th sweep is used to calculate observable quantities


#Lambda = 21.0 #21kcal mol^-1 Angstrom^-3. kcal/mol -> hartree = 1/627.5094736
#Lambda *= KCAL_MOL_ANG_CUB_TO_AU
# Angstrom^-3 -> bohr^-3 = (1/0.52917721)^3
                
m = 1836.15 # mass of particle in a.u
omega = 1500.0 # i.e. dtau which is hbar*beta but hbar=1 in these reduced units
omega *= CM_INV_TO_AU

# convert to reduced units


#h = 0.1 # this parameter is adjusted to meet the predefined acceptance ratio idrate
num_sweeps = 50000          # 100000 sweeps
idrate = 0.8 # this is the ideal acceptance rate. It is used to adjust h
accrate = 0.0



def init(mode, N_tau, h):
    # either hot, cold or approx equilibrated
    if mode=="cold":
        return np.zeros(N_tau)
    elif mode=="hot":
        return np.random.uniform(-0.5, 0.5, N_tau) # init lattice sites to random numbers between -1/2 and 1/2

        
def MCMC_init(N_tau,x,h,idrate,m,omega,Lambda):
    """
    Performs one sweep through the lattice
    """
    global accrate
    accrate=0.0
    for i,x_i in enumerate(x):
        
        x_new = x_i + np.random.uniform(-h,h)     # we now propose an update to position x_i. x_i -> x_i + u
        if accept(x_new,x,i,Lambda):

            x[i] = x_new
            # we stop modifying the acceptance rate after equilibration 
            accrate += 1.0/N_tau
    #h = h*(accrate/idrate)
    return x,h
    
def MCMC(N_tau,x,h,idrate,m,omega,Lambda):
    """
    Performs one sweep through the lattice
    """
    #accrate = 0.0 # this is the acceptance rate.
    global accrate
    # for i,x_i in enumerate(x):
        
    #     x_new = x_i + np.random.uniform(-h,h)     # we now propose an update to position x_i. x_i -> x_i + u
    #     if accept(x_new,x,i,Lambda):
    #         x[i] = x_new
    #         # we stop modifying the acceptance rate after equilibration 
    #         accrate += 1.0

    # randomize site visiting order

    index=getSiteOrder(N_tau)

    for i in index:
        x_new = x[i] + np.random.uniform(-h,h)     # we now propose an update to position x_i. x_i -> x_i + u
        if accept(x_new,x,i,Lambda):

            x[i] = x_new
            accrate += 1.0
        else:
            pass
    
    return x   

def getSiteOrder(N_tau):
    index = []
    for i in range(N_tau):
        rand = np.random.randint(N_tau)

        index.append(np.random.randint(N_tau))
    return index


def accept(x_new, x, i,Lambda):
    
    tau_plus = (i+1)%N_tau
    tau_minus = (i-1)%N_tau
    x_plus = x[tau_plus]
    x_minus = x[tau_minus]
    S_old = 0.5*(m/d_tau)*(x_plus - x[i])**2 + 0.5*(m/d_tau)*(x[i]-x_minus)**2 + 0.5*d_tau*m*(omega**2)*(x[i]**2) + d_tau*(Lambda/3.0)*(x[i]**3)     # we need only consider positions x_(i-1), x_i, x_i+1
    S_new = 0.5*(m/d_tau)*(x_plus - x_new)**2 + 0.5*(m/d_tau)*(x_new-x_minus)**2 + 0.5*d_tau*m*(omega**2)*(x_new**2) + d_tau*(Lambda/3.0)*(x_new**3)  # as all others cancel in the difference delta_S
    delta_S = S_new - S_old
    #if delta_S < 0.0:
     #   return True

    try:
        p_accept =  math.exp(-delta_S)
    except OverflowError as err:
        print('Overflowed after ', delta_S, err)

    # accept update with probability p_accept. If the update lowers the action, then exp(-delta_S)>1, and the update is definitely made.
    # if the update doesn't lower the action, acceptance occurs with probability exp(-delta_S) as ensured by comparison with a unif(0,1) random
    if(p_accept > np.random.uniform(	)):
        return True
    else:
        return False



    
# only 1 sweep in 100 kept as a "configuration" and used for calculating observable averages.

# store output (the x array) in a file for plotting and further analysis.
outfile = open("mcmc_ho_outfile.txt","w")



def samplePos(Lambda,h):

    x = init("hot", N_tau, h)
    equi_len = 50
    samples = []
    num_sweeps_init = 1000

#    for sweep in range(num_sweeps_init):
#        print(sweep)
#        #print(accrate)
#        x,h = MCMC_init(N_tau,x,h,idrate,m,omega,Lambda)
#        if(sweep % N_sep == 0):
#
#            print("accrate = {}\n".format(accrate))
    
    for sweep in range(num_sweeps):
        print(sweep)
        #print(accrate)
        x = MCMC(N_tau,x,h,idrate,m,omega,Lambda)
        outfile.write("{} {}\n".format(sweep,np.sum(x)/N_tau))
        if(sweep % N_sep == 0):
            samples.append(x[0])
            #samples.extend(x)
            print("accrate = {}\n".format(accrate/(N_tau*(sweep+1))))

    return samples[equi_len:]


    

    

# plt.subplot(121)
# counts,xbins,ybins,image = plt.hist2d(q_samples,p_samples,bins=100
#                                       ,norm=LogNorm()
#                                       , cmap = plt.cm.rainbow)


# plt.colorbar()
# plt.subplot(122)
# plt.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
#     linewidths=3, cmap = plt.cm.rainbow)
# plt.show()


# For each position sample, compute local frequency, get QCF and then sample Gaussian momentum distro



#plt.hist(samples[200:], bins='auto')
#plt.show()
        

#outfile.close()
