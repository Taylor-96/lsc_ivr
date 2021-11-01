import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import lft_sampling_common
import sys

nsteps = 10000
n_mom_per_q = 10
#CM_INV_TO_AU = 7.2516e-7
CM_INV_TO_AU = 1./219471.52 # cm-1 to hartree
BOLTZMANN_AU = 3.1668114E-06      #Hartree/K
KCAL_MOL_ANG_CUB_TO_AU = 0.00023614758

# initialize (thermalize) the system (cold first). The system is an array of N_tau position values corresponding to N_tau values of time.
T=300 # temp in kelvin
k = 1.0
kT=BOLTZMANN_AU*T
beta=1/kT
m = 1836.15 # mass of particle in a.u
omega = 1500.0 # i.e. dtau which is hbar*beta but hbar=1 in these reduced units
omega *= CM_INV_TO_AU
dt = 1.0
hbar=1

eig_thresh=1e-9

if (len(sys.argv) < 4):
    print("Too few arguments! Usage: python3 main_common.py potential momentum_approx mc_step_size")
    exit()


if(sys.argv[1] == "aho"):
    Lambda = 21.0 #21kcal mol^-1 Angstrom^-3. kcal/mol -> hartree = 1/627.5094736
    Lambda *= KCAL_MOL_ANG_CUB_TO_AU
    outfile = open("samples_QMAHO.out", "w")
    tcf_outfile = open("tcf_QMAHO.out", "w")

elif(sys.argv[1] == "ho"):
    Lambda = 0.0
    outfile = open("samples_QMHO.out", "w")
    tcf_outfile = open("tcf_QMHO.out", "w")

else:
    print("need to supply a potential")


# Angstrom^-3 -> bohr^-3 = (1/0.52917721)^3
                
V = lambda m,omega,Lambda,x: 0.5*m*omega**2*x**2 + (Lambda/3.0)*x**3
F = lambda m,omega,Lambda,x: -1.0*m*omega**2*x - Lambda*x**2




def LHA(m,omega,q):

    # in the case of QHO, the local frequency Omega is just omega
    Omega = omega
    Omega = np.sqrt(omega**2 + (2.0*Lambda/m)*q)
    Phi = (beta*hbar*Omega/2.0)/np.tanh(beta*hbar*Omega/2.0)
    sigma=np.sqrt(m*Phi/beta) # std dev of Gaussian momentum distribution.
    p = np.random.normal(loc=0.0, scale=sigma)
    return(p)


def LGA(m,omega,Lambda,q):

    Omega_sq = omega**2 + (2.0*Lambda/m)*q
    if(abs(Omega_sq) < eig_thresh):
        Phi=1.0
    elif Omega_sq >= 0:
        Omega=np.sqrt(Omega_sq)
        Phi = (beta*hbar*Omega/2.0)/np.tanh(beta*hbar*Omega/2.0)
    else:
        Omega=np.sqrt(-1.0*Omega_sq)
        Phi = np.tanh(beta*hbar*Omega/2.0)/((beta*hbar*Omega)/2.0)

    sigma=np.sqrt(m*Phi/beta) # std dev of Gaussian momentum distribution.
    p = np.random.normal(loc=0.0, scale=sigma)
    return p


h=float(sys.argv[3])

q_samples = lft_sampling_common.samplePos(Lambda,h) # call to MCMC marginal position distribution sampler
nsamples = len(q_samples)
p_samples = []

for q in q_samples:

    # sample position using PIM(C/D)
    # call sampling routine for momentum
    if sys.argv[2] == "lha":
        p = LHA(m,omega,q)
        p_samples.append(p)
        
    elif sys.argv[2] == "lga":
        for j in range(n_mom_per_q):
            p = LGA(m,omega,Lambda,q)
            p_samples.append(p)
            outfile.write("{} {}\n".format(q,p))


# propagate resulting initial (q,p) value forward in time classically.
    
pp_tcf = np.zeros(nsteps)

for q,p in list(zip(q_samples,p_samples)):
    p_init = p
    # velocity verlet propagation of B operator
    for step in range(nsteps):
        pp_tcf[step] += (p_init*p)

        F_old = F(m,omega,Lambda,q)
        q = q + (p/m)*dt + (0.5/m)*(dt**2)*F_old
        F_new = F(m,omega,Lambda,q)
        p = p + (0.5*dt)*(F_old + F_new)

for step in range(nsteps):
#    print(pp_tcf[step])

    tcf_outfile.write("{}\t{}\n".format(step*dt, pp_tcf[step]/nsamples))

   
   


outfile.close()
tcf_outfile.close()




#plt.subplot(221)
#counts,xbins,ybins,image = plt.hist2d(q_samples,p_samples,bins=100
#                                      ,norm=LogNorm()
#                                      , cmap = plt.cm.rainbow)
#plt.colorbar()
#plt.subplot(222)
#plt.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
#    linewidths=3, cmap = plt.cm.rainbow)
#
#plt.subplot(223)
#
## plot of analytic Wigner distribution of HO
#Theta = (hbar*omega/2.0)/(np.tanh(beta*hbar*omega/2.0))
#sigma_q=np.sqrt(Theta/(m*omega**2))
#sigma_p=np.sqrt(Theta*m)
#
#counts,xbins,ybins,image = plt.hist2d(np.random.normal(0,sigma_q,nsamples),np.random.normal(0,sigma_p,nsamples),bins=100 ,norm=LogNorm(), cmap = plt.cm.rainbow)
#plt.colorbar()
#plt.subplot(224)
#plt.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
 #   linewidths=3, cmap = plt.cm.rainbow)
#plt.show()
