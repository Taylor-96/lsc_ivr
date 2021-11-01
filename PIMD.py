import numpy as np

n_beads = 32
q_init = np.zeros(n_beads) # single particle in 1 dimension described by n_beads
nsteps = 10000 # number of steps for classical dynamics
n_steps_pimd = 200
n_steps_pimd_burnin = 500

#CM_INV_TO_AU = 7.2516e-7
CM_INV_TO_AU = 1./219471.52 # cm-1 to hartree
BOLTZMANN_AU = 3.1668114E-06      #Hartree/K
KCAL_MOL_ANG_CUB_TO_AU = 0.00023614758

fs=2.4188843e-2
AU_TO_THZ=1000/fs

# initialize (thermalize) the system (cold first). The system is an array of N_tau position values corresponding to N_tau values of time.
T=300 # temp in kelvin
k = 1.0
kT=BOLTZMANN_AU*T
beta=1/kT
m = 1836.15 # mass of particle in a.u
omega = 1500.0 # i.e. dtau which is hbar*beta but hbar=1 in these reduced units
omega *= CM_INV_TO_AU
Lambda = 21.0 #21kcal mol^-1 Angstrom^-3. kcal/mol -> hartree = 1/627.5094736
Lambda *= KCAL_MOL_ANG_CUB_TO_AU


dt = 0.25*(1./fs) # timestep for classical dynamics to compute tcf. approx 0.25 fs
d_tau = 0.25*(1./fs) # timestep for PIMD
hbar=1.0


mp= m # scaled mass for momenta
n_samples = 200
n_p_per_q = 10

samples_outfile = open("pimd_wigner_samples.out", "w")
tcf_outfile = open("pimd_tcf.out", "w")

eig_thresh=1e-9


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
    

def update_q(q,O,OT,m,mp,n_beads,beta,n_steps_pimd,d_tau):



    # normal mode momentum distributed as p with std dev scaled by n_beads due to NM transformation
    #p_nm =1/sqrt(n_beads) O_kl p_l
    # initialize PIMD momementum. fictitious momenta are only introduced to recover the prefactor of the density matrix => param m' (mp) doesn't effect estimators so choose to be whatever
    p_nm = np.random.normal(0,np.sqrt(mp*n_beads/beta),size=n_beads) 

    F = getPIforce(q,m,omega,Lambda) 
    F_nm = getNMrep(O,F)

    q_nm = getNMrep(O,q)

#    print(q_nm)

    # BAOAB NM integration as outlined in https://arxiv.org/pdf/1611.06331.pdf
    for step in range(n_steps_pimd):
    
        p_nm = Apply_B(p_nm,F_nm,d_tau/2.0)
        q_nm,p_nm = Apply_A(q_nm,p_nm,d_tau/2.0,m,n_beads)
        p_nm = Apply_O(p_nm,NM_lambda,beta,d_tau,m,n_beads)
        q_nm,p_nm = Apply_A(q_nm,p_nm,d_tau/2.0,m,n_beads)
        q = getSTDrep(OT,q_nm)
        F = getPIforce(q,m,omega,Lambda) 
        F_nm = getNMrep(O,F)
        p_nm = Apply_B(p_nm,F_nm,d_tau/2.0)


    return q


def Apply_B(p_nm,F_nm,d_tau):

    p_nm = p_nm + F_nm*d_tau # move centroid mode too
    return p_nm


def Apply_A(q_nm,p_nm,d_tau,m,n_beads):

    # A step treats centroid mode independently
    q_nm[0] = q_nm[0] + d_tau*p_nm[0]/m

    for k in range(1,n_beads):
        q_nm_init = q_nm[k]
        p_nm_init = p_nm[k]

        q_nm[k] = q_nm_init*np.cos(NM_lambda[k]*d_tau) + p_nm_init*np.sin(NM_lambda[k]*d_tau)/(mp*NM_lambda[k])
        p_nm[k] = p_nm_init*np.cos(NM_lambda[k]*d_tau) - mp*NM_lambda[k]*q_nm_init*np.sin(NM_lambda[k]*d_tau)


    return q_nm,p_nm


def Apply_O(p_nm,NM_lambda,beta,d_tau,m,n_beads):
    
    
#    for k in range(n_beads):
    gamma_nm = NM_lambda
    #gamma = 80.0/AU_TO_THZ
    #gamma_nm = (80.0/AU_TO_THZ)*np.ones(len(NM_lambda))
    c1 = np.exp(-gamma_nm*d_tau)
    c2 = (n_beads/beta)*(1.0-c1**2) # must scale std dev of normal r.v due to nm trans
    eta = np.random.normal(size=n_beads)

    p_nm = p_nm*c1 + np.sqrt(m*c2)*eta

    return p_nm

    
def getNMs():

    A = np.zeros((n_beads,n_beads))
    
    
    for i in range(0,n_beads-1):
        A[i][i] = 2.0
        A[i][i+1] = -1.0
        A[i+1][i] = -1.0
        
    A[n_beads-1][n_beads-1] = 2.0
    A[n_beads-1][0] = -1.0
    A[0][n_beads-1] = -1.0
        
    # diagonalize A matrix to obtain normal mode transformation matrix O

    NM_lambda,O = np.linalg.eig(A)
    OT= np.transpose(O)


    for k,l in enumerate(NM_lambda):
        if l<eig_thresh:
            NM_lambda[k] = 0.0

    NM_lambda = np.sqrt(NM_lambda)*n_beads/beta # this is sqrt(n_beads)*omega_k
    NM_lambda[0] = 0.0

    return O,OT,NM_lambda



    



  
def getPIforce(q, m, omega,Lambda):
    F = -1.0*m*omega**2*q - Lambda*q**2
    return F


def getNMrep(O,arr):
    return np.matmul(O,arr)

def getSTDrep(OT,arr):
    return np.matmul(OT,arr)


O,OT,NM_lambda = getNMs()

q = q_init


# burn in sampler
q=update_q(q,O,OT,m,mp,n_beads,beta,n_steps_pimd_burnin,d_tau)



q_samples = []
p_samples = []

# main loop for sampling q

for n in range(n_samples):
    
    # sample q via PIMD
    q=update_q(q,O,OT,m,mp,n_beads,beta,n_steps_pimd,d_tau)
    # choose a bead at random and use this to generate momentum via LGA/LHA/EW
    q_sample = q[np.random.randint(n_beads)]
#    q_samples.append(q_sample)

    for psamp in range(10):
        #    print(m)
        p_sample = LGA(m,omega,Lambda, q_sample)
        q_samples.append(q_sample)
        p_samples.append(p_sample)


# write out samples
for q_sample,p_sample in list(zip(q_samples,p_samples)):
    samples_outfile.write("{}\t{}\n".format(q_sample,p_sample))


    
# propagate resulting initial (q,p) value forward in time classically.
    
pp_tcf = np.zeros(nsteps)

for q,p in list(zip(q_samples,p_samples)):
    p_init = p
    # velocity verlet propagation of B operator
    for step in range(nsteps):
        pp_tcf[step] += (p_init*p)

        F_old = getPIforce(q, m, omega,Lambda)
        q = q + (p/m)*dt + (0.5/m)*(dt**2)*F_old
        F_new = getPIforce(q, m, omega,Lambda)
        p = p + (0.5*dt)*(F_old + F_new)

for step in range(nsteps):
#    print(pp_tcf[step])

    tcf_outfile.write("{}\t{}\n".format(step*dt, pp_tcf[step]/n_samples))



