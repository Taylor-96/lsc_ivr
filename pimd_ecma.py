import numpy as np

dt_aux = 1.0 # timestep for auxilliary PIMD
n_aux_beads = 32
aux_sim_burnin = 200
aux_sim_steps = 2000

V = lambda m,omega,Lambda,x: 0.5*m*omega**2*x**2 + (Lambda/3.0)*x**3
Force = lambda m,omega,Lambda,x: -1.0*m*omega**2*x - Lambda*x**2






n_beads = 32
q_init = np.zeros(n_beads) # single particle in 1 dimension described by n_beads
nsteps = 10000 # number of steps for classical dynamics
n_steps_pimd = 100
n_steps_pimd_burnin = 500



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
sqrt_m = np.sqrt(1836.15)
omega = 1500.0 # i.e. dtau which is hbar*beta but hbar=1 in these reduced units
omega *= CM_INV_TO_AU
Lambda = 21.0 #21kcal mol^-1 Angstrom^-3. kcal/mol -> hartree = 1/627.5094736
Lambda *= KCAL_MOL_ANG_CUB_TO_AU


dt = 10.0 # timestep for classical dynamics to compute tcf
d_tau = 10.0 # timestep for PIMD
hbar=1


mp= m # scaled mass for momenta
n_samples = 10000

samples_outfile = open("pimd_ecma_wigner_samples_w_const_force.out", "w")
tcf_outfile = open("pimd_ecma_tcf_const_force.out", "w")



    
class aux_polymer():
    def __init__(self, n_aux_beads):
        
        self.n_aux_beads = n_aux_beads
        
        self.aux_polymer_X = np.zeros(n_aux_beads)
        self.aux_polymer_P = np.zeros(n_aux_beads)
        self.aux_polymer_F = np.zeros(n_aux_beads+1)
        self.aux_polymer_X_nm = np.zeros(n_aux_beads)
        self.aux_polymer_P_nm = np.zeros(n_aux_beads)
        self.aux_polymer_F_nm = np.zeros(n_aux_beads)

        self.T = np.zeros((n_aux_beads, n_aux_beads))
        self.Ttr = np.zeros((n_aux_beads, n_aux_beads))
        self.C_k_sqrt = np.zeros((n_aux_beads,2,2))
        self.exp_akt = np.zeros((n_aux_beads,2,2)) # this is eq (15) in WiLD supp material
        self.const_force = np.zeros((n_aux_beads,2)) # this is eq (15) in WiLD supp material

    

def update_q(q,O,OT,m,mp,n_beads,beta,n_steps_pimd,d_tau):



    # normal mode momentum distributed as p with std dev scaled by n_beads due to NM transformation
    #p_nm =1/sqrt(n_beads) O_kl p_l
    # initialize PIMD momementum. fictitious momenta are only introduced to recover the prefactor of the density matrix => param m' (mp) doesn't effect estimators so choose to be whatever
    p_nm = np.random.normal(0,np.sqrt(mp*n_beads/beta), size=n_beads) 

    F = getPIforce(q,m,omega,Lambda) 
    F_nm = getNMrep(O,F)

    q_nm = getNMrep(O,q)

#    #print(q_nm)

    # BAOB NM integration as outlined in https://arxiv.org/pdf/1611.06331.pdf
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
    q_nm = q_nm + d_tau*p_nm/m
    

    for k in range(1,n_beads):
        q_nm_init = q_nm
        p_nm_init = p_nm

        q_nm = q_nm_init*np.cos(NM_lambda[k]*d_tau) + p_nm_init*np.sin(NM_lambda[k]*d_tau)/(m*NM_lambda[k])
        p_nm = p_nm_init*np.cos(NM_lambda[k]*d_tau) - mp*NM_lambda[k]*q_nm*np.sin(NM_lambda[k]*d_tau)


    return q_nm,p_nm


def Apply_O(p_nm,NM_lambda,beta,d_tau,m,n_beads):
    
    
#    for k in range(n_beads):
    gamma_nm = NM_lambda
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
    
#    #print("NM_lambda")
 #   #print(NM_lambda)
    for k,l in enumerate(NM_lambda):
        if l<1e-10:
            NM_lambda[k] = 0.0
    NM_lambda = np.sqrt(NM_lambda)*n_beads/beta # this is sqrt(n_beads)*omega_k

    return O,OT,NM_lambda



    


def getPIpot(q, m, omega,Lambda):
      V = 0.5*m*omega**2*q**2 + (Lambda/3.0)*q**3
      #      V = 0.5*m*omega*2*np.sum(q**2) + (Lambda/3.0)*np.sum(q**3)
      return V
  
def getPIforce(q, m, omega,Lambda):
    F = -1.0*m*omega**2*q - Lambda*q**2
    return F


def getNMrep(O,arr):
    return np.matmul(O,arr)

def getSTDrep(OT,arr):
    return np.matmul(OT,arr)




def sampleECMA_momentum(q, n_aux_beads):

    # make q a property of aux_polymer
    polymer = aux_polymer(n_aux_beads)
    PIOUD_setup(q,polymer)
    polymer.aux_polymer_P_nm = np.random.normal(scale=polymer.n_aux_beads/beta, size=polymer.n_aux_beads)
    # #print("aux_p")
  #  #print(polymer.aux_polymer_P_nm)
    
    
    Delta = 0.0
    kappa_2 = 0.0
    kappa_4 = 0.0
    kappa_6 = 0.0
    CEw = 0.0

    # burn in PIOUD sampler
    for n in range(aux_sim_burnin):
        do_PIOUD_step(q, polymer)

    for n in range(aux_sim_steps):
        do_PIOUD_step(q, polymer)
        Delta = 2.*polymer.aux_polymer_X[-1] # last "bead" stores Delta/2
        #print(Delta)
        kappa_2 += Delta**2
        kappa_4 +=Delta**4
        kappa_6 += Delta**6
    

    # average Delta**2
    kappa_2 /= aux_sim_steps
    # average Delta**4 and subtract 3*<Delta**2>
    kappa_4 /= aux_sim_steps
    kappa_4 -= 3.0*kappa_2

    kappa_6 /= aux_sim_steps
    kappa_6 += -15.*kappa_2*kappa_4 + 30.*kappa_2**3

   

    # now sample momentum from p_kappa and get correction weight associated (Cew)
    #print(kappa_2)
    sigma_kappa = 1./np.sqrt(kappa_2)
    p = np.random.normal(sigma_kappa)

    Cew = get_EW_correction(p,kappa_2,kappa_4, kappa_6,6)

    return p,Cew

def get_EW_correction(p,kappa_2,kappa_4,kappa_6,n):

    """
    Return CEW_n


    """

    C_EW4 = (1./24.)*kappa_4*p**4
    C_EW6 = (1./720.)*kappa_6*p**6

    C_EW4_avg = (kappa_4)/(8*kappa_2**2)
    C_Ew6_avg = -(15./720)*kappa_6/kappa_2**3

    correction = 0.0

    if(n==4):
        correction = (1.+C_EW4)/(1. + C_EW4_avg)
    elif(n==6):
        correction = (1.+C_EW4 + C_EW6)/(1. + C_EW4_avg + C_Ew6_avg)

    return correction
    

def PIOUD_setup(q,polymer):

    theta = np.zeros((polymer.n_aux_beads, polymer.n_aux_beads))
    polymer.C_k_sqrt = np.zeros((polymer.n_aux_beads,2,2))
    C_k = np.zeros((polymer.n_aux_beads,2,2))
    C_k_sqrt = np.zeros((polymer.n_aux_beads,2,2))
                        
    
    
    for i in range(polymer.n_aux_beads-1):
        theta[i][i] = 2.0
        theta[i+1][i] = -1.0
        theta[i][i+1] = -1.0
    theta[polymer.n_aux_beads-1][polymer.n_aux_beads-2] = 1.0
    theta[polymer.n_aux_beads-2][polymer.n_aux_beads-1] = 1.0
    theta[0][polymer.n_aux_beads-1] = -1.0
    theta[polymer.n_aux_beads-1][0] = -1.0
    theta[polymer.n_aux_beads-1][polymer.n_aux_beads-1] = 2.0


    # theta = omega*theta?
    
    omega_k, polymer.T = np.linalg.eig(theta)
    polymer.Ttr= np.transpose(polymer.T)


    # check if eigenvalues are zero to within threshold (necessary for taking square root to avoid negative values)
    for k,l in enumerate(omega_k):
        if l<1e-10:
            omega_k[k] = 0.0
            

    omega_k = np.sqrt(omega_k)
    omega_k *= (polymer.n_aux_beads/beta)

    exp_omega_kdt = np.exp(-1.0*omega_k*dt_aux)
    exp_akt = np.zeros((polymer.n_aux_beads,2,2)) # this is eq (15) in WiLD supp material
#    I_min_exp_akt_times_ak_inv
    a_k = np.zeros((polymer.n_aux_beads,2,2)) # this is eq (15) in WiLD supp material
    a_kinv = np.zeros((polymer.n_aux_beads,2,2)) # this is eq (15) in WiLD supp material



    mu_q = np.zeros(polymer.n_aux_beads)

    mu_q[0] = sqrt_m*q
    mu_q[polymer.n_aux_beads-2] = sqrt_m*q

    mu_q_nm = np.matmul(polymer.Ttr,mu_q)

    
    
    for k in range(polymer.n_aux_beads):


        
        exp_akt[k][0][0] = exp_omega_kdt[k]*(1. - omega_k[k]*dt_aux)
        exp_akt[k][0][1] = exp_omega_kdt[k]*(-dt_aux*omega_k[k]**2)
        exp_akt[k][1][0] = exp_omega_kdt[k]*dt_aux
        exp_akt[k][1][1] = exp_omega_kdt[k]*(1+ omega_k[k]*dt_aux)

        a_k[k][0][0] = 2.*omega_k[k]   # = gamma_k[k]
        a_k[k][0][1] = omega_k[k]**2
        a_k[k][1][0] = -1.
        a_k[k][1][1] = 0.

        a_kinv[k][0][0] = 0.0
        a_kinv[k][0][1] = -1.
        a_kinv[k][0][0] = 1./omega_k[k]**2
        a_kinv[k][0][0] = 2./omega_k[k]
        

    for k in range(polymer.n_aux_beads):    
        C_k[k][0][0] = (polymer.n_aux_beads/beta)* (1. - exp_omega_kdt[k]**2*(1. - 2.*dt_aux*omega_k[k] + 2.*omega_k[k]**2*dt_aux**2))
        C_k[k][1][1] = (1./omega_k[k]**2)*(polymer.n_aux_beads/beta)* (1. - exp_omega_kdt[k]**2 *(1. + 2.*dt_aux*omega_k[k] + 2.*omega_k[k]**2*dt_aux**2))
        C_k[0][1] = (polymer.n_aux_beads/beta)*2.*omega_k[k]*dt_aux**2*exp_omega_kdt[k] # squared?
        C_k[1][0] = C_k[0][1]
        
        # lower triangular part of cholesky decomp of C_k        
        C_k_sqrt[k][0][0] = np.sqrt(C_k[k][0][0])
        C_k_sqrt[k][0][1] = 0.0
        C_k_sqrt[k][1][0] = C_k[k][1][0]/C_k_sqrt[k][0][0]
        # print("C_k_sqrt[k][1][0]")
        # print(C_k_sqrt[k][1][0])
        # print("C_k[k][1][1]")
        # print(C_k[k][1][1])
        if((C_k[k][1][1]-C_k_sqrt[k][1][0]**2)<0.0):
            print(k)
            print(C_k[k][1][1]-C_k_sqrt[k][1][0]**2)
        C_k_sqrt[k][1][1] = np.sqrt(C_k[k][1][1]-C_k_sqrt[k][1][0]**2)


    const_force_mat = np.zeros((polymer.n_aux_beads,2,2)) # this is eq (15) in WiLD supp material

    for k in range(polymer.n_aux_beads):
        const_force_mat[k] = (np.eye(2) - exp_akt[k])
        const_force_mat[k] = np.matmul(const_force_mat[k], a_kinv[k])

        polymer.const_force[k][0] = const_force_mat[k][0][0]*mu_q_nm[k]
        polymer.const_force[k][1] = const_force_mat[k][1][0]*mu_q_nm[k]

    
    #  P_nm +=  const_force_mat[k][0][0]*mu_q_nm[k]
    # X_nm += const_force_mat[k][1][0]*mu_q_nm[k]
    polymer.C_k_sqrt = C_k_sqrt
    #print("polymer.C_k_sqrt")
    #print(polymer.C_k_sqrt)
    polymer.exp_akt = exp_akt
    
    
def do_PIOUD_step(q,polymer):
    
    n1 = np.random.normal(size=(polymer.n_aux_beads))
    n2 = np.random.normal(size=(polymer.n_aux_beads))



    

    
    aux_polymer_X_nm_init = polymer.aux_polymer_X_nm
    aux_polymer_P_nm_init = polymer.aux_polymer_P_nm
    for k in range(polymer.n_aux_beads):
        #print("polymer.C_k_sqrt[k]")
        #print(polymer.C_k_sqrt[k])
        #print(n2)
        #print(polymer.exp_akt[k][1][0])
        #print(aux_polymer_X_nm_init)
        #print("aux p")
        #print(aux_polymer_P_nm_init)
        # polymer.aux_polymer_X_nm[k] = polymer.exp_akt[k][1][1]*aux_polymer_X_nm_init[k] + polymer.exp_akt[k][1][0]*aux_polymer_P_nm_init[k] + polymer.C_k_sqrt[k][1][0]*n1[k] + polymer.C_k_sqrt[k][1][1]*n2[k] 
        
        # polymer.aux_polymer_P_nm[k] = polymer.exp_akt[k][0][1]*aux_polymer_X_nm_init[k] + polymer.exp_akt[k][0][0]*aux_polymer_P_nm_init[k]  + polymer.C_k_sqrt[k][0][0]*n1[k] 

        polymer.aux_polymer_X_nm[k] = polymer.exp_akt[k][1][1]*aux_polymer_X_nm_init[k] + polymer.exp_akt[k][1][0]*aux_polymer_P_nm_init[k] + polymer.C_k_sqrt[k][1][0]*n1[k] + polymer.C_k_sqrt[k][1][1]*n2[k] + polymer.const_force[k][1]
        
        polymer.aux_polymer_P_nm[k] = polymer.exp_akt[k][0][1]*aux_polymer_X_nm_init[k] + polymer.exp_akt[k][0][0]*aux_polymer_P_nm_init[k] + polymer.const_force[k][0] + polymer.C_k_sqrt[k][0][0]*n1[k] 

    #print(",polymer.aux_polymer_X_nm)")
    #print(polymer.aux_polymer_X_nm)
    polymer.aux_polymer_X = (1/np.sqrt(m))*np.matmul(polymer.T,polymer.aux_polymer_X_nm)  # Convert to std representation, remembering X = sqrt(m)*x
        
    #    polymer.aux_polymer_F =  get_bead_forces(q,polymer)
    get_bead_forces(q,polymer)
    apply_bead_forces(polymer)
    #polymer.aux_polymer_P_nm = apply_bead_forces(polymer)

    
    

    
def apply_bead_forces(polymer):

    F = np.zeros(polymer.n_aux_beads)
    F[0:polymer.n_aux_beads-1] = polymer.aux_polymer_F[0:polymer.n_aux_beads-1]/np.sqrt(m)
    F[polymer.n_aux_beads-1] = (polymer.aux_polymer_F[polymer.n_aux_beads-1] - polymer.aux_polymer_F[polymer.n_aux_beads])/np.sqrt(m)
    # perform nm transformation of forces

    #print("F")
    #print(F)
    F_nm = np.matmul(polymer.Ttr,F)

    polymer.aux_polymer_P_nm += dt_aux*F_nm
    
    
def get_bead_forces(q,polymer):


    F = np.zeros(polymer.n_aux_beads+1)  # extra element to store both contributions from V(q+Delta/2) and
                                 # V(q-Delta/2)
    for k in range(polymer.n_aux_beads-1):
        F[k] = Force(m,omega,Lambda,q)

    polymer.aux_polymer_F[polymer.n_aux_beads-1] = 0.5*Force(m,omega,Lambda,q+polymer.aux_polymer_X[-1])
    polymer.aux_polymer_F[polymer.n_aux_beads] = 0.5*Force(m,omega,Lambda,q-polymer.aux_polymer_X[-1])
    



O,OT,NM_lambda = getNMs()
q = q_init




# burn in sampler
q=update_q(q,O,OT,m,mp,n_beads,beta,n_steps_pimd_burnin,d_tau)



q_samples = []
p_samples = []
Cew_list  = []

# main loop for sampling q

for n in range(n_samples):
    
    # sample q via PIMD
    q=update_q(q,O,OT,m,mp,n_beads,beta,n_steps_pimd,d_tau)
    # choose a bead at random and use this to generate momentum via LGA/LHA/EW
    q_sample = q[np.random.randint(n_beads)]
    q_samples.append(q_sample)


    p_sample, Cew = sampleECMA_momentum(q_sample, n_aux_beads)
    p_samples.append(p_sample)
    Cew_list.append(Cew)

    
# write out samples
for q_sample,p_sample,Cew in list(zip(q_samples,p_samples,Cew_list)):
    samples_outfile.write("{}\t{}\t{}\n".format(q_sample,p_sample,Cew))



    
# propagate resulting initial (q,p) value forward in time classically.
    
pp_tcf = np.zeros(nsteps)

for q,p,Cew in list(zip(q_samples,p_samples,Cew_list)):
    #print(q)
    #print(p)
    p_init = p
    # velocity verlet propagation of B operator
    for step in range(nsteps):
        pp_tcf[step] += (p_init*p)*Cew
        F_old = getPIforce(q, m, omega,Lambda)
        q = q + (p/m)*dt + (0.5/m)*(dt**2)*F_old
        F_new = getPIforce(q, m, omega,Lambda)
        p = p + (0.5*dt)*(F_old + F_new)

for step in range(nsteps):
#    #print(pp_tcf[step])

    tcf_outfile.write("{}\t{}\n".format(step*dt, pp_tcf[step]/n_samples))







