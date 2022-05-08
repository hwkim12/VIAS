import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.special import kv

np.random.seed(1)

# Simulation setup
d = 200
n = 50

alpha = 0.005
beta = 0.05

eta = 1e-5

b = 2*beta
s = alpha - 0.5

# Obtain true parameters
theta_true = st.gamma.rvs(scale = 1/beta, a = alpha, size = d)
u = st.multivariate_normal.rvs(mean = np.zeros(d), cov = np.diag(theta_true))

# Plot the true parameter
plt.figure()
plt.plot(u, 's', label = 'Truth', color = 'coral', alpha = 1)
plt.title("Synthetic truth u", fontsize = 16)
plt.tight_layout()

# Construction of Matrix and Data
A = np.random.rand(n,d)
noise_free_y = A@u

error_sd = 0.05*np.max(abs(noise_free_y))
error_var = error_sd**2
y = noise_free_y + st.norm.rvs(0, error_sd, n)

# Plot Data 
plt.figure()
plt.plot(noise_free_y, '.', label = 'Without Error', color = 'green')
plt.plot(y, '.', label = 'Observed Data', color = 'black')
plt.title("Data $y = Au + \eta$", fontsize = 16)
plt.legend(fontsize = 10, loc = 'upper left')
plt.tight_layout()


# Below is ELBO for the model selection
def ELBO_model(m1, C1, a1, alpha1, beta1, error_var1, eps1):
    s1 = alpha1 - 0.5
    b1 = 2*beta1
    
    value1 = -np.matrix.trace(A@C1@np.transpose(A))/error_var1 - np.transpose(A@m1)@(A@m1)/error_var1 + np.log(np.linalg.det(C1)+eps1)
    value1 = value1/2 + np.transpose(y)@A@m1/error_var1
    for ii in range(len(a1)):
        value1 = value1 - s1*np.log(b1/a1[ii])/2 + np.log(2*kv(s1, np.sqrt(a1[ii]*b1)))
    
    return value1

# Below is ELBO for m, C, a
def ELBO_reg(m2, C2, a2, alpha2, beta2, error_var2, eps2):
    s2 = alpha2 - 0.5
    b2 = 2*beta2
    
    value2 = -np.matrix.trace(A@C2@np.transpose(A))/error_var2 - np.transpose(A@m2)@(A@m2)/error_var2 + np.log(np.linalg.det(C2)+eps2)
    value2 = value2/2 + np.transpose(y)@A@m2/error_var
    for ii in range(len(a2)):
        value2 = value2 + s2*np.log(a2[ii])/2 + np.log(2*kv(s2, np.sqrt(a2[ii]*b2)))
    
    return value2

# VIAS implementation starts from here
# Initialization
m = np.ones(d)
C = np.identity(d)
a = m**2 + np.diag(C)
niter = 1000
ELBO_true = np.zeros(niter)

difm = []
difa = []
difC = []
difelb = []

# VIAS iteration
for k in range(niter):
    L = np.zeros(d)
    for i in range(d):
        L[i] = kv(s-1, np.sqrt(a[i]*b))/kv(s, np.sqrt(a[i]*b))*np.sqrt(b/a[i])
    D = np.diag(L)
    
    prevC = np.copy(C)
    prevm = np.copy(m)
    preva = np.copy(a)
    
    C = np.linalg.inv((np.transpose(A)@A)/error_var + D)
    m = C@np.transpose(A)@y/error_var
    a = m**2 + np.diag(C)
    
    eps = 0.0000001 # eps was added to compute ELBO otherwise logdet term blows up
   
    ELBO_true[k] = ELBO_model(m, C, a, alpha, beta, error_var, eps)
    
    difC.append(np.linalg.norm(np.reshape(C-prevC, -1), ord = np.inf))
    difm.append(np.linalg.norm(m-prevm, ord = np.inf))
    difa.append(np.linalg.norm(a-preva, ord = np.inf))
    if k > 0:
        difelb.append(np.abs(ELBO_true[k]-ELBO_true[k-1]))
    
# VIAS result
plt.figure(figsize=(5,5))
plt.plot(u, 'o', label ='Truth',color = 'coral', alpha = 1)
plt.errorbar(np.linspace(1,d,d), m, yerr = 1.96*np.sqrt(np.diag(C)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='VIAS')
plt.legend(fontsize = 10, loc = 'upper left')
plt.xlabel('u')
plt.title('VIAS with true parameters', fontsize = 16)
plt.tight_layout()
    

# Use ELBO as the model selection
cur_alpha = cur_beta = 1; cur_max = -1e9
alphas = np.logspace(-3,1, 20)
betas = np.logspace(3,4, 20)

for k in betas:
    for j in alphas:
        alpha = j
        beta = k
        b = 2*beta
        s = alpha - 0.5
        
        
        m = np.zeros(d)
        C = np.identity(d)
        a = m**2 + np.diag(C)
        n_iter = 300
        ELBO = np.zeros(niter)
        elb = []
        elb.append(0)
        
        eps = 0.0000001

        for kk in range(n_iter):
            L = np.zeros(d)
            for i in range(d):
                L[i] = kv(s-1, np.sqrt(a[i]*b))/kv(s, np.sqrt(a[i]*b))*np.sqrt(b/a[i])
            D = np.diag(L)
    
            C = np.linalg.inv((np.transpose(A)@A)/error_var + D)
            m = C@np.transpose(A)@y/error_var
            a = m**2 + np.diag(C)
             # eps was added to compute ELBO otherwise logdet term blows up
            ELBO[kk] = ELBO_model(m, C, a, alpha, beta, error_var, eps)
            
            if ELBO[kk] > cur_max:
                cur_max = ELBO[kk]
                cur_alpha = j
                cur_beta = k

# Ballpark estimate of hyperparameters        
print(cur_alpha)
print(cur_beta)

alpha = cur_alpha
beta = cur_beta

b = 2*beta
s = alpha - 0.5

# Initialization
m = np.zeros(d)
C = np.identity(d)
a = m**2 + np.diag(C)

ELBO_mod = np.zeros(niter)
difm_model = []
difa_model = []
difC_model = []
difelb_model = []


# Rerun the VIAS based on the ballpark estimate of hyperparameters
for k in range(niter):
    L = np.zeros(d)
    for i in range(d):
        L[i] = kv(s-1, np.sqrt(a[i]*b))/kv(s, np.sqrt(a[i]*b))*np.sqrt(b/a[i])
    D = np.diag(L)
    
    prevC = np.copy(C)
    prevm = np.copy(m)
    preva = np.copy(a)
    
    C = np.linalg.inv((np.transpose(A)@A)/error_var + D)
    m = C@np.transpose(A)@y/error_var
    a = m**2 + np.diag(C)
    
    eps = 0.0000001 # eps was added to compute ELBO otherwise logdet term blows up
    
    ELBO_mod[k] = ELBO_model(m, C, a, alpha, beta, error_var, eps)

    difC_model.append(np.linalg.norm(np.reshape(C-prevC, -1), ord = np.inf))
    difm_model.append(np.linalg.norm(m-prevm, ord = np.inf))
    difa_model.append(np.linalg.norm(a-preva, ord = np.inf))
    if k > 0:
        difelb_model.append(np.abs(ELBO_mod[k]-ELBO_mod[k-1]))
    
# VIAS with model selection 
plt.figure(figsize=(5,5))
plt.plot(u, 'o', label ='Truth',color = 'coral', alpha = 1)
plt.errorbar(np.linspace(1,d,d), m, yerr = 1.96*np.sqrt(np.diag(C)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='VIAS')
plt.legend()
plt.xlabel('u')
plt.title('VIAS with model selection', fontsize = 16)
plt.tight_layout()

# Change of ELBO values over iterations
plt.figure(figsize=(5,5))
plt.plot(range(niter), ELBO_mod, color = 'black', label = 'Model selected parameters')
plt.plot(range(niter), ELBO_true, color = 'blue', label = 'True parameters')
plt.xlabel("Number of iterations")
plt.ylabel("ELBO")
plt.title('ELBO values', fontsize = 16)
plt.legend(fontsize = 10)
plt.tight_layout()

# Convergence plot for each parameter when using true hyperparameter
plt.figure(figsize=(5,5))
plt.plot(range(niter), difm, label = '$\Delta \, m$', marker = '.')
plt.plot(range(niter), difa, label = '$\Delta \, a$', marker = '.')
plt.plot(range(niter), difC, label = '$\Delta \, C$', marker = '.')
plt.plot(range(niter-1), difelb, label = '$\Delta \, ELBO$', marker = '.')
plt.yscale('log')
plt.legend(fontsize = 10)
plt.tight_layout()
plt.xlabel("Number of iterations")
plt.ylabel("Change in value")
plt.title("Convergence - true parameters", fontsize = 16)

# Convergence plot for each parameter when using model selection based hyperparameter
plt.figure(figsize=(5,5))
plt.plot(range(niter), difm_model, label = '$\Delta \, m$', marker = '.')
plt.plot(range(niter), difa_model, label = '$\Delta \, a$', marker = '.')
plt.plot(range(niter), difC_model, label = '$\Delta \, C$', marker = '.')
plt.plot(range(niter-1), difelb_model, label = '$\Delta \, ELBO$', marker = '.')
plt.yscale('log')
plt.legend(fontsize = 10)
plt.tight_layout()
plt.xlabel("Number of iterations")
plt.ylabel("Change in value")
plt.title("Convergence - model selected parameters", fontsize = 16)


# IAS implementation
# Parameter specification
eta = 1e-5
theta_ias = np.zeros(d)
u_ias = np.zeros(d)

# IAS iterations
for iter in range(niter):
    theta_ias = np.sqrt(eta**2/4 + u_ias**2/2) + eta/2
    D_theta = np.diag(1/theta_ias)
    u_ias = np.linalg.inv(np.transpose(A)@A/error_var + D_theta)@np.transpose(A)@y/error_var

# Covariance for Laplace approximation    
H11 = np.transpose(A)@A/error_var + np.diag(1/theta_ias)
H22 = np.diag(u_ias**2/theta_ias**3 + eta/theta_ias**2)
H12 = -np.diag(u_ias/theta_ias**2)
Laplace_block = np.block([[H11, H12], [H12, H22]])
Cov = np.linalg.inv(Laplace_block)
Cov_u = Cov[:d, :d]

# IAS with Laplace approximation
plt.figure(figsize=(5,5))
plt.plot(u, 'o', label ='Truth',color = 'coral', alpha = 1)
plt.errorbar(np.linspace(1,d,d), u_ias, yerr = 1.96*np.sqrt(np.diag(Cov_u)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='IAS')
plt.legend(fontsize = 10, loc = 'upper left')
plt.xlabel('u')
plt.title('IAS with Laplace approximation', fontsize = 16)
plt.tight_layout()
