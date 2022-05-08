import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.special import kv

np.random.seed(1234)

# ELBO for model selection
def ELBO_model(m1, C1, a1, alpha1, beta1, error_var1, eps1):
    s1 = alpha1 - 0.5
    b1 = 2*beta1
    
    value1 = -np.matrix.trace(A@C1@np.transpose(A))/error_var1 - np.transpose(A@m1)@(A@m1)/error_var1 + np.log(np.linalg.det(C1)+eps1)
    value1 = value1/2 + np.transpose(y)@A@m1/error_var1
    for ii in range(len(a1)):
        value1 = value1 - s1*np.log(b1/a1[ii])/2 + np.log(2*kv(s1, np.sqrt(a1[ii]*b1)))
    
    return value1

# Simulation setup
d = 100
n = 50

alpha = 0.005
beta = 0.05

b = 2*beta
s = alpha - 0.5

# Obtain true parameters
u = np.zeros(d)
signal = [5,15,25,35,45,55,65,75,85,95]
u[signal] = 1

# Construction of matrix and data
A = np.random.rand(n,d)
noise_free_y = A@u

error_sd = 0.02*np.max(abs(noise_free_y))
error_var = error_sd**2
y = noise_free_y + st.norm.rvs(0, error_sd, n)

# Initialization
m = np.ones(d)
C = np.identity(d)
a = m**2 + np.diag(C)
niter = 1000
ELBO = np.zeros(niter)

# VIAS implementation
for k in range(niter):
    L = np.zeros(d)
    for i in range(d):
        L[i] = kv(s-1, np.sqrt(a[i]*b))/kv(s, np.sqrt(a[i]*b))*np.sqrt(b/a[i])
    D = np.diag(L)
    
    C = np.linalg.inv((np.transpose(A)@A)/error_var + D)
    m = C@np.transpose(A)@y/error_var
    a = m**2 + np.diag(C)
    
    eps = 0.0000001 # eps was added to compute ELBO otherwise logdet term blows up
    
    ELBO[k] = np.log(np.linalg.det(C)+eps) - np.matrix.trace(A@C@np.transpose(A)) - np.matrix.trace(D@C) - np.transpose(m)@D@m - np.transpose(A@m-y)@(A@m-y)
    ELBO[k] = ELBO[k]/2
    for ii in range(d):
        ELBO[k] = ELBO[k] - s*np.log(b/a[ii])/2 + np.log(2*kv(s, np.sqrt(a[ii]*b))) + a[ii]*D[ii,ii]/2

# IAS implementation
# alpha is assumed to be one
eta = 1e-5
theta_ias = np.zeros(d)
u_ias = np.zeros(d)

# IAS iteration
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

# LASSO implementation
from sklearn.linear_model import LassoLarsIC, LassoLarsCV
model_bic = LassoLarsIC(criterion='bic', fit_intercept=False)
model_bic.fit(A, y)
alpha_bic_ = model_bic.alpha_

model_aic = LassoLarsIC(criterion='aic',fit_intercept=False )
model_aic.fit(A, y)
alpha_aic_ = model_aic.alpha_
 
model_CV = LassoLarsCV(cv=20, normalize=False).fit(A, y)

# Just plotting the result of VIAS
plt.figure()
plt.plot(range(d), u, 's', label = 'Truth', color = 'coral', alpha = 1)
plt.errorbar(np.linspace(1,d,d), m, yerr = 1.96*np.sqrt(np.diag(C)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='VIAS')
plt.title('VIAS', fontsize = 16)
plt.legend(loc = "center right", fontsize = 10)
    

# Use ELBO as the model selection
cur_alpha = cur_beta = 1; cur_max = -1e9
alphas = np.logspace(-4,1, 20)
betas = np.logspace(1,2, 20)

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

# Obtain a ballpark estimate of parameters                
print(cur_alpha)
print(cur_beta)

alpha = cur_alpha
beta = cur_beta

b = 2*beta
s = alpha - 0.5

# Initialization
m = np.ones(d)
C = np.identity(d)
a = m**2 + np.diag(C)

ELBO_mod = np.zeros(niter)
difm_model = []
difa_model = []
difC_model = []
difelb_model = []

# rerun the VIAS with the ballpark estimate of paramters
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
    
# VIAS estimation result
plt.figure()
plt.plot(range(d), u, 's', label = 'Truth', color = 'coral', alpha = 1)
plt.errorbar(np.linspace(1,d,d), m, yerr = 1.96*np.sqrt(np.diag(C)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='VIAS')
plt.title('VIAS estimation', fontsize = 16)
plt.legend(loc = "center right", fontsize = 10)
    

# Plotting variants of model selection methods
plt.figure()
plt.plot(range(d), u, 's', label = 'Truth', color = 'coral', alpha = 1)
#plt.errorbar(range(d), u_ias, yerr=1.96*np.sqrt(np.diag(Cov_u)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='IAS')
plt.plot(range(d), u_ias, 'o', label = 'IAS', color = 'blue', alpha = 1)
plt.errorbar(np.linspace(1,d,d), m, yerr = 1.96*np.sqrt(np.diag(C)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='VIAS')
plt.plot(model_CV.coef_, 'o', label = "LASSO(CV)", color = "red", alpha = 1)
#plt.plot(model_aic.coef_, 'o', label = "LASSO(AIC)", color = "purple", alpha = 1)
#plt.plot(model_bic.coef_, 'o', label = "LASSO(BIC)", color = "green", alpha = 1)
plt.title('Comparison with other methods', fontsize = 16)
plt.legend(loc = "center right", fontsize = 10)
  
