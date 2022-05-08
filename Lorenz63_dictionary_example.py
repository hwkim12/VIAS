import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import kv

np.random.seed(12345)

# Initial state
x0 = -8
y0 = 7
z0 = 27

# Times step
T = 40
NN = 2000

x = np.zeros(NN)
y = np.zeros(NN)
z = np.zeros(NN)
dx = np.zeros(NN)
dy = np.zeros(NN)
dz = np.zeros(NN)

x[0] = x0
y[0] = y0
z[0] = z0

# Generating approximated derivative
for i in range(NN-1):
    dx[i] = 10*(y[i]-x[i])
    dy[i] = (x[i]*(28-z[i])-y[i])
    dz[i] = (x[i]*y[i] - 8*z[i]/3)
    
    x[i+1] = x[i] + dx[i]*(T/NN)
    y[i+1] = y[i] + dy[i]*(T/NN)
    z[i+1] = z[i] + dz[i]*(T/NN)
    
dx[NN-1] = 10*(y[NN-1]-x[NN-1])
dy[NN-1] = (x[NN-1]*(28-z[NN-1])-y[NN-1])
dz[NN-1] = (x[NN-1]*y[NN-1] - 8*z[NN-1]/3)

# Obtaining noise-incorporated approximated derivative, which is our data
sigma = 0.3
bx = dx + np.random.normal(0, sigma, NN)
by = dy + np.random.normal(0, sigma, NN)
bz = dz + np.random.normal(0, sigma, NN)

bx = pd.DataFrame(bx)
by = pd.DataFrame(by)
bz = pd.DataFrame(bz)
x = pd.DataFrame(x)
y = pd.DataFrame(y)
z = pd.DataFrame(z)
dx = pd.DataFrame(dx)
dy = pd.DataFrame(dy)
dz = pd.DataFrame(dz)

# Constructing dictionary matrix
order1 = pd.concat([x, y, z], axis = 1)
order2 = pd.concat([x**2, y**2, z**2, x*y, x*z, y*z], axis = 1)
order3 = pd.concat([x**3, y**3, z**3, x**2*y, x*y**2, x**2*z, x*z**2, y**2*z, y*z**2, x*y*z], axis = 1)
order4 = pd.concat([x**4, y**4, z**4, x**3*y, x**2*y**2, x*y**3, x**3*z, x**2*z**2, x*z**3, y**3*z, y**2*z**2, y*z**3, x**2*y*z, x*y**2*z, x*y*z**2], axis = 1)
order5 = pd.concat([x**5, y**5, z**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, x**4*z, x**3*z**2, x**2*z**3, x*z**4, z**4*y, z**3*y**2, z**2*y**3, z*y**4, x**3*y*z, x**2*y**2*z, x**2*y*z**2, x*y**3*z, x*y**2*z**2, x*y*z**3], axis = 1)

order1.columns = ['x','y','z']
order2.columns = ['x**2','y**2','z**2', 'x*y', 'x*z', 'y*z']
order3.columns = ['x**3','y**3','z**3', 'x**2*y', 'x*y**2', 'x**2*z', 'x*z**2', 'y**2*z', 'y*z**2', 'x*y*z']
order4.columns = ['x**4','y**4','z**4', 'x**3*y', 'x**2*y**2', 'x*y**3', 'x**3*z', 'x**2*z**2', 'x*z**3', 'y**3*z', 'y**2*z**2', 'y*z**3', 'x**2*y*z', 'x*y**2*z', 'x*y*z**2']
order5.columns = ['x**5','y**5','z**5', 'x**4*y', 'x**3*y**2', 'x**2*y**3', 'x*y**4', 'x**4*z', 'x**3*z**2', 'x**2*z**3', 'x*z**4', 'z**4*y', 'z**3*y**2', 'z**2*y**3', 'z*y**4', 'x**3*y*z', 'x**2*y**2*z', 'x**2*y*z**2', 'x*y**3*z', 'x*y**2*z**2', 'x*y*z**3']

A = pd.concat([order1, order2, order3, order4, order5], axis = 1)

# Parameters for hierarchical model
alpha = 0.005
beta = 0.05

b = 2*beta
s = alpha - 0.5

# Number of IAS/VIAS iterations
niter = 5

# X trajectory parameter
PHI1 = np.zeros(len(A.columns)) ### TRUE parameter value
PHI1[0] = -10
PHI1[1] = 10

# Y trajectory parameter
PHI2 = np.zeros(len(A.columns)) ### TRUE parameter value
PHI2[0] = 28
PHI2[1] = -1
PHI2[7] = -1

# Z trajectory parameter
PHI3 = np.zeros(len(A.columns)) ### TRUE parameter value
PHI3[2] = -8/3
PHI3[6] = 1

# IAS for x trajectory
# IAS implementation
# alpha is assumed to be one
eta = 1e-5
theta_x_ias = np.zeros(len(A.columns))
u_x_ias = np.zeros(len(A.columns))

# IAS iteration for x trajectory
iter = 0
for iter in range(niter):
    theta_x_ias = np.sqrt(eta**2/4 + u_x_ias**2/2) + eta/2
    theta_x_ias = np.reshape(np.asarray(theta_x_ias), (1, len(theta_x_ias)))[0]
    D_theta = np.diag(1./theta_x_ias)
    u_x_ias = np.linalg.inv(D_theta + (np.transpose(A)@A/sigma**2).values)@np.transpose(A)@bx/sigma**2

theta_x_ias = np.reshape(np.asarray(theta_x_ias), (1, len(theta_x_ias)))[0]
u_x_ias = np.reshape(np.asarray(u_x_ias), (1, len(u_x_ias)))[0]
H11 = np.transpose(A)@A/sigma**2 + np.diag(np.asarray(1/np.reshape(np.asarray(1/theta_x_ias), (1, len(theta_x_ias)))[0]))
H22 = np.diag(u_x_ias**2/theta_x_ias**3 + eta/theta_x_ias**2)
H12 = -np.diag(u_x_ias/theta_x_ias**2)
Laplace_block = np.block([[H11, H12], [H12, H22]])
Cov = np.linalg.inv(Laplace_block)
Cov_x_u = Cov[:len(A.columns), :len(A.columns)]

# initialization
m_x = np.ones(len(A.columns))
C_x = np.eye(len(A.columns))
a_x = m_x**2 + np.array(np.diag(C_x))

# VIAS for x trajectory
k = 0
for k in range(niter):
    L = np.zeros(len(A.columns))
    for i in range(len(A.columns)):
        L[i] = kv(s-1, np.sqrt(a_x[i]*b))/kv(s, np.sqrt(a_x[i]*b))*np.sqrt(b/a_x[i])
    D = np.diag(L)
    
    C_x = np.linalg.inv((np.transpose(A)@A)/sigma**2 + D)
    m_x = C_x@np.transpose(A)@bx/sigma**2
    m_x = np.transpose(m_x)
    a_x = m_x**2 + np.array(np.diag(C_x))

m_x = np.transpose(m_x)
m_x = np.array(m_x)

# Plot for x trajectory
plt.figure(figsize=(5,5))
plt.plot(range(len(A.columns)), PHI1, '*', label = 'Truth', color = 'red', alpha = 1)
plt.plot(range(len(A.columns)), m_x, '.', label = 'VIAS', color = 'blue', alpha = 1)
plt.plot(range(len(A.columns)), u_x_ias, 'p', label = 'IAS', color = 'green', alpha = 1)
#plt.errorbar(range(len(A.columns)), m_x, yerr = 1.96*np.sqrt(np.diag(C_x)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='VIAS')
plt.title('x trajectory parameter estimate')
plt.legend(loc = 'upper right')
plt.tight_layout()

# IAS for y trajectory
# IAS implementation
# alpha is assumed to be one
eta = 1e-5
theta_y_ias = np.ones(len(A.columns))
u_y_ias = np.ones(len(A.columns))

iter = 0
for iter in range(niter):    
    theta_y_ias = np.sqrt(eta**2/4 + u_y_ias**2/2) + eta/2
    theta_y_ias = np.reshape(np.asarray(theta_y_ias), (1, len(theta_y_ias)))[0]
    D_theta = np.diag(1./theta_y_ias)
    u_y_ias = np.linalg.inv(D_theta + (np.transpose(A)@A/sigma**2).values)@np.transpose(A)@by/sigma**2

theta_y_ias = np.reshape(np.asarray(theta_y_ias), (1, len(theta_y_ias)))[0]
u_y_ias = np.reshape(np.asarray(u_y_ias), (1, len(u_y_ias)))[0]
H11 = np.transpose(A)@A/sigma**2 + np.diag(np.asarray(1/np.reshape(np.asarray(1/theta_y_ias), (1, len(theta_y_ias)))[0]))
H22 = np.diag(u_y_ias**2/theta_y_ias**3 + eta/theta_y_ias**2)
H12 = -np.diag(u_y_ias/theta_y_ias**2)
Laplace_block = np.block([[H11, H12], [H12, H22]])
Cov = np.linalg.inv(Laplace_block)
Cov_y_u = Cov[:len(A.columns), :len(A.columns)]

# initialization
m_y = np.ones(len(A.columns))
C_y = np.eye(len(A.columns))
a_y = m_y**2 + np.array(np.diag(C_y))

# VIAS for y trajectory
k = 0
for k in range(niter):
    L = np.zeros(len(A.columns))
    for i in range(len(A.columns)):
        L[i] = kv(s-1, np.sqrt(a_y[i]*b))/kv(s, np.sqrt(a_y[i]*b))*np.sqrt(b/a_y[i])
    D = np.diag(L)
    
    C_y = np.linalg.inv((np.transpose(A)@A)/sigma**2 + D)
    m_y = C_y@np.transpose(A)@by/sigma**2
    m_y = np.transpose(m_y)
    a_y = m_y**2 + np.array(np.diag(C_y))

m_y = np.transpose(m_y)
m_y = np.array(m_y)
plt.figure(figsize=(5,5))
plt.plot(range(len(A.columns)), PHI2, '*', label = 'Truth', color = 'red', alpha = 1)
plt.plot(range(len(A.columns)), m_y, '.', label = 'VIAS', color = 'blue', alpha = 1)
plt.plot(range(len(A.columns)), u_y_ias, 'p', label = 'IAS', color = 'green', alpha = 1)
#plt.errorbar(range(len(A.columns)), m_y, yerr = 1.96*np.sqrt(np.diag(C_y)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='VIAS')
plt.title('y trajectory parameter estimate')
plt.legend(loc = 'upper right')
plt.tight_layout()

# IAS for z trajectory
# IAS implementation
# alpha is assumed to be one
eta = 1e-5
theta_z_ias = np.ones(len(A.columns))
u_z_ias = np.ones(len(A.columns))

iter = 0
for iter in range(niter): 
    theta_z_ias = np.sqrt(eta**2/4 + u_z_ias**2/2) + eta/2
    theta_z_ias = np.reshape(np.asarray(theta_z_ias), (1, len(theta_z_ias)))[0]
    D_theta = np.diag(1./theta_z_ias)
    u_z_ias = np.linalg.inv(D_theta + (np.transpose(A)@A/sigma**2).values)@np.transpose(A)@bz/sigma**2


theta_z_ias = np.reshape(np.asarray(theta_z_ias), (1, len(theta_z_ias)))[0]
u_z_ias = np.reshape(np.asarray(u_z_ias), (1, len(u_z_ias)))[0]
H11 = np.transpose(A)@A/sigma**2 + np.diag(np.asarray(1/np.reshape(np.asarray(1/theta_z_ias), (1, len(theta_z_ias)))[0]))
H22 = np.diag(u_z_ias**2/theta_z_ias**3 + eta/theta_z_ias**2)
H12 = -np.diag(u_z_ias/theta_z_ias**2)
Laplace_block = np.block([[H11, H12], [H12, H22]])
Cov = np.linalg.inv(Laplace_block)
Cov_z_u = Cov[:len(A.columns), :len(A.columns)]

# initialization
m_z = np.ones(len(A.columns))
C_z = np.eye(len(A.columns))
a_z = m_z**2 + np.array(np.diag(C_z))

# VIAS for z trajectory
k = 0
for k in range(niter):
    L = np.zeros(len(A.columns))
    for i in range(len(A.columns)):
        L[i] = kv(s-1, np.sqrt(a_z[i]*b))/kv(s, np.sqrt(a_z[i]*b))*np.sqrt(b/a_z[i])
    D = np.diag(L)
    
    C_z = np.linalg.inv((np.transpose(A)@A)/sigma**2 + D)
    m_z = C_z@np.transpose(A)@bz/sigma**2
    m_z = np.transpose(m_z)
    a_z = m_z**2 + np.array(np.diag(C_z))

m_z = np.transpose(m_z)
m_z = np.array(m_z)
plt.figure(figsize=(5,5))
plt.plot(range(len(A.columns)), PHI3, '*', label = 'Truth', color = 'red', alpha = 1)
plt.plot(range(len(A.columns)), m_z, '.', label = 'VIAS', color = 'blue', alpha = 1)
plt.plot(range(len(A.columns)), u_z_ias, 'p', label = 'IAS', color = 'green', alpha = 1)
#plt.errorbar(range(len(A.columns)), m_z, yerr = 1.96*np.sqrt(np.diag(C_z)), fmt='.', color='black', ecolor='gray', capsize=5, capthick = 1.25, label='VIAS')
plt.title('z trajectory parameter estimate')
plt.legend(loc = 'upper right')
plt.tight_layout()
