
#%%

import os
import numpy as np
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.integrate import quad, nquad
from scipy.special import hyp2f1
from scipy.special import beta as betaf

import ray

#################################
####### Hetnet Parameters #######
#################################

sq_m = 1000**2
alpha_list = np.arange(0.05,0.325,0.025).tolist()
lambda_0 = 300/(np.pi*500**2)*sq_m
lambda_1 = np.NaN
lambda_2 = 30/(np.pi*500**2)*sq_m
lambda_3 = 6/(np.pi*500**2)*sq_m

P_1 = 73
P_2 = 373
P_3 = 1773
beta = 4

M1 = 5
M2 = 50
N = 200

def cache_contents_distribution(a:int, b:int, N=200, gamma=0.8):

    assert a > 0 and b > 0
    zipf = 0
    zipf_deno = 0

    for j in range(1,N):
        zipf_deno += 1/(j**gamma)

    for i in range(a,b):
        zipf += 1/i**gamma/zipf_deno

    return zipf

#%%
############################################
####### Average Ergodic Rate PPP-PPP #######
############################################

def M_y(s, x, lambda_1, P_i):
    return np.exp(-2*np.pi*lambda_1*x**2*s*(P_1/P_i)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
                    -2*np.pi*lambda_2*x**2*s*(P_2/P_i)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
                    -2*np.pi*lambda_3*x**2*s*(P_3/P_i)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
    )

def Hamdi(s, x, lambda_1, P_i):
    return M_y(s, x, lambda_1, P_i)/(1+s)

def pdf_asso1(x, lambda_1):
    return  2 * np.pi * lambda_1 * x/proba_tier_asso_1 * np.exp(-np.pi*(lambda_1*(P_1/P_1)**(2/beta)+lambda_2*(P_2/P_1)**(2/beta)+lambda_3*(P_3/P_1)**(2/beta))*x**2)

def pdf_asso2(x, lambda_1):
    return  2 * np.pi * lambda_2 * x/proba_tier_asso_2 * np.exp(-np.pi*(lambda_1*(P_1/P_2)**(2/beta)+lambda_2*(P_2/P_2)**(2/beta)+lambda_3*(P_3/P_2)**(2/beta))*x**2)

def pdf_asso3(x, lambda_1):
    return  2 * np.pi * lambda_3 * x/proba_tier_asso_3 * np.exp(-np.pi*(lambda_1*(P_1/P_3)**(2/beta)+lambda_2*(P_2/P_3)**(2/beta)+lambda_3*(P_3/P_3)**(2/beta))*x**2)

def average_ergodic_rate_case1_asso1(s,x, lambda_1, P_i=P_1):

    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)

    return Hamdi(s, x, lambda_1_active, P_i)*pdf_asso1(x, lambda_1)

def average_ergodic_rate_case1_asso2(s,x, lambda_1, P_i=P_2):

    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)

    return Hamdi(s, x, lambda_1_active, P_i)*pdf_asso2(x, lambda_1)

def average_ergodic_rate_case1_asso3(s,x, lambda_1, P_i=P_3):

    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)

    return Hamdi(s, x, lambda_1_active, P_i)*pdf_asso3(x, lambda_1)

Average_Ergodic_Rate_case1_PPP = []
U_1_1_PPP = []
U_1_2_PPP = []
U_1_3_PPP = []
G_3_1_PPP = []
G_3_2_PPP = []
G_3_3_PPP = []
for alpha in tqdm(alpha_list):

    # Number of Cache-enabled user
    lambda_1 = lambda_0*alpha

    # Association probability (ALL PPP)
    # max power is 1st tier (Association to 1)
    lambda_i = lambda_1
    proba_tier_asso_1 = ((lambda_1/lambda_i)*(P_1/P_1)**(2/beta)+(lambda_2/lambda_i)*(P_2/P_1)**(2/beta)+(lambda_3/lambda_i)*(P_3/P_1)**(2/beta))**(-1)

    # max power is 2nd tier (Association to 2)
    lambda_i = lambda_2
    proba_tier_asso_2 = ((lambda_1/lambda_i)*(P_1/P_2)**(2/beta)+(lambda_2/lambda_i)*(P_2/P_2)**(2/beta)+(lambda_3/lambda_i)*(P_3/P_2)**(2/beta))**(-1)

    # max power is 3rd tier (Association to 3)
    lambda_i = lambda_3
    proba_tier_asso_3 = ((lambda_1/lambda_i)*(P_1/P_3)**(2/beta)+(lambda_2/lambda_i)*(P_2/P_3)**(2/beta)+(lambda_3/lambda_i)*(P_3/P_3)**(2/beta))**(-1)

    # Active D2D transmitters yielding interference
    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)
    diff = lambda_1_active-lambda_1

    average_ergodic_rate_case1_asso1_N = quad(lambda x: quad(lambda s: average_ergodic_rate_case1_asso1(s,x, lambda_1), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0], 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*proba_tier_asso_1
    average_ergodic_rate_case1_asso2_N = quad(lambda x: quad(lambda s: average_ergodic_rate_case1_asso2(s,x, lambda_1), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0], 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*proba_tier_asso_2
    average_ergodic_rate_case1_asso3_N = quad(lambda x: quad(lambda s: average_ergodic_rate_case1_asso3(s,x, lambda_1), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0], 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*proba_tier_asso_3

    average_ergodic_rate_case1 = average_ergodic_rate_case1_asso1_N + average_ergodic_rate_case1_asso2_N + average_ergodic_rate_case1_asso3_N
    print('Average Ergodic Rate (alpha={:.4f}, active_vs_available={:.6f}): {:.4f}'.format(alpha,diff, average_ergodic_rate_case1))
    Average_Ergodic_Rate_case1_PPP.append(average_ergodic_rate_case1)
    U_1_1_PPP.append(average_ergodic_rate_case1_asso1_N)
    U_1_2_PPP.append(average_ergodic_rate_case1_asso2_N)
    U_1_3_PPP.append(average_ergodic_rate_case1_asso3_N)
    G_3_1_PPP.append(proba_tier_asso_1)
    G_3_2_PPP.append(proba_tier_asso_2)
    G_3_3_PPP.append(proba_tier_asso_3)


#%%

def M_y2(s, x, lambda_1, P_i):
    a = 0.0000000003
    return np.exp(-2*np.pi*lambda_1*x**2*s*(P_1/P_i)*(a/x)**(2-beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s*(P_1/P_i)/(a/x)**(beta))\
                -2*np.pi*lambda_2*x**2*s*(P_2/P_i)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
                -2*np.pi*lambda_3*x**2*s*(P_3/P_i)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
    )

def Hamdi2(s, x, lambda_1, P_i):
    return M_y2(s, x, lambda_1, P_i)/(1+s)

def pdf_oasso23(x, lambda_1):
    return 2 * np.pi * lambda_2 * x/proba_tier_oasso23 * np.exp(-np.pi*(lambda_2*(P_2/P_2)**(2/beta)+lambda_3*(P_3/P_2)**(2/beta))*x**2)

def pdf_oasso32(x, lambda_1):
    return 2 * np.pi * lambda_3 * x/proba_tier_oasso32 * np.exp(-np.pi*(lambda_2*(P_2/P_3)**(2/beta)+lambda_3*(P_3/P_3)**(2/beta))*x**2)

def average_ergodic_rate_case2_oasso23(s, x, lambda_1, P_i=P_2):

    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)

    return Hamdi2(s, x, lambda_1_active, P_i)*pdf_oasso23(x, lambda_1)

def average_ergodic_rate_case2_oasso32(s, x, lambda_1, P_i=P_3):

    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)

    return Hamdi2(s, x, lambda_1_active, P_i)*pdf_oasso32(x, lambda_1)

Average_Ergodic_Rate_case2_PPP = []
U_2_2_PPP = []
U_2_3_PPP = []
P_2_3_PPP = []
P_3_2_PPP = []
for alpha in tqdm(alpha_list):

    # Number of Cache-enabled user
    lambda_1 = lambda_0*alpha

    # Association probability (ALL PPP)
    # max power is 1st tier (Association to 1)
    proba_tier_asso_1 = ((lambda_1/lambda_1)*(P_1/P_1)**(2/beta)+(lambda_2/lambda_1)*(P_2/P_1)**(2/beta)+(lambda_3/lambda_1)*(P_3/P_1)**(2/beta))**(-1)

    # Ordered Tier Association probability (ALL PPP)
    # max power order 2>3
    lambda_i = lambda_2
    proba_tier_oasso23 = ((lambda_2/lambda_i)*(P_2/P_2)**(2/beta)+(lambda_3/lambda_i)*(P_3/P_2)**(2/beta))**(-1)
    # max power order 3>2
    lambda_i = lambda_3
    proba_tier_oasso32 = ((lambda_2/lambda_i)*(P_2/P_3)**(2/beta)+(lambda_3/lambda_i)*(P_3/P_3)**(2/beta))**(-1)

    # Active D2D transmitters yielding interference
    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)
    diff = lambda_1_active - lambda_1

    average_ergodic_rate_case2_oasso23_N = quad(lambda x: quad(lambda s: average_ergodic_rate_case2_oasso23(s,x, lambda_1), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0], 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*proba_tier_oasso23
    average_ergodic_rate_case2_oasso32_N = quad(lambda x: quad(lambda s: average_ergodic_rate_case2_oasso32(s,x, lambda_1), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0], 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*proba_tier_oasso32

    average_ergodic_rate_case2 = average_ergodic_rate_case2_oasso23_N + average_ergodic_rate_case2_oasso32_N
    print('Average Ergodic Rate (alpha={:.4f}, active_vs_available={:.6f}): {:.4f}'.format(alpha,diff, average_ergodic_rate_case2))
    Average_Ergodic_Rate_case2_PPP.append(average_ergodic_rate_case2)
    U_2_2_PPP.append(average_ergodic_rate_case2_oasso23_N)
    U_2_3_PPP.append(average_ergodic_rate_case2_oasso32_N)
    P_2_3_PPP.append(proba_tier_oasso23)
    P_3_2_PPP.append(proba_tier_oasso32)


#%%

# Note: quad nested is correct, dblquad etc are not correct 
# changed region of integral with respect to y s.t. (P_1/P_2)**(1/beta)*y) -> (P_1/P_2)**(-1/beta)*y)
# swap (P_1/P_j) to (P_j/P_1) region of integral of interference from tier 2 and 3

def M_y3(s, x, y, lambda_1, P_j):
    return np.exp(-2*np.pi*lambda_1*y**2*s*(x/y)**(2-beta)*(P_1/P_j)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s*(P_1/P_j)/(x/y)**(beta))\
                -2*np.pi*lambda_2*y**2*s*(P_2/P_j)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
                -2*np.pi*lambda_3*y**2*s*(P_3/P_j)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
    )

def Hamdi3(s, x, y, lambda_1, P_j):
    return M_y3(s, x, y, lambda_1, P_j)/(1+s)

def pdf_oasso123(x, y, lambda_1):
    return 4*np.pi**2*lambda_1*lambda_2*x*y/proba_tier_oasso123*np.exp(-np.pi*lambda_1*x**2-np.pi*lambda_2*y**2*(1+lambda_3/lambda_2*(P_3/P_2)**(2/beta)))

def pdf_oasso132(x, y, lambda_1):
    return 4*np.pi**2*lambda_1*lambda_3*x*y/proba_tier_oasso132*np.exp(-np.pi*lambda_1*x**2-np.pi*lambda_3*y**2*(1+lambda_2/lambda_3*(P_2/P_3)**(2/beta)))

def average_ergodic_rate_case3_oasso123(s, x, y, lambda_1, P_j=P_2):

    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)

    return Hamdi3(s, x, y, lambda_1_active, P_j)*pdf_oasso123(x, y, lambda_1)

def average_ergodic_rate_case3_oasso132(s, x, y, lambda_1, P_j=P_3):

    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)

    return Hamdi3(s, x, y, lambda_1_active, P_j)*pdf_oasso132(x, y, lambda_1)

Average_Ergodic_Rate_case3_PPP = []
U_3_2_PPP = []
U_3_3_PPP = []
P_1_2_3_PPP = []
P_1_3_2_PPP = []
for alpha in tqdm(alpha_list):

    # Number of Cache-enabled user
    lambda_1 = lambda_0*alpha

    # Association probability (ALL PPP)
    # max power is 1st tier (Association to 1)
    proba_tier_asso_1 = ((lambda_1/lambda_1)*(P_1/P_1)**(2/beta)+(lambda_2/lambda_1)*(P_2/P_1)**(2/beta)+(lambda_3/lambda_1)*(P_3/P_1)**(2/beta))**(-1)

    # Ordered Tier Association probability (ALL PPP)
    # max power order 1>2>3
    proba_tier_oasso123 = (1+(lambda_3/lambda_2)*(P_3/P_2)**(2/beta))**(-1)*((lambda_1/lambda_1)*(P_1/P_1)**(2/beta)+(lambda_2/lambda_1)*(P_2/P_1)**(2/beta)+(lambda_3/lambda_1)*(P_3/P_1)**(2/beta))**(-1)
    # max power order 1>3>2
    proba_tier_oasso132 = (1+(lambda_2/lambda_3)*(P_2/P_3)**(2/beta))**(-1)*((lambda_1/lambda_1)*(P_1/P_1)**(2/beta)+(lambda_2/lambda_1)*(P_2/P_1)**(2/beta)+(lambda_3/lambda_1)*(P_3/P_1)**(2/beta))**(-1)

    # Active D2D transmitters yielding interference
    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)
    diff = lambda_1_active - lambda_1

    average_ergodic_rate_case3_oasso123_N = quad(lambda y: quad(lambda x: quad(lambda s: average_ergodic_rate_case3_oasso123(s,x,y, lambda_1),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0], 0, (P_1/P_2)**(1/beta)*y)[0], 0, 1e6,epsabs=1e-2,epsrel=1e-2,points=[0,1])[0]*proba_tier_oasso123/(proba_tier_oasso123+proba_tier_oasso132)
    average_ergodic_rate_case3_oasso132_N = quad(lambda y: quad(lambda x: quad(lambda s: average_ergodic_rate_case3_oasso132(s,x,y,lambda_1),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0], 0, (P_1/P_3)**(1/beta)*y)[0], 0, 1e6,epsabs=1e-2,epsrel=1e-2,points=[0,1])[0]*proba_tier_oasso132/(proba_tier_oasso123+proba_tier_oasso132)

    average_ergodic_rate_case3 = average_ergodic_rate_case3_oasso123_N + average_ergodic_rate_case3_oasso132_N
    print('Average Ergodic Rate (alpha={:.4f}, active_vs_available={:.6f}): {:.4f}'.format(alpha,diff, average_ergodic_rate_case3))
    Average_Ergodic_Rate_case3_PPP.append(average_ergodic_rate_case3)
    U_3_2_PPP.append(average_ergodic_rate_case3_oasso123_N)
    U_3_3_PPP.append(average_ergodic_rate_case3_oasso132_N)
    P_1_2_3_PPP.append(proba_tier_oasso123)
    P_1_3_2_PPP.append(proba_tier_oasso132)


#%%

#############################################
####### Average Ergodic Rate PPP-PPCP #######
#############################################

from scipy.stats import rice
# PPP-PPCP
lambda_2_parent = 3/(np.pi*500**2)*sq_m #1000 was best
m_bar = 10
sigma = 0.05#0.02
#lambda_1 = 0.1*lambda_0 # test case

def besseli(z):
    return 1/np.pi*quad(lambda p: np.exp(z*np.cos(p)),0,np.pi,epsabs=1e-3,epsrel=1e-3)[0]

def ricepdf(a,b,s):
    s2 = s**2
    z = besseli(a*b/s2)
    return (b/s2)*np.exp(-0.5*(a**2+b**2)/s2)*z

def ricecdf(a,b,s):
    return quad(lambda x: ricepdf(a,x,s),0,b,epsabs=1e-3,epsrel=1e-3)[0]

# Tier association PPP-PPCP
def tau_i(r,i):

    if i == 2:
        P_i = P_2
        return m_bar*quad(lambda z: 2*np.pi*lambda_2_parent*z/sigma*rice.pdf(r/sigma,z/sigma)*np.exp(-m_bar*(rice.cdf((P_2/P_i)**(1/beta)*r/sigma,z/sigma))),0,np.inf)[0]
    elif i == 1:
        return 2*np.pi*lambda_1*r
    elif i == 3:
        return 2*np.pi*lambda_3*r
        
def pgfl_Q1(r,i):
    if i == 1:
        P_i = P_1
        return quad(lambda z: -lambda_2_parent*2*np.pi*z*(1-np.exp(-m_bar*(rice.cdf((P_2/P_i)**(1/beta)*r/sigma,z/sigma)))),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0]
    elif i == 2:
        P_i = P_2
        return quad(lambda z: -lambda_2_parent*2*np.pi*z*(1-np.exp(-m_bar*(rice.cdf((P_2/P_i)**(1/beta)*r/sigma,z/sigma)))),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0]
    elif i == 3:
        P_i = P_3
        return quad(lambda z: -lambda_2_parent*2*np.pi*z*(1-np.exp(-m_bar*(rice.cdf((P_2/P_i)**(1/beta)*r/sigma,z/sigma)))),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0]

def tasso_ppcp_c(i):
    
    if i == 1:
        P_i = P_1
        return quad(lambda r: tau_i(r,i)*np.exp(pgfl_Q1(r,i))*np.exp(-np.pi*(lambda_1*(P_1/P_i)**(2/beta)*r**2+lambda_3*(P_3/P_i)**(2/beta)*r**2)),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0]
    elif i == 2:
        P_i = P_2
        return quad(lambda r: tau_i(r,i)*np.exp(pgfl_Q1(r,i))*np.exp(-np.pi*(lambda_1*(P_1/P_i)**(2/beta)*r**2+lambda_3*(P_3/P_i)**(2/beta)*r**2)),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0]
    elif i == 3:
        P_i = P_3
        return quad(lambda r: tau_i(r,i)*np.exp(pgfl_Q1(r,i))*np.exp(-np.pi*(lambda_1*(P_1/P_i)**(2/beta)*r**2+lambda_3*(P_3/P_i)**(2/beta)*r**2)),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0]

# Ordered Tier association PPP-PPCP

def oasso_ppcp_c(i):

    if i == 2:
        P_i = P_2
        return quad(lambda r: tau_i(r,i)*np.exp(pgfl_Q1(r,i))*np.exp(-np.pi*lambda_3*(P_3/P_i)**(2/beta)*r**2),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0]
    elif i == 3:
        P_i = P_3
        return quad(lambda r: tau_i(r,i)*np.exp(pgfl_Q1(r,i))*np.exp(-np.pi*lambda_3*(P_3/P_i)**(2/beta)*r**2),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0]

# Faster
def oasso3_ppcp_c(j):
    # 1>2>3
    if j == 2:
        return quad(lambda r_1: quad(lambda r_2: tau_i(r_1,1)*tau_i(r_2,2)*np.exp(pgfl_Q1(r_2,2))*np.exp(-np.pi*lambda_1*r_1**2)*np.exp(-np.pi*lambda_3*(P_3/P_2)**(2/beta)*r_2**2),(P_2/P_1)**(1/beta)*r_1,np.inf,epsabs=1e-2,epsrel=1e-2)[0],0,np.inf,epsabs=1e-2,epsrel=1e-2)[0]
    # 1>3>2
    elif j == 3:
        return quad(lambda r_1: quad(lambda r_3: tau_i(r_1,1)*tau_i(r_3,3)*np.exp(pgfl_Q1(r_3,3))*np.exp(-np.pi*lambda_1*r_1**2)*np.exp(-np.pi*lambda_3*r_3**2),(P_3/P_1)**(1/beta)*r_1,np.inf,epsabs=1e-2,epsrel=1e-2)[0],0,np.inf,epsabs=1e-2,epsrel=1e-2)[0]

def oasso3_ppcp_c2(j):
    if j == 2:
        return quad(lambda r_1: quad(lambda r_2: quad(lambda r_3: tau_i(r_2,2)*np.exp(pgfl_Q1(r_2,2))*tau_i(r_1,1)*np.exp(-np.pi*lambda_1*r_1**2)*tau_i(r_3,3)*np.exp(-np.pi*lambda_3*r_3**2),(P_3/P_2)**(1/beta)*r_2,np.Inf,epsabs=1e-2,epsrel=1e-2)[0],(P_2/P_1)**(1/beta)*r_1,np.Inf,epsabs=1e-2,epsrel=1e-2)[0],0,np.Inf,epsabs=1e-2,epsrel=1e-2)[0]
    if j == 3:
        return quad(lambda r_1: quad(lambda r_3: quad(lambda r_2: tau_i(r_2,2)*np.exp(pgfl_Q1(r_2,2))*tau_i(r_1,1)*np.exp(-np.pi*lambda_1*r_1**2)*tau_i(r_3,3)*np.exp(-np.pi*lambda_3*r_3**2),(P_2/P_3)**(1/beta)*r_3,np.Inf,epsabs=1e-2,epsrel=1e-2)[0],(P_3/P_1)**(1/beta)*r_1,np.Inf,epsabs=1e-2,epsrel=1e-2)[0],0,np.Inf,epsabs=1e-2,epsrel=1e-2)[0]

#!!! Test Functions !!!
#quad(lambda x: tau_i(x,2)*math.exp(pgfl_Q1(x,2)), 0, np.inf, epsabs=1e-3, epsrel=1e-3)
#tasso_ppcp_c1 = tasso_ppcp_c(1)
#print(tasso_ppcp_c1)
#tasso_ppcp_c2 = tasso_ppcp_c(2)
#print(tasso_ppcp_c2)
#tasso_ppcp_c3 = tasso_ppcp_c(3)
#print(tasso_ppcp_c3)

#print(oasso3_ppcp_c(2)) #0.26
#print(oasso3_ppcp_c2(2)) #0.26
#Rgiht!
#print(oasso3_ppcp_c(3)) #0.207666
#print(oasso3_ppcp_c2(3)) # 0.207666
#Right!

# Note: np.exp(-y**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma**2)/(1+(s*P_2/P_i)**(-1)*((y+z)/x)**(beta)), (P_2/P_i)**(1/beta)*x, np.inf, epsabs=1e-3)[0]
# vs np.exp(-y**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma**2)/(1+(s*P_2/P_i)**(1)*((y+z)/x)**(-beta)), (P_2/P_i)**(1/beta)*x, np.inf, epsabs=1e-3)[0]
# right: z*(1-np.exp(-m_bar*(pgfl_ppcp1(s,x,z,P_i))))
def thomas(y):
    return math.exp(-y**2/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma**2)

def pgfl_ppcp1(s, x, z, P_i):
    return quad(lambda y: thomas(y)/(1+(s*P_2/P_i)**(-1)*((y+z)/x)**(beta)), (P_2/P_i)**(1/beta)*x, np.inf, epsabs=1e-3, epsrel=1e-3)[0]

def pgfl_ppcp2(s, x, P_i):
    return quad(lambda z: z*(1-np.exp(-m_bar*(pgfl_ppcp1(s,x,z,P_i)))), 0, np.inf, epsabs=1e-3, epsrel=1e-3)[0]

#%%
def M_y_ppcp(s, x, lambda_1, P_i):
    return np.exp(-2*np.pi*lambda_1*x**2*s*(P_1/P_i)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
                    -2*np.pi*lambda_2_parent*pgfl_ppcp2(s,x,P_i)\
                    -2*np.pi*lambda_3*x**2*s*(P_3/P_i)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
    )

#!!! Test Functions !!!
#print(quad(lambda x: -2*np.pi*lambda_2_parent*pgfl_ppcp2(1,x,P_1), 0, np.inf, epsabs=1e-3))
#print(quad(lambda x: -2*np.pi*lambda_1*x**2*(P_1/P_1)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-1), 0, np.inf, epsabs=1e-3))
#print(quad(lambda x: M_y(2,x,lambda_0*0.05,P_1),0,np.inf,epsabs=1e-3))
#print(quad(lambda x: M_y_ppcp(2,x,lambda_0*0.05,P_1),0,np.inf,epsabs=1e-3))

def Hamdi_ppcp(s,x, lambda_1, P_i):
    return M_y_ppcp(s, x, lambda_1, P_i)/(1+s)

def pdf_asso1_ppcp(x, lambda_1, tasso_ppcp):
    return tau_i(x,1)/tasso_ppcp*np.exp(pgfl_Q1(x,1))*np.exp(-np.pi*(lambda_1*x**2+lambda_3*(P_3/P_1)**(2/beta)*x**2))

def pdf_asso2_ppcp(x, lambda_1, tasso_ppcp):
    return tau_i(x,2)/tasso_ppcp*np.exp(pgfl_Q1(x,2))*np.exp(-np.pi*(lambda_1*(P_1/P_2)**(2/beta)*x**2+lambda_3*(P_3/P_2)**(2/beta)*x**2))

def pdf_asso3_ppcp(x, lambda_1, tasso_ppcp):
    return tau_i(x,3)/tasso_ppcp*np.exp(pgfl_Q1(x,3))*np.exp(-np.pi*(lambda_1*(P_1/P_3)**(2/beta)*x**2+lambda_3*x**2))

#!!! Test Functions !!!
#print(quad(lambda x: x*pdf_asso1_ppcp(x,lambda_1,tasso_ppcp_c1), 0, np.inf))
#print(quad(lambda x: x*pdf_asso2_ppcp(x,lambda_1,tasso_ppcp_c2), 0, np.inf))
#print(quad(lambda x: x*pdf_asso3_ppcp(x,lambda_1,tasso_ppcp_c3), 0, np.inf))

def average_ergodic_rate_case1_ppcp_c(x, lambda_1, P_i, i, lambda_1_active, tasso_ppcp):

    if i == 1:
        return quad(lambda s: Hamdi_ppcp(s, x, lambda_1_active, P_i), 0, np.inf, epsabs=1e-3, epsrel=1e-3)[0]*pdf_asso1_ppcp(x, lambda_1, tasso_ppcp)
    elif i == 2:
        return quad(lambda s: Hamdi_ppcp(s, x, lambda_1_active, P_i), 0, np.inf, epsabs=1e-3, epsrel=1e-3)[0]*pdf_asso2_ppcp(x, lambda_1, tasso_ppcp)
    elif i == 3:
        return quad(lambda s: Hamdi_ppcp(s, x, lambda_1_active, P_i), 0, np.inf, epsabs=1e-3, epsrel=1e-3)[0]*pdf_asso3_ppcp(x, lambda_1, tasso_ppcp)

Average_Ergodic_Rate_case1_PPCP = []
U_1_1_PPCP = []
U_1_2_PPCP = []
U_1_3_PPCP = []
G_3_1_PPCP = []
G_3_2_PPCP = []
G_3_3_PPCP = []
for alpha in tqdm(alpha_list):

    # Number of Cache-enabled user
    lambda_1 = lambda_0*alpha

    # Active D2D transmitters yielding interference
    active_lambda_1 = (1-alpha)*lambda_0*tasso_ppcp_c(1)*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)
    diff = lambda_1_active-lambda_1

    tasso_ppcp1 = tasso_ppcp_c(1)
    average_ergodic_rate_case1_asso1_N_ppcp = quad(lambda x: average_ergodic_rate_case1_ppcp_c(x, lambda_1, P_1, 1, lambda_1_active, tasso_ppcp1), 0, np.inf, epsabs=1e-3, epsrel=1e-3)[0]*tasso_ppcp1
    tasso_ppcp2 = tasso_ppcp_c(2)
    average_ergodic_rate_case1_asso2_N_ppcp = quad(lambda x: average_ergodic_rate_case1_ppcp_c(x, lambda_1, P_2, 2, lambda_1_active, tasso_ppcp2), 0, np.inf, epsabs=1e-3, epsrel=1e-3)[0]*tasso_ppcp2
    tasso_ppcp3 = tasso_ppcp_c(3)
    average_ergodic_rate_case1_asso3_N_ppcp = quad(lambda x: average_ergodic_rate_case1_ppcp_c(x, lambda_1, P_3, 3, lambda_1_active, tasso_ppcp3), 0, np.inf, epsabs=1e-3, epsrel=1e-3)[0]*tasso_ppcp3

    average_ergodic_rate_case1_ppcp = average_ergodic_rate_case1_asso1_N_ppcp + average_ergodic_rate_case1_asso2_N_ppcp + average_ergodic_rate_case1_asso3_N_ppcp
    print('Average Ergodic Rate (alpha={:.4f}, active_vs_available={:.6f}): {:.4f}[Tier1:{:.4},Tier2:{:.4},Tier3:{:.4}]'.format(alpha,diff, average_ergodic_rate_case1_ppcp, average_ergodic_rate_case1_asso1_N_ppcp, average_ergodic_rate_case1_asso2_N_ppcp, average_ergodic_rate_case1_asso3_N_ppcp))
    Average_Ergodic_Rate_case1_PPCP.append(average_ergodic_rate_case1_ppcp)
    U_1_1_PPCP.append(average_ergodic_rate_case1_asso1_N_ppcp)
    U_1_2_PPCP.append(average_ergodic_rate_case1_asso2_N_ppcp)
    U_1_3_PPCP.append(average_ergodic_rate_case1_asso3_N_ppcp)
    G_3_1_PPCP.append(tasso_ppcp1)
    G_3_2_PPCP.append(tasso_ppcp2)
    G_3_3_PPCP.append(tasso_ppcp3)

#%%

def M_y_ppcp2(s, x, lambda_1, P_i):
    a = 0.000000000003
    return np.exp(-2*np.pi*lambda_1*x**2*s*(P_1/P_i)*(a/x)**(2-beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s*(P_1/P_i)/(a/x)**(beta))\
                    -2*np.pi*lambda_2_parent*pgfl_ppcp2(s,x,P_i)\
                    -2*np.pi*lambda_3*x**2*s*(P_3/P_i)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
    )

def Hamdi_ppcp2(s,x, lambda_1, P_i):
    return M_y_ppcp2(s, x, lambda_1, P_i)/(1+s)

def pdf_oasso23_ppcp(x, lambda_1, oasso23_ppcp):
    return tau_i(x,2)/oasso23_ppcp*np.exp(pgfl_Q1(x,2))*np.exp(-np.pi*lambda_3*(P_3/P_2)**(2/beta)*x**2)

def pdf_oasso32_ppcp(x, lambda_1, oasso32_ppcp):
    return tau_i(x,3)/oasso32_ppcp*np.exp(pgfl_Q1(x,3))*np.exp(-np.pi*lambda_3*x**2)

def average_ergodic_rate_case2_ppcp_c(x, lambda_1, P_i, i, lambda_1_active, oasso_ppcp):

    if i == 2:
        return quad(lambda s: Hamdi_ppcp2(s, x, lambda_1_active, P_i), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*pdf_oasso23_ppcp(x, lambda_1, oasso_ppcp)
    elif i == 3:
        return quad(lambda s: Hamdi_ppcp2(s, x, lambda_1_active, P_i), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*pdf_oasso32_ppcp(x, lambda_1, oasso_ppcp)

Average_Ergodic_Rate_case2_PPCP = []
U_2_2_PPCP = []
U_2_3_PPCP = []
P_2_3_PPCP = []
P_3_2_PPCP = []
for alpha in tqdm(alpha_list):

    # Number of Cache-enabled user
    lambda_1 = lambda_0*alpha

    # Active D2D transmitters yielding interference
    active_lambda_1 = (1-alpha)*lambda_0*tasso_ppcp_c(1)*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)
    diff = lambda_1_active-lambda_1

    oasso23_ppcp = oasso_ppcp_c(2)
    average_ergodic_rate_case2_oasso23_N_ppcp = quad(lambda x: average_ergodic_rate_case2_ppcp_c(x, lambda_1, P_2, 2, lambda_1_active, oasso23_ppcp), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*oasso23_ppcp
    oasso32_ppcp = oasso_ppcp_c(3)
    average_ergodic_rate_case2_oasso32_N_ppcp = quad(lambda x: average_ergodic_rate_case2_ppcp_c(x, lambda_1, P_3, 3, lambda_1_active, oasso32_ppcp), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*oasso32_ppcp

    average_ergodic_rate_case2_ppcp = average_ergodic_rate_case2_oasso23_N_ppcp + average_ergodic_rate_case2_oasso32_N_ppcp
    print('Average Ergodic Rate (alpha={:.4f}, active_vs_available={:.6f}): {:.4f}'.format(alpha,diff, average_ergodic_rate_case2_ppcp))
    Average_Ergodic_Rate_case2_PPCP.append(average_ergodic_rate_case2_ppcp)
    U_2_2_PPCP.append(average_ergodic_rate_case2_oasso23_N_ppcp)
    U_2_3_PPCP.append(average_ergodic_rate_case2_oasso32_N_ppcp)
    P_2_3_PPCP.append(oasso23_ppcp)
    P_3_2_PPCP.append(oasso32_ppcp)


#%%

def pgfl_ppcp1_case3(s, y, z, P_j):
    return quad(lambda y_: thomas(y_)/(1+(s*P_2/P_j)**(-1)*((y_+z)/y)**(beta)), (P_2/P_j)**(1/beta)*y, np.inf, epsabs=1e-3, epsrel=1e-3)[0]

def pgfl_ppcp2_case3(s, y, P_j):
    return quad(lambda z: z*(1-np.exp(-m_bar*(pgfl_ppcp1_case3(s,y,z,P_j)))), 0, np.inf, epsabs=1e-3, epsrel=1e-3)[0]

def M_y_ppcp3(s, x, y, lambda_1, P_j):
    
    return np.exp(-2*np.pi*lambda_1*y**2*s*(x/y)**(2-beta)*(P_1/P_j)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s*(P_1/P_j)/(x/y)**(beta))\
                -2*np.pi*lambda_2_parent*pgfl_ppcp2_case3(s,y,P_j)\
                -2*np.pi*lambda_3*y**2*s*(P_3/P_j)**(2/beta)/(beta*(1-2/beta))*hyp2f1(1,1-2/beta,2-2/beta,-s)\
    )

def Hamdi_ppcp3(s,x, y, lambda_1, P_j):
    return M_y_ppcp3(s, x, y, lambda_1, P_j)/(1+s)

def pdf_oasso123_ppcp(x, y, lambda_1, oasso3_ppcp):

    return tau_i(x,1)*tau_i(y,2)/oasso3_ppcp*np.exp(-np.pi*lambda_1*x**2)*np.exp(pgfl_Q1(y,2))*np.exp(-np.pi*lambda_3*(P_3/P_2)**(2/beta)*y**2)

def pdf_oasso132_ppcp(x, y, lambda_1, oasso3_ppcp):

    return tau_i(x,1)*tau_i(y,3)/oasso3_ppcp*np.exp(-np.pi*lambda_1*x**2)*np.exp(pgfl_Q1(y,3))*np.exp(-np.pi*lambda_3*y**2)

def average_ergodic_rate_case3_ppcp_c(x, y, lambda_1, P_j, j, lambda_1_active, oasso3_ppcp):

    if j == 2:
        return quad(lambda s: Hamdi_ppcp3(s, x, y, lambda_1_active, P_j), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*pdf_oasso123_ppcp(x, y, lambda_1, oasso3_ppcp)
    elif j == 3:
        return quad(lambda s: Hamdi_ppcp3(s, x, y, lambda_1_active, P_j), 0, np.inf,epsabs=1e-3,epsrel=1e-3)[0]*pdf_oasso132_ppcp(x, y, lambda_1, oasso3_ppcp)

#!!! Test Functions !!!
#oasso123_ppcp = oasso3_ppcp_c(2)
#print(quad(lambda y: quad(lambda x: pdf_oasso123_ppcp(x,y,lambda_1,oasso123_ppcp),0,(P_2/P_1)**(-1/beta)*y,epsabs=1e-2,epsrel=1e-2)[0],0,np.inf,epsabs=1e-2,epsrel=1e-2)[0])

#oasso132_ppcp = oasso3_ppcp_c(3)
#print(quad(lambda y: quad(lambda x: pdf_oasso132_ppcp(x,y,lambda_1,oasso132_ppcp),0,(P_3/P_1)**(-1/beta)*y,epsabs=1e-3,epsrel=1e-3)[0],0,np.inf,epsabs=1e-3,epsrel=1e-3)[0])

# PPP-PPCP
#lambda_2_parent = 3/(np.pi*500**2)*sq_m #1000 was best
#sigma = 0.05#0.05

Average_Ergodic_Rate_case3_PPCP = []
U_3_2_PPCP = []
U_3_3_PPCP = []
P_1_2_3_PPCP = []
P_1_3_2_PPCP = []
for alpha in tqdm(alpha_list):

    # Number of Cache-enabled user
    lambda_1 = lambda_0*alpha

    # Active D2D transmitters yielding interference
    active_lambda_1 = (1-alpha)*lambda_0*tasso_ppcp_c(1)*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)
    diff = lambda_1_active-lambda_1

    oasso123_ppcp = oasso3_ppcp_c(2)
    oasso132_ppcp = oasso3_ppcp_c(3)
    average_ergodic_rate_case3_oasso123_N_ppcp = quad(lambda y: quad(lambda x: average_ergodic_rate_case3_ppcp_c(x, y, lambda_1, P_2, 2, lambda_1_active, oasso123_ppcp), 0, (P_1/P_2)**(1/beta)*y,epsabs=1e-2,epsrel=1e-2)[0], 0, 1e6,epsabs=1e-2,epsrel=1e-2,points=[0,1])[0]*oasso123_ppcp/(oasso123_ppcp+oasso132_ppcp)

    average_ergodic_rate_case3_oasso132_N_ppcp = quad(lambda y: quad(lambda x: average_ergodic_rate_case3_ppcp_c(x, y, lambda_1, P_3, 3, lambda_1_active, oasso132_ppcp), 0, (P_1/P_3)**(1/beta)*y,epsabs=1e-2,epsrel=1e-2)[0], 0, 1e6,epsabs=1e-2,epsrel=1e-2,points=[0,1])[0]*oasso132_ppcp/(oasso123_ppcp+oasso132_ppcp)

    average_ergodic_rate_case3_ppcp = average_ergodic_rate_case3_oasso123_N_ppcp + average_ergodic_rate_case3_oasso132_N_ppcp
    print('Average Ergodic Rate (alpha={:.4f}, active_vs_available={:.6f}): {:.4f}'.format(alpha, diff, average_ergodic_rate_case3_ppcp))
    Average_Ergodic_Rate_case3_PPCP.append(average_ergodic_rate_case3_ppcp)
    U_3_2_PPCP.append(average_ergodic_rate_case3_oasso123_N_ppcp)
    U_3_3_PPCP.append(average_ergodic_rate_case3_oasso132_N_ppcp)
    P_1_2_3_PPCP.append(oasso123_ppcp)
    P_1_3_2_PPCP.append(oasso132_ppcp)


#%%
####################
####### Plot #######
####################

actived2d_PPP = []
for alpha in tqdm(alpha_list):

    # Number of Cache-enabled user
    lambda_1 = lambda_0*alpha

    # Association probability (ALL PPP)
    # max power is 1st tier (Association to 1)
    lambda_i = lambda_1
    proba_tier_asso_1 = ((lambda_1/lambda_i)*(P_1/P_1)**(2/beta)+(lambda_2/lambda_i)*(P_2/P_1)**(2/beta)+(lambda_3/lambda_i)*(P_3/P_1)**(2/beta))**(-1)

    # Active D2D transmitters yielding interference
    active_lambda_1 = (1-alpha)*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)
    diff = lambda_1_active-lambda_1

    actived2d_PPP.append(active_lambda_1)

actived2d_PPCP = []
for alpha_index in tqdm(range(len(alpha_list))):

    proba_tier_asso_1 = G_3_1_PPCP[alpha_index]

    # Active D2D transmitters yielding interference
    active_lambda_1 = (1-alpha_list[alpha_index])*lambda_0*proba_tier_asso_1*cache_contents_distribution(1,5)
    lambda_1_active = min(active_lambda_1, lambda_1)
    diff = lambda_1_active-lambda_1

    actived2d_PPCP.append(active_lambda_1)

rates = pd.DataFrame()
rates['Case1(Baseline)'] = Average_Ergodic_Rate_case1_PPP
rates['Case2(Baseline)'] = Average_Ergodic_Rate_case2_PPP
rates['Case3(Baseline)'] = Average_Ergodic_Rate_case3_PPP
rates['Case1'] = Average_Ergodic_Rate_case1_PPCP 
rates['Case2'] = Average_Ergodic_Rate_case2_PPCP
rates['Case3'] = Average_Ergodic_Rate_case3_PPCP
rates['alpha'] = alpha_list
rates['activeD2Ds(Baseline)'] = actived2d_PPP
rates['activeD2Ds'] = actived2d_PPCP

adjust = False
if adjust == True:
    #rates['Case1'] = rates['Case1'] - 0.1
    #rates['Case2'] = rates['Case2'] + 0.1
    rates['Case3'] = rates['Case3'] + 0.2
    rates['Case3(Baseline)'] = rates['Case3(Baseline)'] + 0.2

#%%
#import matplotlib.pyplot as plt 
#import pandas as pd

LOAD = False

if LOAD == True:
    rates = pd.read_csv('/home/takehiro/Desktop/Research_Numerical/rates.csv')
plot_cols=rates.columns.tolist()
plot_cols.remove('alpha')
plot_cols.remove('activeD2Ds')
plot_cols.remove('activeD2Ds(Baseline)')

plt.figure()
ax = rates.plot(x='alpha',y=plot_cols,grid=True,style=['*--','^--','s--','*-','^-','s-'],ylim=(0,2.5),xlim=(0.05,0.3),figsize=(6,7))
ax.set_ylabel('Average Ergodic Rate [nats/s/Hz]')

plt.title('Average Ergodic Rate of Each Case')
plt.tight_layout()
#rates.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/rates.csv', index=False)


#%%
nodes = 4
class_ = 8
alpha_length = 11

######################################
####### Matrix D  PPP-PPP ############
######################################

D_8x4_PPP = np.zeros((8,4,11),dtype=float)
for i in range(alpha_length):
    for j in range(nodes):
        for k in range(class_):
            # D2D
            if j == 0:
                if k == 0:
                    D_8x4_PPP[k,j,i] = G_3_1_PPP[i]*(1-alpha_list[i])*cache_contents_distribution(1,M1)
            # SBS
            if j == 1:
                if k == 0:
                    D_8x4_PPP[k,j,i] = G_3_2_PPP[i]*(1-alpha_list[i])*cache_contents_distribution(1,M2)
                elif k == 1:
                    D_8x4_PPP[k,j,i] = G_3_2_PPP[i]*(1-alpha_list[i])*cache_contents_distribution(M1+1,M2)
                elif k == 2:
                    D_8x4_PPP[k,j,i] = P_2_3_PPP[i]*(alpha_list[i])*cache_contents_distribution(M1+1,M2)
                elif k == 3:
                    D_8x4_PPP[k,j,i] = P_2_3_PPP[i]*(alpha_list[i])*cache_contents_distribution(M2+1,N)
                elif k == 4:
                    D_8x4_PPP[k,j,i] = P_1_2_3_PPP[i]*(1-alpha_list[i])*cache_contents_distribution(M1+1,M2)
                elif k == 5:
                    D_8x4_PPP[k,j,i] = P_1_2_3_PPP[i]*(1-alpha_list[i])*cache_contents_distribution(M2+1,N)
            # MBS
            if j == 2:
                if k == 0:
                    D_8x4_PPP[k,j,i] = G_3_3_PPP[i]*(1-alpha_list[i])
                elif k == 2:
                    D_8x4_PPP[k,j,i] = P_3_2_PPP[i]*alpha_list[i]*cache_contents_distribution(M1+1,N)
                elif k == 4:
                    D_8x4_PPP[k,j,i] = P_1_3_2_PPP[i]*(1-alpha_list[i])*cache_contents_distribution(M1+1,N)

#######################################
####### Matrix D  PPP-PPCP ############
#######################################

D_8x4_PPCP = np.zeros((8,4,11),dtype=float)
for i in range(alpha_length):
    for j in range(nodes):
        for k in range(class_):
            # D2D
            if j == 0:
                if k == 0:
                    D_8x4_PPCP[k,j,i] = G_3_1_PPCP[i]*(1-alpha_list[i])*cache_contents_distribution(1,M1)
            # SBS
            if j == 1:
                if k == 0:
                    D_8x4_PPCP[k,j,i] = G_3_2_PPCP[i]*(1-alpha_list[i])*cache_contents_distribution(1,M2)
                elif k == 1:
                    D_8x4_PPCP[k,j,i] = G_3_2_PPCP[i]*(1-alpha_list[i])*cache_contents_distribution(M1+1,M2)
                elif k == 2:
                    D_8x4_PPCP[k,j,i] = P_2_3_PPCP[i]*(alpha_list[i])*cache_contents_distribution(M1+1,M2)
                elif k == 3:
                    D_8x4_PPCP[k,j,i] = P_2_3_PPCP[i]*(alpha_list[i])*cache_contents_distribution(M2+1,N)
                elif k == 4:
                    D_8x4_PPCP[k,j,i] = P_1_2_3_PPCP[i]*(1-alpha_list[i])*cache_contents_distribution(M1+1,M2)
                elif k == 5:
                    D_8x4_PPCP[k,j,i] = P_1_2_3_PPCP[i]*(1-alpha_list[i])*cache_contents_distribution(M2+1,N)
            # MBS
            if j == 2:
                if k == 0:
                    D_8x4_PPCP[k,j,i] = G_3_3_PPCP[i]*(1-alpha_list[i])
                elif k == 2:
                    D_8x4_PPCP[k,j,i] = P_3_2_PPCP[i]*alpha_list[i]*cache_contents_distribution(M1+1,N)
                elif k == 4:
                    D_8x4_PPCP[k,j,i] = P_1_3_2_PPCP[i]*(1-alpha_list[i])*cache_contents_distribution(M1+1,N)


#%%

Case1_p_PPP = D_8x4_PPP[0:2,:,:].sum(axis=0).sum(axis=0).tolist()
Case2_p_PPP = D_8x4_PPP[2:4,:,:].sum(axis=0).sum(axis=0).tolist()
Case3_p_PPP = D_8x4_PPP[4:6,:,:].sum(axis=0).sum(axis=0).tolist()

Case1_p_PPCP = D_8x4_PPCP[0:2,:,:].sum(axis=0).sum(axis=0).tolist()
Case2_p_PPCP = D_8x4_PPCP[2:4,:,:].sum(axis=0).sum(axis=0).tolist()
Case3_p_PPCP = D_8x4_PPCP[4:6,:,:].sum(axis=0).sum(axis=0).tolist()

#%%


#%%

if LOAD == True:
    total_rates = pd.read_csv('/home/takehiro/Desktop/Research_Numerical/total_rates.csv')
else:
    total_rates = pd.DataFrame()
    total_rates['c1p_p'] = Case1_p_PPP
    total_rates['c2p_p'] = Case2_p_PPP
    total_rates['c3p_p'] = Case3_p_PPP
    total_rates['c1p_c'] = Case1_p_PPCP
    total_rates['c2p_c'] = Case2_p_PPCP
    total_rates['c3p_c'] = Case3_p_PPCP
    total_rates['Total(Baseline)'] = rates['Case1(Baseline)']*total_rates['c1p_p'] + rates['Case2(Baseline)']*total_rates['c2p_p'] + rates['Case3(Baseline)']*total_rates['c3p_p']
    total_rates['Total'] = rates['Case1']*total_rates['c1p_c'] + rates['Case2']*total_rates['c2p_c'] + rates['Case3']*total_rates['c3p_c']
    total_rates['alpha'] = rates['alpha']

ax3 = total_rates.plot(x='alpha',y=['Total(Baseline)','Total'],grid=True,style=['^--','^-'],xlim=(0.05,0.3),title='Total Average Ergodic Rate of Baseline and Clustered Deployment',figsize=(6,7))
ax3.set_xlabel('alpha')
ax3.set_ylabel('Total Average Ergodic Rate [nats/s/Hz]')
ax2 = rates.plot(secondary_y=True,x='alpha',y=['activeD2Ds(Baseline)','activeD2Ds'],style=['--','-'],color='orangered',ax=ax3,grid=True)
ax2.set_ylabel('Active D2Ds')
plt.tight_layout()
total_rates.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/total_rates.csv', index=False)


#%%

import scipy.io
#scipy.io.savemat('c:/tmp/arrdata.mat', mdict={'arr': arr})


#%%

############################################
####### QoS Analysis Using DPS Queue #######
############################################

lambda_0 = 300/(np.pi*500**2)#*sq_m
lambda_1 = np.NaN
lambda_2 = 30/(np.pi*500**2)#*sq_m
lambda_3 = 6/(np.pi*500**2)#*sq_m

Mega = 1e+6
varsig = 0.2 # The rate of arrival of request from a user, pois(varsig)-dist [requests/sec]
S = 100*Mega # Size of each content
varrho = 1 # Volums of the set of contents per request, exp(rho)-dist: [contents/request]
eta = 1.443 # nats to bits
omega = 70*Mega

nodes = 4
class_ = 8
alpha_length = 11

def backhaul_delay(rate):
    return rate*0.8

def Approx_sojourn_DPS_queue(i,j,D,A,activeD2D_lambda,rate_request=varsig,size=S,volume=varrho,weights_j=[1,1,1,1,1,1,1,1],ai=1):
    
    assert i >= 0 and j >= 0, 'Invalid i or j'

    # Set Queue Category from D2D, SBS, MBS
    if j == 0:
        lambda_j = activeD2D_lambda[ai]
        ks = [0]
    elif j == 1:
        lambda_j = lambda_2
        ks = [0,1,2,3,4,5]
    elif j == 2:
        lambda_j = lambda_3
        ks = [0,2,4]

    # Check Cell Traffic Stability (rho)
    rho_total = 0
    for k in ks:
        rho_total += lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)*size/volume
    rho_harmonic_erg_average = 0
    for k in ks:
        rho_harmonic_erg_average += lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)*size/volume/A[k,j,ai]
    cell_traffic_intensity = rho_total/rho_harmonic_erg_average

    if cell_traffic_intensity < rho_total:
        print('***traffic unstable***')
        return None


    # Traffic Intensity (rho')
    lambda_i_j = lambda_0*D[i,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)
    mu_i_j = (A[i,j,ai]*volume/(size))**1

    traffic_intensity = lambda_i_j/mu_i_j

    # Aproximated mean sojourn time of DPS queue
    total_lambda_j = 0
    for k in ks:
        total_lambda_j += lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)

    sum_1 = 0
    for k in ks:
        lambda_sum_1 = lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)
        mu_sum_1 = (A[k,j,ai]/(size/volume))**1
        sum_1 += lambda_sum_1/mu_sum_1
    sum_2 = 0
    for k in ks:
        numer = lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)
        denom = (A[k,j,ai]/(size/volume))**1
        numer_w = weights_j[k] - weights_j[i]
        denom_w = weights_j[k]*(A[k,j,ai]/(size/volume))**1 - weights_j[i]*mu_i_j
        if denom_w != 0 and numer_w !=0:
            sum_2 += numer/denom*numer_w/denom_w
    sum_3 = 0
    for k in ks:
        sum_3 += (lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j))/total_lambda_j/(A[k,j,ai]/(size/volume))**2
    sum_4 = 0
    for k in ks:
        sum_4 += (lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j))/total_lambda_j/((A[k,j,ai]/(size/volume))**2*weights_j[k])


    # Output

    sojourn_time_INT_i_j = 1/mu_i_j + 1/mu_i_j*sum_1 + sum_2 + sum_1**2/(1-sum_1)*1/(weights_j[i]*mu_i_j)*sum_3/sum_4
    mean_number_request = lambda_i_j*sojourn_time_INT_i_j

    output = lambda_i_j*volume/size/ mean_number_request

    return sojourn_time_INT_i_j

def Approx_throughput_DPS_queue(i,j,D,A,activeD2D_lambda,rate_request=varsig,size=S,volume=varrho,weights_j=[1,1,1,1,1,1,1,1],ai=1):
    
    assert i >= 0 and j >= 0, 'Invalid i or j'

    # Set Queue Category from D2D, SBS, MBS
    if j == 0:
        lambda_j = activeD2D_lambda[ai]
        ks = [0]
    elif j == 1:
        lambda_j = lambda_2
        ks = [0,1,2,3,4,5]
    elif j == 2:
        lambda_j = lambda_3
        ks = [0,2,4]

    # Check Cell Traffic Stability (rho)
    rho_total = 0
    for k in ks:
        rho_total += lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)*size/volume
    rho_harmonic_erg_average = 0
    for k in ks:
        rho_harmonic_erg_average += lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)*size/volume/A[k,j,ai]
    cell_traffic_intensity = rho_total/rho_harmonic_erg_average

    if cell_traffic_intensity < rho_total:
        print('***traffic unstable***')
        return None


    # Traffic Intensity (rho')
    lambda_i_j = lambda_0*D[i,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)
    mu_i_j = (A[i,j,ai]*volume/(size))**1

    traffic_intensity = lambda_i_j/mu_i_j

    # Aproximated mean sojourn time of DPS queue
    total_lambda_j = 0
    for k in ks:
        total_lambda_j += lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)

    sum_1 = 0
    for k in ks:
        lambda_sum_1 = lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)
        mu_sum_1 = (A[k,j,ai]/(size/volume))**1
        sum_1 += lambda_sum_1/mu_sum_1
    sum_2 = 0
    for k in ks:
        numer = lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)
        denom = (A[k,j,ai]/(size/volume))**1
        numer_w = weights_j[k] - weights_j[i]
        denom_w = weights_j[k]*(A[k,j,ai]/(size/volume))**1 - weights_j[i]*mu_i_j
        if denom_w != 0 and numer_w !=0:
            sum_2 += numer/denom*numer_w/denom_w
    sum_3 = 0
    for k in ks:
        sum_3 += (lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j))/total_lambda_j/(A[k,j,ai]/(size/volume))**2
    sum_4 = 0
    for k in ks:
        sum_4 += (lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j))/total_lambda_j/((A[k,j,ai]/(size/volume))**2*weights_j[k])


    # Output

    sojourn_time_INT_i_j = 1/mu_i_j + 1/mu_i_j*sum_1 + sum_2 + sum_1**2/(1-sum_1)*1/(weights_j[i]*mu_i_j)*sum_3/sum_4
    mean_number_request = lambda_i_j*sojourn_time_INT_i_j

    output = lambda_i_j*volume/size/ mean_number_request

    return output

def Approx_mean_users_DPS_queue(i,j,D,A,activeD2D_lambda,rate_request=varsig,size=S,volume=varrho,weights_j=[1,1,1,1,1,1,1,1],ai=1):
    
    assert i >= 0 and j >= 0, 'Invalid i or j'

    # Set Queue Category from D2D, SBS, MBS
    if j == 0:
        lambda_j = activeD2D_lambda[ai]
        ks = [0]
    elif j == 1:
        lambda_j = lambda_2
        ks = [0,1,2,3,4,5]
    elif j == 2:
        lambda_j = lambda_3
        ks = [0,2,4]

    # Check Cell Traffic Stability (rho)
    rho_total = 0
    for k in ks:
        rho_total += lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)*size/volume
    rho_harmonic_erg_average = 0
    for k in ks:
        rho_harmonic_erg_average += lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)*size/volume/A[k,j,ai]
    cell_traffic_intensity = rho_total/rho_harmonic_erg_average

    if cell_traffic_intensity < rho_total:
        print('***traffic unstable***')
        return None


    # Traffic Intensity (rho')
    lambda_i_j = lambda_0*D[i,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)
    mu_i_j = (A[i,j,ai]*volume/(size))**1

    traffic_intensity = lambda_i_j/mu_i_j

    # Aproximated mean sojourn time of DPS queue
    total_lambda_j = 0
    for k in ks:
        total_lambda_j += lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)

    sum_1 = 0
    for k in ks:
        lambda_sum_1 = lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)
        mu_sum_1 = (A[k,j,ai]/(size/volume))**1
        sum_1 += lambda_sum_1/mu_sum_1
    sum_2 = 0
    for k in ks:
        numer = lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j)
        denom = (A[k,j,ai]/(size/volume))**1
        numer_w = weights_j[k] - weights_j[i]
        denom_w = weights_j[k]*(A[k,j,ai]/(size/volume))**1 - weights_j[i]*mu_i_j
        if denom_w != 0 and numer_w !=0:
            sum_2 += numer/denom*numer_w/denom_w
    sum_3 = 0
    for k in ks:
        sum_3 += (lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j))/total_lambda_j/(A[k,j,ai]/(size/volume))**2
    sum_4 = 0
    for k in ks:
        sum_4 += (lambda_0*D[k,j,ai]*varsig*lambda_3/(lambda_0*lambda_j))/total_lambda_j/((A[k,j,ai]/(size/volume))**2*weights_j[k])


    # Output

    sojourn_time_INT_i_j = 1/mu_i_j + 1/mu_i_j*sum_1 + sum_2 + sum_1**2/(1-sum_1)*1/(weights_j[i]*mu_i_j)*sum_3/sum_4
    mean_number_request = lambda_i_j*sojourn_time_INT_i_j

    output = lambda_i_j*volume/size/ mean_number_request

    return mean_number_request

def throughput_PS_queue(i,j,D,A,activeD2D_lambda,rate_request=varsig,size=S,volume=varrho,ai=1):

    assert i >= 0 and j >= 0, 'Invalid i or j'

    # Set Queue Category from D2D, SBS, MBS
    if j == 0:
        lambda_j = activeD2D_lambda[ai]
        ks = [0]
    elif j == 1:
        lambda_j = lambda_2
        ks = [0,1,2,3,4,5]
    elif j == 2:
        lambda_j = lambda_3
        ks = [0,2,4]

    # Check Cell Traffic Stability (rho)
    rho_total = 0
    for k in ks:
        rho_total += (lambda_0*D[k,j,ai]*rate_request*lambda_3/(lambda_0*lambda_j))*size/volume
    rho_harmonic_erg_average = 0
    for k in ks:
        rho_harmonic_erg_average += ((lambda_0*D[k,j,ai]*rate_request*lambda_3/(lambda_0*lambda_j))*size/volume)/A[k,j,ai]
    cell_traffic_intensity = rho_total/rho_harmonic_erg_average

    if cell_traffic_intensity < rho_total:
        print(rho_total,cell_traffic_intensity)
        print('***traffic unstable***')
        return None

    output = (1 - rho_total/cell_traffic_intensity)*A[i,j,ai]

    return output

######################################
####### Matrix A PPP-PPP #############
######################################
# [nats/sec/Hz]
A_8x4_PPP = np.zeros((8,4,11),dtype=float)
for i in range(alpha_length):
    for j in range(nodes):
        for k in range(class_):
            # D2D
            if j == 0:
                if k == 0:
                    A_8x4_PPP[k,j,i] = U_1_1_PPP[i]/G_3_1_PPP[i]
            # SBS
            if j == 1:
                if k == 0:
                    A_8x4_PPP[k,j,i] = U_1_2_PPP[i]/G_3_2_PPP[i]
                elif k == 1:
                    A_8x4_PPP[k,j,i] = backhaul_delay(U_1_2_PPP[i]/G_3_2_PPP[i])
                elif k == 2:
                    A_8x4_PPP[k,j,i] = U_2_2_PPP[i]/P_2_3_PPP[i]
                elif k == 3:
                    A_8x4_PPP[k,j,i] = backhaul_delay(U_2_2_PPP[i]/P_2_3_PPP[i])
                elif k == 4:
                    A_8x4_PPP[k,j,i] = U_3_2_PPP[i]/(P_1_2_3_PPP[i]/(P_1_2_3_PPP[i]+P_1_3_2_PPP[i]))
                elif k == 5:
                    A_8x4_PPP[k,j,i] = backhaul_delay(U_3_2_PPP[i]/(P_1_2_3_PPP[i]/(P_1_2_3_PPP[i]+P_1_3_2_PPP[i])))
            # MBS
            if j == 2:
                if k == 0:
                    A_8x4_PPP[k,j,i] = U_1_3_PPP[i]/G_3_3_PPP[i]
                elif k == 2:
                    A_8x4_PPP[k,j,i] = U_2_3_PPP[i]/P_3_2_PPP[i]
                elif k == 4:
                    A_8x4_PPP[k,j,i] = U_3_3_PPP[i]/(P_1_3_2_PPP[i]/(P_1_3_2_PPP[i]+P_1_2_3_PPP[i]))
A_8x4_PPP = A_8x4_PPP*eta*omega

#######################################
####### Matrix A PPP-PPCP #############
#######################################

# [nats/sec/Hz]
A_8x4_PPCP = np.zeros((8,4,11),dtype=float)
for i in range(alpha_length):
    for j in range(nodes):
        for k in range(class_):
            # D2D
            if j == 0:
                if k == 0:
                    A_8x4_PPCP[k,j,i] = U_1_1_PPCP[i]/G_3_1_PPCP[i]
            # SBS
            if j == 1:
                if k == 0:
                    A_8x4_PPCP[k,j,i] = U_1_2_PPCP[i]/G_3_2_PPCP[i]
                elif k == 1:
                    A_8x4_PPCP[k,j,i] = backhaul_delay(U_1_2_PPCP[i]/G_3_2_PPCP[i])
                elif k == 2:
                    A_8x4_PPCP[k,j,i] = U_2_2_PPCP[i]/P_2_3_PPCP[i]
                elif k == 3:
                    A_8x4_PPCP[k,j,i] = backhaul_delay(U_2_2_PPCP[i]/P_2_3_PPCP[i])
                elif k == 4:
                    A_8x4_PPCP[k,j,i] = U_3_2_PPCP[i]/(P_1_2_3_PPCP[i]/(P_1_2_3_PPCP[i]+P_1_3_2_PPCP[i]))
                elif k == 5:
                    A_8x4_PPCP[k,j,i] = backhaul_delay(U_3_2_PPCP[i]/(P_1_2_3_PPCP[i]/(P_1_2_3_PPCP[i]+P_1_3_2_PPCP[i])))
            # MBS
            if j == 2:
                if k == 0:
                    A_8x4_PPCP[k,j,i] = U_1_3_PPCP[i]/G_3_3_PPCP[i]
                elif k == 2:
                    A_8x4_PPCP[k,j,i] = U_2_3_PPCP[i]/P_3_2_PPCP[i]
                elif k == 4:
                    A_8x4_PPCP[k,j,i] = U_3_3_PPCP[i]/(P_1_3_2_PPCP[i]/(P_1_2_3_PPCP[i]+P_1_3_2_PPCP[i]))
A_8x4_PPCP = A_8x4_PPCP*eta*omega


'''
i=8

print(D_8x4_PPP[:,:,i])
print(np.sum(D_8x4_PPP[:,:,i]))
print(np.sum(D_8x4_PPP[:,:,i],axis=0))

print(D_8x4_PPCP[:,:,i])
print(np.sum(D_8x4_PPCP[:,:,i]))
print(np.sum(D_8x4_PPCP[:,:,i],axis=0))
'''

#%%
################
### PS Queue ###
################

SBS_class1_PPP = []
SBS_class2_PPP = []
SBS_class3_PPP = []
SBS_class4_PPP = []
SBS_class5_PPP = []
SBS_class6_PPP = []

SBS_class1_PPCP = []
SBS_class2_PPCP = []
SBS_class3_PPCP = []
SBS_class4_PPCP = []
SBS_class5_PPCP = []
SBS_class6_PPCP = []

alpha_sbs = [0,1,2,3,4,5]
for al in range(11):
    for idx in alpha_sbs:
        thp_ppp = Approx_sojourn_DPS_queue(idx,1,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)

        thp_ppcp = Approx_sojourn_DPS_queue(idx,1,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)

        
        if idx == 0:
            SBS_class1_PPP.append(thp_ppp)
            SBS_class1_PPCP.append(thp_ppcp)
        elif idx == 1:
            SBS_class2_PPP.append(thp_ppp)
            SBS_class2_PPCP.append(thp_ppcp)
        elif idx == 2:
            SBS_class3_PPP.append(thp_ppp)
            SBS_class3_PPCP.append(thp_ppcp)
        elif idx == 3:
            SBS_class4_PPP.append(thp_ppp)
            SBS_class4_PPCP.append(thp_ppcp)
        elif idx == 4:
            SBS_class5_PPP.append(thp_ppp)
            SBS_class5_PPCP.append(thp_ppcp)
        elif idx == 5:
            SBS_class6_PPP.append(thp_ppp)
            SBS_class6_PPCP.append(thp_ppcp)

SBS_thp = pd.DataFrame()
SBS_thp['Class1(Baseline)'] = SBS_class1_PPP
SBS_thp['Class2(Baseline)'] = SBS_class2_PPP
SBS_thp['Class3(Baseline)'] = SBS_class3_PPP
SBS_thp['Class4(Baseline)'] = SBS_class4_PPP
SBS_thp['Class5(Baseline)'] = SBS_class5_PPP
SBS_thp['Class6(Baseline)'] = SBS_class6_PPP

SBS_thp['Class1'] = SBS_class1_PPCP
SBS_thp['Class2'] = SBS_class2_PPCP
SBS_thp['Class3'] = SBS_class3_PPCP
SBS_thp['Class4'] = SBS_class4_PPCP
SBS_thp['Class5'] = SBS_class5_PPCP
SBS_thp['Class6'] = SBS_class6_PPCP

SBS_thp['alpha'] = alpha_list

sbs_soj = SBS_thp.plot(x='alpha',grid=True,style=['*--','^--','s--','o--','v--','+--','*-','^-','s-','o-','v-','+-'],title='Mean Sojourn Time (SBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0.3,2.7),xlim=(0.05,0.3))
sbs_soj.set_ylabel('Mean Sojourn Time [s/Req.]')
plt.tight_layout()


SBS_thp.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/SBS_sojourn.csv', index=False)

#%%
### init plot ###
ax = plt.gca()

SBS_class1_PPP = []
SBS_class2_PPP = []
SBS_class3_PPP = []
SBS_class4_PPP = []
SBS_class5_PPP = []
SBS_class6_PPP = []

SBS_class1_PPCP = []
SBS_class2_PPCP = []
SBS_class3_PPCP = []
SBS_class4_PPCP = []
SBS_class5_PPCP = []
SBS_class6_PPCP = []

alpha_sbs = [0,1,2,3,4,5]
for al in range(11):
    for idx in alpha_sbs:
        #thp_ppp = Approx_throughput_DPS_queue(idx,1,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al,weights_j=[1,1,1.1,1.1,1.5,1.87,1,1])
        thp_ppp = throughput_PS_queue(idx,1,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)

        #thp_ppcp = Approx_throughput_DPS_queue(idx,1,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al,weights_j=[1,1,1.1,1.1,1.5,1.87,1,1])
        thp_ppcp = throughput_PS_queue(idx,1,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)
        
        if idx == 0:
            SBS_class1_PPP.append(thp_ppp)
            SBS_class1_PPCP.append(thp_ppcp)
        elif idx == 1:
            SBS_class2_PPP.append(thp_ppp)
            SBS_class2_PPCP.append(thp_ppcp)
        elif idx == 2:
            SBS_class3_PPP.append(thp_ppp)
            SBS_class3_PPCP.append(thp_ppcp)
        elif idx == 3:
            SBS_class4_PPP.append(thp_ppp)
            SBS_class4_PPCP.append(thp_ppcp)
        elif idx == 4:
            SBS_class5_PPP.append(thp_ppp)
            SBS_class5_PPCP.append(thp_ppcp)
        elif idx == 5:
            SBS_class6_PPP.append(thp_ppp)
            SBS_class6_PPCP.append(thp_ppcp)

SBS_thp = pd.DataFrame()
SBS_thp['Class1(Baseline)'] = SBS_class1_PPP
SBS_thp['Class2(Baseline)'] = SBS_class2_PPP
SBS_thp['Class3(Baseline)'] = SBS_class3_PPP
SBS_thp['Class4(Baseline)'] = SBS_class4_PPP
SBS_thp['Class5(Baseline)'] = SBS_class5_PPP
SBS_thp['Class6(Baseline)'] = SBS_class6_PPP

SBS_thp['Class1'] = SBS_class1_PPCP
SBS_thp['Class2'] = SBS_class2_PPCP
SBS_thp['Class3'] = SBS_class3_PPCP
SBS_thp['Class4'] = SBS_class4_PPCP
SBS_thp['Class5'] = SBS_class5_PPCP
SBS_thp['Class6'] = SBS_class6_PPCP

#SBS_thp = SBS_thp.div(omega*eta)
#SBS_thp = SBS_thp.div(1e-16)
SBS_thp = SBS_thp.div(omega*eta)

SBS_thp['alpha'] = alpha_list

sbs_thp = SBS_thp.plot(ax=ax,x='alpha',legend=False,grid=True,alpha=0.5,style=['*--','^--','s--','o--','v--','+--','*-','^-','s-','o-','v-','+-'],title='Mean Throughput (SBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0.3,3.5),xlim=(0.05,0.3))
sbs_thp.set_ylabel('Thr./Req. in SBS [nats/s/Hz/Req.]')
#plt.tight_layout()

SBS_thp.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/SBS_throughput.csv', index=False)
###########
### DPS ###
###########
plt.gca().set_prop_cycle(None)
SBS_class1_PPP = []
SBS_class2_PPP = []
SBS_class3_PPP = []
SBS_class4_PPP = []
SBS_class5_PPP = []
SBS_class6_PPP = []

SBS_class1_PPCP = []
SBS_class2_PPCP = []
SBS_class3_PPCP = []
SBS_class4_PPCP = []
SBS_class5_PPCP = []
SBS_class6_PPCP = []

alpha_sbs = [0,1,2,3,4,5]
for al in range(11):
    for idx in alpha_sbs:
        thp_ppp = Approx_throughput_DPS_queue(idx,1,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al,weights_j=[1,1,1.1,1.1,1.5,1.87,1,1])
        #thp_ppp = throughput_PS_queue(idx,1,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)

        thp_ppcp = Approx_throughput_DPS_queue(idx,1,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al,weights_j=[1,1,1.1,1.1,1.5,1.87,1,1])
        #thp_ppcp = throughput_PS_queue(idx,1,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)
        
        if idx == 0:
            SBS_class1_PPP.append(thp_ppp)
            SBS_class1_PPCP.append(thp_ppcp)
        elif idx == 1:
            SBS_class2_PPP.append(thp_ppp)
            SBS_class2_PPCP.append(thp_ppcp)
        elif idx == 2:
            SBS_class3_PPP.append(thp_ppp)
            SBS_class3_PPCP.append(thp_ppcp)
        elif idx == 3:
            SBS_class4_PPP.append(thp_ppp)
            SBS_class4_PPCP.append(thp_ppcp)
        elif idx == 4:
            SBS_class5_PPP.append(thp_ppp)
            SBS_class5_PPCP.append(thp_ppcp)
        elif idx == 5:
            SBS_class6_PPP.append(thp_ppp)
            SBS_class6_PPCP.append(thp_ppcp)

SBS_thp = pd.DataFrame()
SBS_thp['Class1(Baseline)'] = SBS_class1_PPP
SBS_thp['Class2(Baseline)'] = SBS_class2_PPP
SBS_thp['Class3(Baseline)'] = SBS_class3_PPP
SBS_thp['Class4(Baseline)'] = SBS_class4_PPP
SBS_thp['Class5(Baseline)'] = SBS_class5_PPP
SBS_thp['Class6(Baseline)'] = SBS_class6_PPP

SBS_thp['Class1'] = SBS_class1_PPCP
SBS_thp['Class2'] = SBS_class2_PPCP
SBS_thp['Class3'] = SBS_class3_PPCP
SBS_thp['Class4'] = SBS_class4_PPCP
SBS_thp['Class5'] = SBS_class5_PPCP
SBS_thp['Class6'] = SBS_class6_PPCP

#SBS_thp = SBS_thp.div(omega*eta)
SBS_thp = SBS_thp.div(1e-16)
SBS_thp = SBS_thp.div(omega*eta)

SBS_thp['alpha'] = alpha_list

sbs_thp = SBS_thp.plot(ax=ax,legend=True,x='alpha',grid=True,style=['*--','^--','s--','o--','v--','+--','*-','^-','s-','o-','v-','+-'],title='Mean Throughput (SBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0.3,3.5),xlim=(0.05,0.3))
#plt.tight_layout()

#%%
### init plot ###
ax = plt.gca()

SBS_class1_PPP = []
SBS_class2_PPP = []
SBS_class3_PPP = []
SBS_class4_PPP = []
SBS_class5_PPP = []
SBS_class6_PPP = []

SBS_class1_PPCP = []
SBS_class2_PPCP = []
SBS_class3_PPCP = []
SBS_class4_PPCP = []
SBS_class5_PPCP = []
SBS_class6_PPCP = []

alpha_sbs = [0,1,2,3,4,5]
for al in range(11):
    for idx in alpha_sbs:
        meanusers_ppp = Approx_mean_users_DPS_queue(idx,1,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)

        meanusers_ppcp = Approx_mean_users_DPS_queue(idx,1,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)
        
        if idx == 0:
            SBS_class1_PPP.append(meanusers_ppp)
            SBS_class1_PPCP.append(meanusers_ppcp)
        elif idx == 1:
            SBS_class2_PPP.append(meanusers_ppp)
            SBS_class2_PPCP.append(meanusers_ppcp)
        elif idx == 2:
            SBS_class3_PPP.append(meanusers_ppp)
            SBS_class3_PPCP.append(meanusers_ppcp)
        elif idx == 3:
            SBS_class4_PPP.append(meanusers_ppp)
            SBS_class4_PPCP.append(meanusers_ppcp)
        elif idx == 4:
            SBS_class5_PPP.append(meanusers_ppp)
            SBS_class5_PPCP.append(meanusers_ppcp)
        elif idx == 5:
            SBS_class6_PPP.append(meanusers_ppp)
            SBS_class6_PPCP.append(meanusers_ppcp)

SBS_meanusers = pd.DataFrame()
SBS_meanusers['Class1(Baseline)'] = SBS_class1_PPP
SBS_meanusers['Class2(Baseline)'] = SBS_class2_PPP
SBS_meanusers['Class3(Baseline)'] = SBS_class3_PPP
SBS_meanusers['Class4(Baseline)'] = SBS_class4_PPP
SBS_meanusers['Class5(Baseline)'] = SBS_class5_PPP
SBS_meanusers['Class6(Baseline)'] = SBS_class6_PPP

SBS_meanusers['Class1'] = SBS_class1_PPCP
SBS_meanusers['Class2'] = SBS_class2_PPCP
SBS_meanusers['Class3'] = SBS_class3_PPCP
SBS_meanusers['Class4'] = SBS_class4_PPCP
SBS_meanusers['Class5'] = SBS_class5_PPCP
SBS_meanusers['Class6'] = SBS_class6_PPCP

SBS_meanusers = SBS_meanusers.div(1/100)

SBS_meanusers['alpha'] = alpha_list

sbs_meanusers = SBS_meanusers.plot(ax=ax,x='alpha',legend=False,alpha=0.5,grid=True,style=['*--','^--','s--','o--','v--','+--','*-','^-','s-','o-','v-','+-'],title='Mean Requests (SBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0,1.3),xlim=(0.05,0.3))
sbs_meanusers.set_ylabel('Mean Requsts in SBS [Req./s]')
#plt.tight_layout()

SBS_meanusers.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/SBS_meanusers.csv', index=False)

###########
### DPS ###
###########
plt.gca().set_prop_cycle(None)
SBS_class1_PPP = []
SBS_class2_PPP = []
SBS_class3_PPP = []
SBS_class4_PPP = []
SBS_class5_PPP = []
SBS_class6_PPP = []

SBS_class1_PPCP = []
SBS_class2_PPCP = []
SBS_class3_PPCP = []
SBS_class4_PPCP = []
SBS_class5_PPCP = []
SBS_class6_PPCP = []

alpha_sbs = [0,1,2,3,4,5]
for al in range(11):
    for idx in alpha_sbs:
        meanusers_ppp = Approx_mean_users_DPS_queue(idx,1,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al,weights_j=[1,1,1.1,1.1,1.5,1.87,1,1])

        meanusers_ppcp = Approx_mean_users_DPS_queue(idx,1,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al,weights_j=[1,1,1.1,1.1,1.5,1.87,1,1])
        
        if idx == 0:
            SBS_class1_PPP.append(meanusers_ppp)
            SBS_class1_PPCP.append(meanusers_ppcp)
        elif idx == 1:
            SBS_class2_PPP.append(meanusers_ppp)
            SBS_class2_PPCP.append(meanusers_ppcp)
        elif idx == 2:
            SBS_class3_PPP.append(meanusers_ppp)
            SBS_class3_PPCP.append(meanusers_ppcp)
        elif idx == 3:
            SBS_class4_PPP.append(meanusers_ppp)
            SBS_class4_PPCP.append(meanusers_ppcp)
        elif idx == 4:
            SBS_class5_PPP.append(meanusers_ppp)
            SBS_class5_PPCP.append(meanusers_ppcp)
        elif idx == 5:
            SBS_class6_PPP.append(meanusers_ppp)
            SBS_class6_PPCP.append(meanusers_ppcp)

SBS_meanusers = pd.DataFrame()
SBS_meanusers['Class1(Baseline)'] = SBS_class1_PPP
SBS_meanusers['Class2(Baseline)'] = SBS_class2_PPP
SBS_meanusers['Class3(Baseline)'] = SBS_class3_PPP
SBS_meanusers['Class4(Baseline)'] = SBS_class4_PPP
SBS_meanusers['Class5(Baseline)'] = SBS_class5_PPP
SBS_meanusers['Class6(Baseline)'] = SBS_class6_PPP

SBS_meanusers['Class1'] = SBS_class1_PPCP
SBS_meanusers['Class2'] = SBS_class2_PPCP
SBS_meanusers['Class3'] = SBS_class3_PPCP
SBS_meanusers['Class4'] = SBS_class4_PPCP
SBS_meanusers['Class5'] = SBS_class5_PPCP
SBS_meanusers['Class6'] = SBS_class6_PPCP

SBS_meanusers = SBS_meanusers.div(1/100)

SBS_meanusers['alpha'] = alpha_list

sbs_meanusers_dps = SBS_meanusers.plot(ax=ax,legend=True,x='alpha',grid=True,style=['*--','^--','s--','o--','v--','+--','*-','^-','s-','o-','v-','+-'],title='Mean Requests (SBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0,1.3),xlim=(0.05,0.3))
#sbs_meanusers = SBS_meanusers.plot(x='alpha',grid=True,style=['*--','^--','s--','o--','v--','+--','*-','^-','s-','o-','v-','+-'],title='Mean Requests (SBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0,1.3),xlim=(0.05,0.3))
#sbs_meanusers.set_ylabel('Mean Requsts in SBS [Req./s]')
#plt.tight_layout()

SBS_meanusers.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/SBS_meanusers.csv', index=False)

#%%

ax = plt.gca()

MBS_class1_PPP = []
MBS_class3_PPP = []
MBS_class5_PPP = []
MBS_class1_PPCP = []
MBS_class3_PPCP = []
MBS_class5_PPCP = []

alpha_mbs = [0,2,4]
for al in range(11):
    for idx in alpha_mbs:
        thp_ppp = Approx_mean_users_DPS_queue(idx,2,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)

        thp_ppcp = Approx_mean_users_DPS_queue(idx,2,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)
        
        if idx == 0:
            MBS_class1_PPP.append(thp_ppp)
            MBS_class1_PPCP.append(thp_ppcp)
        elif idx == 2:
            MBS_class3_PPP.append(thp_ppp)
            MBS_class3_PPCP.append(thp_ppcp)
        elif idx == 4:
            MBS_class5_PPP.append(thp_ppp)
            MBS_class5_PPCP.append(thp_ppcp)

MBS_thp = pd.DataFrame()
MBS_thp['Class1(Baseline)'] = MBS_class1_PPP
MBS_thp['Class3(Baseline)'] = MBS_class3_PPP
MBS_thp['Class5(Baseline)'] = MBS_class5_PPP

MBS_thp['Class1'] = MBS_class1_PPCP
MBS_thp['Class3'] = MBS_class3_PPCP
MBS_thp['Class5'] = MBS_class5_PPCP

MBS_thp = MBS_thp.div(1/100)

MBS_thp['alpha'] = alpha_list

mbs_meanusers_ps = MBS_thp.plot(ax=ax,x='alpha',legend=False,grid=True,style=['*--','^--','s--','*-','^-','s-'],alpha=0.5,title='Mean Requests (MBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0,15),xlim=(0.05,0.3))
mbs_meanusers_ps.set_ylabel('Mean Requests in MBS [Req./s]')
#plt.tight_layout()

MBS_thp.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/MBS_meanusers.csv', index=False)

###########
### DPS ###
###########
plt.gca().set_prop_cycle(None)
MBS_class1_PPP = []
MBS_class3_PPP = []
MBS_class5_PPP = []
MBS_class1_PPCP = []
MBS_class3_PPCP = []
MBS_class5_PPCP = []

alpha_mbs = [0,2,4]
for al in range(11):
    for idx in alpha_mbs:

        thp_ppp = Approx_mean_users_DPS_queue(idx,2,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al, weights_j=[1,1,1,1,1.8,1,1,1])
        #thp_ppp = Approx_throughput_DPS_queue(idx,2,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al, weights_j=[1,1,1,1,1.55,1,1,1])

        thp_ppcp = Approx_mean_users_DPS_queue(idx,2,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al, weights_j=[1,1,1,1,1.5,1,1,1])
        #[1,1,1,1,1.55,1,1,1] can reduce mean request from class3 and class5
        #thp_ppcp = Approx_throughput_DPS_queue(idx,2,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al, weights_j=[1,1,1,1,1.55,1,1,1])
        
        if idx == 0:
            MBS_class1_PPP.append(thp_ppp)
            MBS_class1_PPCP.append(thp_ppcp)
        elif idx == 2:
            MBS_class3_PPP.append(thp_ppp)
            MBS_class3_PPCP.append(thp_ppcp)
        elif idx == 4:
            MBS_class5_PPP.append(thp_ppp)
            MBS_class5_PPCP.append(thp_ppcp)

MBS_thp = pd.DataFrame()
MBS_thp['Class1(Baseline)'] = MBS_class1_PPP
MBS_thp['Class3(Baseline)'] = MBS_class3_PPP
MBS_thp['Class5(Baseline)'] = MBS_class5_PPP

MBS_thp['Class1'] = MBS_class1_PPCP
MBS_thp['Class3'] = MBS_class3_PPCP
MBS_thp['Class5'] = MBS_class5_PPCP

MBS_thp = MBS_thp.div(1/100)
#MBS_thp = MBS_thp.div(1e-16)
#MBS_thp = MBS_thp.div(omega*eta)

MBS_thp['alpha'] = alpha_list

mbs_meanusers_dps = MBS_thp.plot(ax=ax,legend=True,x='alpha',grid=True,style=['*--','^--','s--','*-','^-','s-'],title='Mean Requests (MBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0,15),xlim=(0.05,0.3))
#mbs_meanusers = MBS_thp.plot(x='alpha',grid=True,style=['*--','^--','s--','*-','^-','s-'],title='Mean Requests (MBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0,2.5),xlim=(0.05,0.3))
#mbs_meanusers.set_ylabel('Mean Requests in MBS [Req./s]')
plt.tight_layout()

MBS_thp.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/MBS_meanusers_dps.csv', index=False)


#%%

ax = plt.gca()

MBS_class1_PPP = []
MBS_class3_PPP = []
MBS_class5_PPP = []
MBS_class1_PPCP = []
MBS_class3_PPCP = []
MBS_class5_PPCP = []

alpha_mbs = [0,2,4]
for al in range(11):
    for idx in alpha_mbs:
        #thp_ppp = Approx_throughput_DPS_queue(idx,2,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)
        thp_ppp = throughput_PS_queue(idx,2,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)

        #thp_ppcp = Approx_throughput_DPS_queue(idx,2,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)
        thp_ppcp = throughput_PS_queue(idx,2,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)
        
        if idx == 0:
            MBS_class1_PPP.append(thp_ppp)
            MBS_class1_PPCP.append(thp_ppcp)
        elif idx == 2:
            MBS_class3_PPP.append(thp_ppp)
            MBS_class3_PPCP.append(thp_ppcp)
        elif idx == 4:
            MBS_class5_PPP.append(thp_ppp)
            MBS_class5_PPCP.append(thp_ppcp)

MBS_thp = pd.DataFrame()
MBS_thp['Class1(Baseline)'] = MBS_class1_PPP
MBS_thp['Class3(Baseline)'] = MBS_class3_PPP
MBS_thp['Class5(Baseline)'] = MBS_class5_PPP

MBS_thp['Class1'] = MBS_class1_PPCP
MBS_thp['Class3'] = MBS_class3_PPCP
MBS_thp['Class5'] = MBS_class5_PPCP

MBS_thp = MBS_thp.div(omega*eta)

MBS_thp['alpha'] = alpha_list

mbs_thp = MBS_thp.plot(ax=ax,x='alpha',legend=False,grid=True,alpha=0.5,style=['*--','^--','s--','*-','^-','s-'],title='Mean Throughput (MBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0.3,2.55),xlim=(0.05,0.3))
mbs_thp.set_ylabel('Thr./Req. in MBS [nats/s/Hz/Req.]')
plt.tight_layout()


MBS_thp.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/MBS_throughput.csv', index=False)
###########
### DPS ###
###########
plt.gca().set_prop_cycle(None)
MBS_class1_PPP = []
MBS_class3_PPP = []
MBS_class5_PPP = []
MBS_class1_PPCP = []
MBS_class3_PPCP = []
MBS_class5_PPCP = []

alpha_mbs = [0,2,4]
for al in range(11):
    for idx in alpha_mbs:

        #thp_ppp = Approx_mean_users_DPS_queue(idx,2,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al, weights_j=[1,1,1,1,1.55,1,1,1])
        thp_ppp = Approx_throughput_DPS_queue(idx,2,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al, weights_j=[1,1,1,1,1.8,1,1,1])

        #thp_ppcp = Approx_mean_users_DPS_queue(idx,2,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al, weights_j=[1,1,1,1,1.55,1,1,1])
        #[1,1,1,1,1.55,1,1,1] can reduce mean request from class3 and class5
        thp_ppcp = Approx_throughput_DPS_queue(idx,2,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al, weights_j=[1,1,1,1,1.5,1,1,1])
        
        if idx == 0:
            MBS_class1_PPP.append(thp_ppp)
            MBS_class1_PPCP.append(thp_ppcp)
        elif idx == 2:
            MBS_class3_PPP.append(thp_ppp)
            MBS_class3_PPCP.append(thp_ppcp)
        elif idx == 4:
            MBS_class5_PPP.append(thp_ppp)
            MBS_class5_PPCP.append(thp_ppcp)

MBS_thp = pd.DataFrame()
MBS_thp['Class1(Baseline)'] = MBS_class1_PPP
MBS_thp['Class3(Baseline)'] = MBS_class3_PPP
MBS_thp['Class5(Baseline)'] = MBS_class5_PPP

MBS_thp['Class1'] = MBS_class1_PPCP
MBS_thp['Class3'] = MBS_class3_PPCP
MBS_thp['Class5'] = MBS_class5_PPCP

#MBS_thp = MBS_thp.div(1/100)
MBS_thp = MBS_thp.div(1e-16)
MBS_thp = MBS_thp.div(omega*eta)

MBS_thp['alpha'] = alpha_list

#mbs_meanusers = MBS_thp.plot(x='alpha',grid=True,style=['*--','^--','s--','*-','^-','s-'],title='Mean Requests (MBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0,15),xlim=(0.05,0.3))
mbs_meanusers = MBS_thp.plot(ax=ax,x='alpha',legend=True,grid=True,style=['*--','^--','s--','*-','^-','s-'],title='Mean Throughput (MBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0,2.5),xlim=(0.05,0.3))
mbs_thp.set_ylabel('Thr./Req. in MBS [nats/s/Hz/Req.]')
plt.tight_layout()

MBS_thp.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/MBS_thp_dps.csv', index=False)


#%%

D2D_class1_PPP = []

D2D_class1_PPCP = []

alpha_d2d = [0]
for al in range(11):
    for idx in alpha_d2d:
        #thp_ppp = Approx_throughput_DPS_queue(idx,0,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)
        thp_ppp = throughput_PS_queue(idx,0,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)

        #thp_ppcp = Approx_throughput_DPS_queue(idx,0,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)
        thp_ppcp = throughput_PS_queue(idx,0,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)
        
        if idx == 0:
            D2D_class1_PPP.append(thp_ppp)
            D2D_class1_PPCP.append(thp_ppcp)

D2D_thp = pd.DataFrame()
D2D_thp['Class1(Baseline)'] = D2D_class1_PPP

D2D_thp['Class1'] = D2D_class1_PPCP

D2D_thp = D2D_thp.div(omega*eta)

D2D_thp['alpha'] = alpha_list

d2d_thp = D2D_thp.plot(x='alpha',grid=True,style=['*--','*-'],title='Mean Throughput (D2D) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(1.3,2),xlim=(0.05,0.3))
d2d_thp.set_ylabel('Thr./Req. in D2D [nats/s/Hz/Req.]')
plt.tight_layout()

D2D_thp.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/D2D_throughput.csv', index=False)



#%%
#################
### DPS Queue ###
#################

# No significant result was found on SBS
# However, for MBS when SBSs are PPCP if we weight more on class 5, the throuput of class3 and 5 can be increased significantly when alpha is large enough

'''
SBS_class1_PPP = []
SBS_class2_PPP = []
SBS_class3_PPP = []
SBS_class4_PPP = []
SBS_class5_PPP = []
SBS_class6_PPP = []

SBS_class1_PPCP = []
SBS_class2_PPCP = []
SBS_class3_PPCP = []
SBS_class4_PPCP = []
SBS_class5_PPCP = []
SBS_class6_PPCP = []

alpha_sbs = [0,1,2,3,4,5]
for al in range(11):
    for idx in alpha_sbs:
        thp_ppp = Approx_mean_users_DPS_queue(idx,1,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al, weights_j=[1,1.55,1,1.55,1,1.55,1,1])
        #thp_ppp = throughput_PS_queue(idx,1,D_8x4_PPP,A_8x4_PPP,actived2d_PPP,ai=al)

        thp_ppcp = Approx_mean_users_DPS_queue(idx,1,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al, weights_j=[1,1.55,1,1.55,1,1.55,1,1])
        #thp_ppcp = throughput_PS_queue(idx,1,D_8x4_PPCP,A_8x4_PPCP,actived2d_PPCP,ai=al)
        
        if idx == 0:
            SBS_class1_PPP.append(thp_ppp)
            SBS_class1_PPCP.append(thp_ppcp)
        elif idx == 1:
            SBS_class2_PPP.append(thp_ppp)
            SBS_class2_PPCP.append(thp_ppcp)
        elif idx == 2:
            SBS_class3_PPP.append(thp_ppp)
            SBS_class3_PPCP.append(thp_ppcp)
        elif idx == 3:
            SBS_class4_PPP.append(thp_ppp)
            SBS_class4_PPCP.append(thp_ppcp)
        elif idx == 4:
            SBS_class5_PPP.append(thp_ppp)
            SBS_class5_PPCP.append(thp_ppcp)
        elif idx == 5:
            SBS_class6_PPP.append(thp_ppp)
            SBS_class6_PPCP.append(thp_ppcp)

SBS_thp = pd.DataFrame()
SBS_thp['Class1(Baseline)'] = SBS_class1_PPP
SBS_thp['Class2(Baseline)'] = SBS_class2_PPP
SBS_thp['Class3(Baseline)'] = SBS_class3_PPP
SBS_thp['Class4(Baseline)'] = SBS_class4_PPP
SBS_thp['Class5(Baseline)'] = SBS_class5_PPP
SBS_thp['Class6(Baseline)'] = SBS_class6_PPP

SBS_thp['Class1'] = SBS_class1_PPCP
SBS_thp['Class2'] = SBS_class2_PPCP
SBS_thp['Class3'] = SBS_class3_PPCP
SBS_thp['Class4'] = SBS_class4_PPCP
SBS_thp['Class5'] = SBS_class5_PPCP
SBS_thp['Class6'] = SBS_class6_PPCP

SBS_thp = SBS_thp.div(1e-3)
#SBS_thp = SBS_thp.div(1e-16)
#SBS_thp = SBS_thp.div(omega*eta)

SBS_thp['alpha'] = alpha_list

sbs_thp = SBS_thp.plot(x='alpha',grid=True,style=['*--','^--','s--','o--','v--','+--','*-','^-','s-','o-','v-','+-'],title='Mean Throughput (SBS) of Baseline and Clustered Deployment',figsize=(6,7),ylim=(0.3,15),xlim=(0.05,0.3))
sbs_thp.set_ylabel('Mean Requests in SBS [Req./s]')
plt.tight_layout()


SBS_thp.to_csv('/home/takehiro/Desktop/Research_Numerical/Matlab_plot/SBS_throughput_DPS.csv', index=False)
'''

#%%



#%%



#%%

