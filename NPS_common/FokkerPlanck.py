"""
Close form solutions of the Fokker-Planck Equation
Including:
* Second order Langevin equation of 1D particle in a Harmonic potential
  d^2 x(t)/dt^2 = - gamma dx/dt - omega^2 x + xi(t), <xi(t) xi(t') = 2 lambda delta(t-t') = 2 gamma kBT/m delta(t-t')
    ** beta^2-4omega^2 > 0: Overdamped
        *** omega=0:        Special case of free particle
    ** beta^2-4omega^2 < 0: Underdamped/periodic
    ** beta = 2omega:       Critically damped/aperiodic

* First order Langevin in a harmonic potential, i.e. 
  dx/dt = - gamma x + sigma dW, D = sigma^2/2
"""

BACKEND='torch'

if BACKEND == 'numpy':
    import numpy as np
elif BACKEND == 'torch':
    import numpy as np
    import torch
elif BACKEND == 'jax':
    import jax
    import jax.numpy as np
else:
    raise NotImplemented(f'unknown backend {BACKEND}')


# from Physica A 422, 203 (2015)
def W_JimenezAquino(x, v, x0, v0, t, beta, omega, lamb):
    beta1 = np.sqrt(beta**2 - 4* omega**2)
    # mu1, mu2,
    # a, b, h,
    # xi0, eta0, xi, eta, Deltac,
    # expterm, qterm, qtermFree
    mu1 = -beta/2 + beta1/2
    mu2 = -beta/2 - beta1/2
    a = lamb/mu1* (1 - np.exp(-2 *mu1* t))
    b = lamb/mu2* (1 - np.exp(-2 *mu2* t))
    h = -2 *lamb/(mu1 + mu2)* (1 - np.exp(-(mu1 + mu2)* t))
    xi0  = x0*mu1 - v0
    eta0 = x0*mu2 - v0
    xi  = (x*mu1 - v)* np.exp(-mu2* t)
    eta = (x*mu2 - v)* np.exp(-mu1* t)
    Deltac = a * b - h**2
    qterm = a *(xi-xi0)**2 + 2* h* (xi-xi0)* (eta-eta0) + b* (eta-eta0)**2
    # qterm = (lamb* mu2* (1 - Exp[-2 mu1 t]) (xi - xi0)^2 + 
    #     omega^2  2 h (xi - xi0) (eta - eta0) + 
    #     lambda mu1 (1 - Exp[-2 mu2 t]) (eta - eta0)^2)/omega^2;
    expterm = -1/(2* Deltac)* qterm
    return np.exp(beta* t)/(2* np.pi* np.sqrt(Deltac))* np.exp(expterm)
#     qtermFree = 
#         lamb* t* (xi - xi0)**2 + 2 h (xi - xi0) (eta - eta0) + 
#         b (eta - eta0)^2 /. omega -> 0;
#         {Exp[beta t]/(2 Pi Sqrt[Deltac]) Exp[expterm], expterm, qterm, 
#         qtermFree}
#    ];

# Risken "The Fokker-Planck Equation" 1055, page 238
def W_Risken(x, v, x0, v0, t, gamma, omega, kBT, m=1, normalization=True):
    gamma1 = np.sqrt(gamma**2 - 4* omega**2)
    lm1 = (gamma+gamma1)/2
    lm2 = (gamma-gamma1)/2
    vth = np.sqrt(kBT/m)
    lm1_m_lm2 = gamma1
    lm1_m_lm2_sq = gamma**2 - 4*omega**2
    lm1_p_lm2 = gamma
    lm1_ti_lm2 = omega**2
    exp_mlm1t = np.exp(-lm1*t)
    exp_mlm2t = np.exp(-lm2*t)
    sigma_xx = gamma*vth**2/lm1_m_lm2_sq*(lm1_p_lm2/lm1_ti_lm2 + 4/lm1_p_lm2*(np.exp(-lm1_p_lm2*t)-1) \
             - 1/lm1*np.exp(-2*lm1*t) - 1/lm2*np.exp(-2*lm2*t))
    sigma_xv = gamma*vth**2/lm1_m_lm2_sq* (np.exp(-lm1*t) - np.exp(-lm2*t))**2
    sigma_vv = gamma*vth**2/lm1_m_lm2_sq*(lm1_p_lm2 + 4*lm1_ti_lm2/lm1_p_lm2*(np.exp(-lm1_p_lm2*t)-1) \
             -lm1*np.exp(-2*lm1*t) - lm2*np.exp(-2*lm2*t))
    det_sigma = sigma_xx * sigma_vv - sigma_xv**2
    sigma_inv_xx = sigma_vv/det_sigma
    sigma_inv_xv = - sigma_xv/det_sigma
    sigma_inv_vv = sigma_xx/det_sigma
    exp_mgammat_xx = (lm1*exp_mlm2t - lm2*exp_mlm1t)/lm1_m_lm2
    exp_mgammat_xv = (exp_mlm2t - exp_mlm1t)/lm1_m_lm2
    exp_mgammat_vx = omega**2 *(exp_mlm1t - exp_mlm2t)/lm1_m_lm2
    exp_mgammat_vv = (lm1*exp_mlm1t - lm2*exp_mlm2t)/lm1_m_lm2
    xt = exp_mgammat_xx* x0 + exp_mgammat_xv* v0
    vt = exp_mgammat_vx* x0 + exp_mgammat_vv* v0
    norm_fac = 1/(2*np.pi)/np.sqrt(det_sigma) if normalization else 1
    return norm_fac * np.exp(-1/2*sigma_inv_xx*(x-xt)**2 \
        -sigma_inv_xv*(x-xt)*(v-vt) -1/2*sigma_inv_vv* (v-vt)**2)


# Simplified from Risken "The Fokker-Planck Equation" 1055, page 238
# Fock-Planck Equation of Harmonic oscillator
# 2nd order Langevin 
def W(x, v, x0, v0, t, gamma, omega, kBT, m=1, normalization=True):
    assert gamma > 0 and kBT > 0 and m > 0 and t > 0
    exp_M_gammat = np.exp(-gamma*t)
    omeSq = np.real(omega**2)
    if omega ==0:
        Gamma = np.array([[1, 1/gamma*(1-exp_M_gammat)], 
                         [0, exp_M_gammat]])
        Sigma = kBT/m* np.array([[1/gamma**2*(2*gamma*t+4*exp_M_gammat-exp_M_gammat**2-3), 1/gamma*(1-exp_M_gammat)**2],
                                 [1/gamma*(1-exp_M_gammat)**2, 1-exp_M_gammat**2]])
    else:
        omega1 = np.sqrt(np.abs(omeSq - gamma**2/4))
        if gamma**2 > 4*omeSq: # over-damped
            C, S = np.cosh(omega1*t), np.sinh(omega1*t)/omega1
        elif gamma == 2*omega:
            C, S = 1, t
        else:
            C, S = np.cos (omega1*t), np.sin (omega1*t)/omega1
        Gamma = np.exp(-gamma*t/2)* np.array([[C+gamma*S/2, S],
                                              [-omeSq*S, C-gamma*S/2]])
        Sigma = kBT/m*(np.array([[1/omeSq,0],[0,1]])-exp_M_gammat*np.array(
            [[1/omeSq*(1+gamma*C*S+gamma**2*S**2/2), -gamma*S**2],
             [-gamma*S**2, 1-gamma*C*S+gamma**2*S**2/2]]))
    xt = Gamma[0,0]* x0 + Gamma[0,1]* v0
    vt = Gamma[1,0]* x0 + Gamma[1,1]* v0
    det_sigma = Sigma[0,0]*Sigma[1,1] - Sigma[1,0]*Sigma[0,1]
    Sigma_inv = np.linalg.inv(Sigma)
    norm_fac = 1/(2*np.pi)/np.sqrt(det_sigma) if normalization else 1
    # print(xt, vt)
    return norm_fac * np.exp(-1/2*Sigma_inv[0,0]*(x-xt)**2 -Sigma_inv[0,1]*(x-xt)*(v-vt) -1/2*Sigma_inv[1,1]* (v-vt)**2)


# Fock-Planck Equation of Harmonic oscillator 
# 1st order Langevin
#   dx/dt = - k x + sigma dW, D = sigma^2/2
# Risken 5.28, page 100
def W_OrnsteinUhlenbeck(x, x0, t, k, D, normalization=True):
    xt = x0* np.exp(-k*t)
    var = D * (1-np.exp(-2*k*t)) /k if k != 0 else D*2*t
    norm_fac = 1/np.sqrt((2*np.pi*var)) if normalization else 1
    return norm_fac * np.exp(-1/2*(x-xt)**2/var)

