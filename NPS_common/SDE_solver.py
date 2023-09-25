# baoab solver adapted from https://hockygroup.hosting.nyu.edu/exercise/langevin-dynamics.html

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



#this is step A
# @torch.jit.script
def position_update(x,v,dt):
    x_new = x + v*dt/2.
    return x_new

#this is step B
# @torch.jit.script
def velocity_update(v,F,dt):
    v_new = v + F*dt/2.
    return v_new

# def random_velocity_update(v,gamma,kBT,dt):
#     R = np.random.normal()
#     c1 = np.exp(-gamma*dt)
#     c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT)
#     v_new = c1*v + R*c2
#     return v_new
def random_velocity_update(v,c1,c2):
    if BACKEND == 'numpy':
        R = np.random.normal(*v.shape)
    else:
        R = torch.randn_like(v)
    # c1 = np.exp(-gamma*dt)
    # c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT)
    v_new = c1*v + R*c2
    return v_new

if BACKEND == 'jax':
    @jax.jit
    def random_velocity_update(v,c1,c2, key):
        key, subkey = jax.random.split(key)
        R = jax.random.normal(subkey, v.shape)
        # c1 = np.exp(-gamma_dt)
        # c2 = np.sqrt((1-c1*c1)*kBT)
        v_new = c1*v + R*c2
        return v_new, key


# @torch.jit.script
# def random_velocity_update(v,c1,c2):
#     R = torch.randn_like(v)
#     # c1 = (-gamma_dt).exp()
#     # c2 = ((1-c1*c1)*kBT).sqrt()
#     v_new = c1*v + R*c2
#     return v_new

# @torch.jit.script
def baoab(potential, max_time, dt, gamma, kBT, initial_position, initial_velocity, save_frequency=3, device='cpu'):
    x = initial_position
    v = initial_velocity
    step_number = 0
    tot_step = int(max_time//dt)
    tot_save = (tot_step-1)//save_frequency + 1
    c1 = np.exp(-gamma*dt).astype(x.dtype)
    c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT).astype(x.dtype)
    if BACKEND == 'jax':
        RNGkey = jax.random.PRNGKey(999)
    if BACKEND == 'torch':
        x = torch.from_numpy(x).float().to(device)
        v = torch.from_numpy(v).float().to(device)
        traj = x.new_zeros((4, tot_save, *x.shape)).to(device)
        c1 = torch.tensor(c1).float().to(device)
        c2 = torch.tensor(c2).float().to(device)
    else:
        traj = np.zeros((4, tot_save, *x.shape))

    for step_number in range(tot_step):
        # B
        potential_energy, force = potential(x)
        if step_number%save_frequency == 0:
            e_total = .5*v*v + potential_energy
            step = step_number//save_frequency
            traj[0, step] = x
            traj[1, step] = v
            traj[2, step] = e_total
            traj[3, step] = dt*step_number
        v = velocity_update(v,force,dt)
        #A
        x = position_update(x,v,dt)
        #O
        if BACKEND == 'jax':
            v, RNGkey = random_velocity_update(v,c1,kBT, RNGkey)
        else:
            v = random_velocity_update(v, c1, c2)
        #A
        x = position_update(x,v,dt)
        # B
        potential_energy, force = potential(x)
        v = velocity_update(v,force,dt)

    return traj.cpu()


# @torch.jit.script
def Euler_Maruyama(a_det, b_sto, max_time, dt, initial_position, t0=0, save_frequency=3, device='cpu'):
    """
    initial_position or x shape: N_batch x N_dim
    delta x = a*dt + b*gaussian*sqrt(dt)
    """
    x = initial_position
    step_number = 0
    tot_step = int(max_time//dt)
    tot_save = (tot_step-1)//save_frequency + 1
    if BACKEND == 'jax':
        RNGkey = jax.random.PRNGKey(999)
    if BACKEND == 'torch':
        x = torch.from_numpy(x).float().to(device)
        traj = x.new_zeros((tot_save, *x.shape)).to(device)
    else:
        traj = np.zeros((tot_save, *x.shape))

    sqrt_dt = np.sqrt(dt)
    for step_number, t in zip(list(range(tot_step)), t0+np.arange(tot_step)*dt):
        if step_number%save_frequency == 0:
            step = step_number//save_frequency
            traj[step] = x
        a = a_det(x, t)
        b = b_sto(x, t)
        if BACKEND == 'jax':
            raise NotImplementedError('')
            v, RNGkey = random_velocity_update(v,c1,kBT, RNGkey)
        else:
            x += a*dt + b*torch.randn_like(x)*sqrt_dt

    return traj.cpu()


# @torch.jit.script
# @jax.jit
def harmonic_oscillator_energy_force(x,k=1,x0=0):
    #calculate the energy on force on the right hand side of the equal signs
    energy = 0.5*k*(x-x0)**2
    force = -k*(x-x0)
    return energy, force

def double_well_energy_force(x,k,a):
    #calculate the energy on force on the right hand side of the equal signs
    energy = 0.25*k*((x/a-1)**2) * ((x/a+1)**2)
    force = -k*x*(x-a)*(x+a)/a**4
    return energy, force


