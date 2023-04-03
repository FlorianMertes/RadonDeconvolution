import os

os.environ['JAX_ENABLE_X64'] = 'True'
import jax
import jax.numpy as jnp

'''
For now, we hardcode the jax functions for the specific arrays
'''

@jax.jit
def expJ_func(dt,gamma):
    return jnp.array([
        [1, 0, 0, 0], 
        [0, jnp.exp(-dt*gamma), 0, 0], 
        [0, 0, jnp.exp(-1.18608347118403e-6*dt), 0], 
        [0, 0, 0, jnp.exp(-0.181300266938676*dt)]
        ])

@jax.jit
def P_func(dt,gamma):
    return jnp.array([
        [-1,-0.18130026693867579/(gamma*gamma - 0.18130026693867579*gamma),1.0000065421372553,1],
        [0.,0.,1.,0.],
        [1.,-1/gamma,0.,0.],
        [0.,1.,0.,0.]
    ])

@jax.jit
def P_inv_func(dt,gamma):
    return jnp.array([
        [0, 0, 1, 1/gamma], 
        [0, 0, 0, 1], 
        [0, 1, 0, 0], 
        [1, -1.00000654213726, 1, 1/(gamma - 0.181300266938676)]
        ])

@jax.jit
def U_func(dt,gamma,q):
    return jnp.array([
        [dt*q/gamma**2, q/gamma**2 - q*jnp.exp(-dt*gamma)/gamma**2, 0, jnp.select([jnp.not_equal(0.181300266938676*gamma**2 - 0.0328697867920351*gamma, 0),True],
        [q/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma) - q*jnp.exp(-0.181300266938676*dt)/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma),dt*q/(gamma**2 - 0.181300266938676*gamma)], default=jnp.nan)],
        [q/gamma**2 - q*jnp.exp(-dt*gamma)/gamma**2, (1/2)*q/gamma - 1/2*q*jnp.exp(-2*dt*gamma)/gamma, 0, -q/((gamma - 0.181300266938676)*(gamma*jnp.exp(0.181300266938676*dt)*jnp.exp(dt*gamma) + 0.181300266938676*jnp.exp(0.181300266938676*dt)*jnp.exp(dt*gamma))) + q/((gamma - 0.181300266938676)*(gamma + 0.181300266938676))], 
        [0, 0, 0, 0], 
        [jnp.select([jnp.not_equal(0.181300266938676*gamma**2 - 0.0328697867920351*gamma, 0),True], 
        [q/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma) - q*jnp.exp(-0.181300266938676*dt)/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma),dt*q/(gamma**2 - 0.181300266938676*gamma)], default=jnp.nan), -q/((gamma - 0.181300266938676)*(gamma*jnp.exp(0.181300266938676*dt)*jnp.exp(dt*gamma) + 0.181300266938676*jnp.exp(0.181300266938676*dt)*jnp.exp(dt*gamma))) + q/((gamma - 0.181300266938676)*(gamma + 0.181300266938676)), 0, jnp.select([jnp.not_equal(0.362600533877352*gamma**2 - 0.13147914716814*gamma + 0.0119186022392266, 0),True], [q/(0.362600533877352*gamma**2 - 0.13147914716814*gamma + 0.0119186022392266) - q*jnp.exp(-0.362600533877352*dt)/(0.362600533877352*gamma**2 - 0.13147914716814*gamma + 0.0119186022392266),dt*q/(gamma**2 - 0.362600533877352*gamma + 0.0328697867920351)], default=jnp.nan)]])

@jax.jit
def K_func(rt,gamma):
    return jnp.array([
        [rt, 0, 0, 0], 
        [0, 1/gamma - jnp.exp(-gamma*rt)/gamma, 0, 0], 
        [0, 0, 843110.98189551 - 843110.98189551*jnp.exp(-1.18608347118403e-6*rt), 0], 
        [0, 0, 0, 5.51571168032669 - 5.51571168032669*jnp.exp(-0.181300266938676*rt)]
        ])

@jax.jit
def B_func(rt,gamma,q):
    return jnp.array([
        [(1/3)*q*rt**3/gamma**2, (1/2)*q*rt**2/gamma**2 - q/gamma**4 - (-gamma*q*rt - q)*jnp.exp(-gamma*rt)/gamma**4, 0, -q*rt**2/(0.362600533877352*gamma**2 - 0.0657395735840702*gamma) + q*rt**2/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma) + jnp.select([jnp.not_equal(0.00595930111961332*gamma**2 - 0.00108042288375384*gamma, 0),True], [-q/(0.00595930111961332*gamma**2 - 0.00108042288375384*gamma) - (-0.181300266938676*q*rt - q)*jnp.exp(-0.181300266938676*rt)/(0.00595930111961332*gamma**2 - 0.00108042288375384*gamma),q*rt**2/(0.362600533877352*gamma**2 - 0.0657395735840702*gamma) - q*rt**2/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma)], default=jnp.nan)],
        [(1/2)*q*rt**2/gamma**2 - q/gamma**4 - (-gamma*q*rt - q)*jnp.exp(-gamma*rt)/gamma**4, q*rt/gamma**2 - 3/2*q/gamma**3 - 1/2*(-4*gamma**3*q*jnp.exp(-gamma*rt) + gamma**3*q*jnp.exp(-2*gamma*rt))/gamma**6, 0, -q*(-gamma**2*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - 0.181300266938676*gamma*jnp.exp(0.181300266938676*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - 0.181300266938676*gamma*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) + 0.181300266938676*gamma/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - 0.0328697867920351*jnp.exp(0.181300266938676*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)))/(gamma - 0.181300266938676) + q*(0.181300266938676*gamma**2*rt*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - gamma**2*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(gamma*rt)) + 0.0328697867920351*gamma*rt*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - 0.181300266938676*gamma*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(gamma*rt)) - 0.181300266938676*gamma*jnp.exp(0.181300266938676*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)) + 0.181300266938676*gamma/(0.0328697867920351*gamma**3 + 0.00595930111961332*gamma**2) - 0.0328697867920351*jnp.exp(0.181300266938676*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)))/(gamma - 0.181300266938676)], [0, 0, 0, 0], [-q*rt**2/(0.362600533877352*gamma**2 - 0.0657395735840702*gamma) + q*rt**2/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma) + jnp.select([jnp.not_equal(0.00595930111961332*gamma**2 - 0.00108042288375384*gamma, 0),True], [-q/(0.00595930111961332*gamma**2 - 0.00108042288375384*gamma) - (-0.181300266938676*q*rt - q)*jnp.exp(-0.181300266938676*rt)/(0.00595930111961332*gamma**2 - 0.00108042288375384*gamma),q*rt**2/(0.362600533877352*gamma**2 - 0.0657395735840702*gamma) - q*rt**2/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma)], default=jnp.nan), -q*(-gamma**2*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - 0.181300266938676*gamma*jnp.exp(0.181300266938676*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - 0.181300266938676*gamma*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) + 0.181300266938676*gamma/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - 0.0328697867920351*jnp.exp(0.181300266938676*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)))/(gamma - 0.181300266938676) + q*(0.181300266938676*gamma**2*rt*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - gamma**2*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(gamma*rt)) + 0.0328697867920351*gamma*rt*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - 0.181300266938676*gamma*jnp.exp(gamma*rt)/(0.0328697867920351*gamma**3*jnp.exp(gamma*rt) + 0.00595930111961332*gamma**2*jnp.exp(gamma*rt)) - 0.181300266938676*gamma*jnp.exp(0.181300266938676*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)) + 0.181300266938676*gamma/(0.0328697867920351*gamma**3 + 0.00595930111961332*gamma**2) - 0.0328697867920351*jnp.exp(0.181300266938676*rt)/(0.0328697867920351*gamma**3*jnp.exp(0.181300266938676*rt) + 0.00595930111961332*gamma**2*jnp.exp(0.181300266938676*rt)))/(gamma - 0.181300266938676), 0, q*rt/(0.0328697867920351*gamma**2 - 0.0119186022392266*gamma + 0.00108042288375384) + jnp.select([jnp.not_equal(7.10265396684492e-5*gamma**4 - 5.15085224064811e-5*gamma**3 + 1.40077632928677e-5*gamma**2 - 1.6930741494738e-6*gamma + 7.67386988116428e-8, 0),True], [-((-0.0238372044784533*gamma**2*q + 0.00864338307003076*gamma*q - 0.000783523828924904*q)*jnp.exp(-0.181300266938676*rt) + (0.00595930111961332*gamma**2*q - 0.00216084576750769*gamma*q + 0.000195880957231226*q)*jnp.exp(-0.362600533877352*rt))/(7.10265396684492e-5*gamma**4 - 5.15085224064811e-5*gamma**3 + 1.40077632928677e-5*gamma**2 - 1.6930741494738e-6*gamma + 7.67386988116428e-8) + (-0.01787790335884*gamma**2*q + 0.00648253730252307*gamma*q - 0.000587642871693678*q)/(7.10265396684492e-5*gamma**4 - 5.15085224064811e-5*gamma**3 + 1.40077632928677e-5*gamma**2 - 1.6930741494738e-6*gamma + 7.67386988116428e-8),-q*rt/(0.0328697867920351*gamma**2 - 0.0119186022392266*gamma + 0.00108042288375384)], default=jnp.nan)]
        ])

@jax.jit
def C_func(rt,gamma,q):
    return jnp.array([[(1/2)*q*rt**2/gamma**2, q*rt/gamma**2 - q/gamma**3 + q*jnp.exp(-gamma*rt)/gamma**3, 0, q*rt/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma) + jnp.select([jnp.not_equal(0.0328697867920351*gamma**2 - 0.00595930111961332*gamma, 0),True], [-q/(0.0328697867920351*gamma**2 - 0.00595930111961332*gamma) + q*jnp.exp(-0.181300266938676*rt)/(0.0328697867920351*gamma**2 - 0.00595930111961332*gamma),-q*rt/(0.181300266938676*gamma**2 - 0.0328697867920351*gamma)], default=jnp.nan)], [q/gamma**3 - (gamma*q*rt + q)*jnp.exp(-gamma*rt)/gamma**3, (1/2)*q/gamma**2 - 1/2*(2*gamma**2*q*jnp.exp(-gamma*rt) - gamma**2*q*jnp.exp(-2*gamma*rt))/gamma**4, 0, q*(gamma*jnp.exp(0.181300266938676*rt)/(0.181300266938676*gamma**2*jnp.exp(0.181300266938676*rt) + 0.0328697867920351*gamma*jnp.exp(0.181300266938676*rt)) - gamma/(0.181300266938676*gamma**2 + 0.0328697867920351*gamma) + 0.181300266938676*jnp.exp(0.181300266938676*rt)/(0.181300266938676*gamma**2*jnp.exp(0.181300266938676*rt) + 0.0328697867920351*gamma*jnp.exp(0.181300266938676*rt)))/(gamma - 0.181300266938676) - q*(gamma*jnp.exp(0.181300266938676*rt)/(0.181300266938676*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.0328697867920351*gamma*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - gamma/(0.181300266938676*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.0328697867920351*gamma*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) + 0.181300266938676*jnp.exp(0.181300266938676*rt)/(0.181300266938676*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.0328697867920351*gamma*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)))/(gamma - 0.181300266938676)], [0, 0, 0, 0], [jnp.select([jnp.not_equal(0.0328697867920351*gamma**2 - 0.00595930111961332*gamma, 0),True], [q/(0.0328697867920351*gamma**2 - 0.00595930111961332*gamma) - (0.181300266938676*q*rt + q)*jnp.exp(-0.181300266938676*rt)/(0.0328697867920351*gamma**2 - 0.00595930111961332*gamma),-q*rt**2/(2*gamma**2 - 0.362600533877352*gamma) + q*rt**2/(gamma**2 - 0.181300266938676*gamma)], default=jnp.nan), q*(gamma*jnp.exp(gamma*rt)/(0.181300266938676*gamma**2*jnp.exp(gamma*rt) + 0.0328697867920351*gamma*jnp.exp(gamma*rt)) + 0.181300266938676*jnp.exp(gamma*rt)/(0.181300266938676*gamma**2*jnp.exp(gamma*rt) + 0.0328697867920351*gamma*jnp.exp(gamma*rt)) - 0.181300266938676/(0.181300266938676*gamma**2 + 0.0328697867920351*gamma))/(gamma - 0.181300266938676) - q*(gamma*jnp.exp(gamma*rt)/(0.181300266938676*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.0328697867920351*gamma*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) + 0.181300266938676*jnp.exp(gamma*rt)/(0.181300266938676*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.0328697867920351*gamma*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)) - 0.181300266938676/(0.181300266938676*gamma**2*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt) + 0.0328697867920351*gamma*jnp.exp(0.181300266938676*rt)*jnp.exp(gamma*rt)))/(gamma - 0.181300266938676), 0, jnp.select([jnp.not_equal(0.00216084576750769*gamma**4 - 0.00156704765784981*gamma**3 + 0.000426159238010695*gamma**2 - 5.15085224064811e-5*gamma + 2.33462721547795e-6, 0),True], [-((-0.0328697867920351*gamma**2*q + 0.0119186022392266*gamma*q - 0.00108042288375384*q)*jnp.exp(-0.362600533877352*rt) + (0.0657395735840702*gamma**2*q - 0.0238372044784533*gamma*q + 0.00216084576750769*q)*jnp.exp(-0.181300266938676*rt))/(0.00216084576750769*gamma**4 - 0.00156704765784981*gamma**3 + 0.000426159238010695*gamma**2 - 5.15085224064811e-5*gamma + 2.33462721547795e-6) + (0.0328697867920351*gamma**2*q - 0.0119186022392266*gamma*q + 0.00108042288375384*q)/(0.00216084576750769*gamma**4 - 0.00156704765784981*gamma**3 + 0.000426159238010695*gamma**2 - 5.15085224064811e-5*gamma + 2.33462721547795e-6),0], default=jnp.nan)]])




@jax.jit
def discretize_timestep(dt,theta):
    P = P_func(dt,theta[0])
    expJ = expJ_func(dt,theta[0])
    Pinv = P_inv_func(dt,theta[0])
    U = U_func(dt,theta[0],2*theta[1]*theta[0])
    return P@expJ@Pinv,P@U@P.T


@jax.jit
def discretize_measurement_step(rt,theta):
    '''
    Return Psi,U,K,B,C
    '''
    P = P_func(rt,theta[0])
    expJ = expJ_func(rt,theta[0])
    Pinv = P_inv_func(rt,theta[0])
    U = U_func(rt,theta[0],2*theta[1]*theta[0])
    K = K_func(rt,theta[0])
    B = B_func(rt,theta[0],2*theta[1]*theta[0])
    C = C_func(rt,theta[0],2*theta[1]*theta[0])
    return P@expJ@Pinv,P@U@P.T,P@K@Pinv,P@B@P.T,P@C@P.T