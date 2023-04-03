import sympy as sp
import numpy as np


def write_code(fname,codegen_returnval):
    with open(fname+'.c','w+') as file:
        file.write(codegen_returnval[0][1])
    with open(fname+'.h','w+') as file:
        file.write(codegen_returnval[1][1])

lam_rn = sp.Symbol('lRn',real=True,positive=True,nonzero=True)#np.log(2)/(3.8232*86400)
lam_ra = sp.Symbol('lRa',real=True,positive=True,nonzero=True)#np.log(2)/(1600*86400*365)
gamma = sp.Symbol('gamma',real=True,positive=True)
q = sp.Symbol('q',positive=True,nonzero=True,real=True)

F = sp.Matrix(
    [
        [-lam_rn,lam_rn,-lam_rn,0],
        [0,-lam_ra,0,0],
        [0,0,0,1],
        [0,0,0,-gamma]
    ]
)

L = sp.Matrix([
    [0],
    [0],
    [0],
    [1]
])

subs_dict = {
    lam_rn:np.log(2)/(3.8232),
    lam_ra:np.log(2)/(1600*365.25)
}

Q = sp.Matrix([[q]])


P,J = F.jordan_form()

'''
Generate code for the JordanForm of F
'''
from sympy.utilities.codegen import codegen

J_sym = sp.MatrixSymbol('J',*F.shape)
cg = codegen(('JFnc',sp.Eq(J_sym,J.subs(subs_dict))),language='c')

write_code('generated_code\\JFnc',cg)

P_sym = sp.MatrixSymbol('P',*F.shape)

cg = codegen(('PFnc',sp.Eq(P_sym,P.subs(subs_dict))),language='c')

write_code('generated_code\\PFnc',cg)

P_inv_sym = sp.MatrixSymbol('Pinv',*F.shape)

cg = codegen(('PinvFnc',sp.Eq(P_inv_sym,P.inv().subs(subs_dict))),language='c')

write_code('generated_code\\PinvFnc',cg)


'''
Generate code for additional transition noise matrix given by

P * integral t0 t1 e^{J*(t1-tau)}P.inv()LQL^Te^P.inv().T{J(t1-tau)}dtau * P.T (where we omit the outer multiplications)
'''
dt = sp.Symbol('dt')
tau = sp.Symbol('tau')

hlf_inte = sp.exp(J*(dt-tau))*P.inv()*L
inte = hlf_inte*Q*hlf_inte.T
U = sp.integrate(inte,(tau,0,dt))

U_sym = sp.MatrixSymbol('U',*F.shape)
cg = codegen(('UFnc',sp.Eq(U_sym,U.subs(subs_dict))),language='c')

write_code('generated_code\\UFnc',cg)


'''
Generate code for integration (for the measurements)
y = H*K*x

where

K = P integral 0 (t1-t0) e^{J*(tau)} dtau P.T
'''

K_sym = sp.MatrixSymbol('K',*F.shape)

K = sp.integrate(sp.exp(J*tau),(tau,0,dt))

cg = codegen(('KFnc',sp.Eq(K_sym,K.subs(subs_dict))),language='c')

write_code('generated_code\\KFnc',cg)

'''
Generate code for additional variance from integrating the state for measurements
B = integral(t0,t)integral(tau,t),integral(tau,t) e^{J*(a-tau)} P.inv() LQL.T P.inv().T e^{J*(b-tau)} da db dtau
'''
a = sp.Symbol('a')
t0 = sp.Symbol('t0')
t1 = sp.Symbol('t1')

inner_integral = sp.integrate(sp.exp(J*(a-tau)),(a,tau,dt))
integrand = (inner_integral*P.inv()*L*Q*L.T*P.inv().T*inner_integral.T)#.simplify()
B = sp.integrate(integrand,(tau,0,dt))
#B = B.subs({t1-t0:dt})

B_sym = sp.MatrixSymbol('B',*F.shape)

cg = codegen(('BFnc',sp.Eq(B_sym,B.subs(subs_dict))),language='c')

write_code('generated_code\\BFnc',cg)

'''
Generade code for cross-covariance between state-propagation and simultaneous integration over the respective domain
C = integral(t0,t)e^{J*(a-tau)}P.inv()LQL.T P.inv().T integral(tau,t)e^{J*(a-tau)}da dtau
'''

integrand = sp.exp(J*(dt-tau))*P.inv()*L*Q*L.T*P.inv().T*inner_integral.T
C = sp.integrate(integrand,(tau,0,dt))

C_sym = sp.MatrixSymbol('C',*F.shape)

cg = codegen(('CFnc',sp.Eq(C_sym,C.subs(subs_dict))),language='c')

write_code('generated_code\\CFnc',cg)