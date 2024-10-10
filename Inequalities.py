import torch
import numpy as np
import random
from operator import itemgetter
from utils import *

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, requires_grad=False)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, requires_grad=False)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, requires_grad=False)


def svetlichny_iter(rho, N, angles, device):
    
    X_device = X.to(device)
    Y_device = Y.to(device)
    Z_device = Z.to(device)
    
    ops = []
    l = torch.tensor([0, 1]).to(device)
    M = torch.cartesian_prod(*[l] * N).to(device)

    S = torch.zeros_like(rho, device=device)

    for A in range(N):
        alpha1 = angles[A][0]
        alpha2 = angles[A][1]
        beta1 = angles[A][2]
        beta2 = angles[A][3]
        A1 = torch.cos(alpha1) * torch.sin(beta1) * X_device + torch.sin(alpha1) * torch.sin(beta1) * Y_device + torch.cos(beta1) * Z_device
        A2 = torch.cos(alpha2) * torch.sin(beta2) * X_device + torch.sin(alpha2) * torch.sin(beta2) * Y_device + torch.cos(beta2) * Z_device
        ops.append([A1, A2])

    for I in M:
        Term = [ops[j][I[j]] for j in range(len(I))]
        T = nu(sign(I)) * KroneckerMultiplyList(Term, device)
        S += T   # Sum terms

    return S

def mermin_iter(angs, n, device):

    X_device = X.to(device)
    Y_device = Y.to(device)
    Z_device = Z.to(device)
    
    # Inizializziamo i valori di m1 e m1p per n=1
    m1 = torch.cos(angs[0]) * torch.sin(angs[1]) * X_device + torch.sin(angs[0]) * torch.sin(angs[1]) * Y_device + torch.cos(angs[1]) * Z_device
    m1p = torch.cos(angs[2]) * torch.sin(angs[3]) * X_device + torch.sin(angs[2]) * torch.sin(angs[3]) * Y_device + torch.cos(angs[3]) * Z_device
    
    # Se n è 1, restituiamo semplicemente [m1, m1p]
    if n == 1:
        return [m1, m1p]

    # Per n > 1, usiamo un ciclo iterativo per costruire i valori
    prev_mn = m1
    prev_mnp = m1p

    for i in range(2, n+1):
        coeff = 4 * (i - 1)
        
        new_angs0 = angs[coeff]
        new_angs1 = angs[coeff + 1]
        new_angs2 = angs[coeff + 2]
        new_angs3 = angs[coeff + 3]

        A1_new = torch.cos(new_angs0) * torch.sin(new_angs1) * X_device + torch.sin(new_angs0) * torch.sin(new_angs1) * Y_device + torch.cos(new_angs1) * Z_device
        A2_new = torch.cos(new_angs2) * torch.sin(new_angs3) * X_device + torch.sin(new_angs2) * torch.sin(new_angs3) * Y_device + torch.cos(new_angs3) * Z_device

        # Calcolo dei nuovi mn e mnp utilizzando i valori precedenti
        mn = 1/2 * (torch.kron(prev_mn, A1_new + A2_new) + torch.kron(prev_mnp, A1_new - A2_new))
        mnp = 1/2 * (torch.kron(prev_mnp, A2_new + A1_new) + torch.kron(prev_mn, A2_new - A1_new))

        # Aggiorniamo prev_mn e prev_mnp per il prossimo passo iterativo
        prev_mn = mn
        prev_mnp = mnp

    return [prev_mn, prev_mnp]

'''
    Daniel Collins,Nicolas Gisin, Sandu Popescu,David Roberts and Valerio Scarani
    Bell-Type Inequalities to Detect True n-Body Nonseparability
    PHYSICAL REVIEW LETTERS 29 APRIL 2002

    Recursive Definition of Generalized Svetlichny polynomials

    Sve_n = Mermin_n,  of n is even
    Sve_n= 1/2*(mMrmin_n+ (Mermin_n)',  if n is odd

    where (Mermin_k)'  is obtained from Mermin_k by exchanging all the
    primed and non-primed a_i’s.

    This function defines the Mermin-Klyshko obserbvables

    Params:
        angs:   the number of angles that are necessary for the dichotomic observables;
        n:      numbero of qubit

''' 
def mermin_rec(angs, n, device):
    X_device = X.to(device)
    Y_device = Y.to(device)
    Z_device = Z.to(device)

    if n==1:  
        # m1 is a1: the first observable of qubit1 and m1P is a1': the first observable of qubit2
        m1 = torch.cos(angs[0]) * torch.sin(angs[1])*X_device + torch.sin(angs[0])*torch.sin(angs[1])*Y_device + torch.cos(angs[1])*Z_device
        m1p = torch.cos(angs[2]) * torch.sin(angs[3])*X_device + torch.sin(angs[2])*torch.sin(angs[3])*Y_device + torch.cos(angs[3])*Z_device
        return[m1, m1p]
        
    else:
        coeff=4*n-4
       
        new_angs0=angs[coeff]
        new_angs1=angs[coeff+1]
        new_angs2=angs[coeff+2]
        new_angs3=angs[coeff+3]
        
        A1_new = torch.cos(new_angs0)*torch.sin(new_angs1)*X + torch.sin(new_angs0)*torch.sin(new_angs1)*Y + torch.cos(new_angs1)*Z
        A2_new = torch.cos(new_angs2)*torch.sin(new_angs3)*X + torch.sin(new_angs2)*torch.sin(new_angs3)*Y + torch.cos(new_angs3)*Z
        
        # mn is Mermin_n and mnp is (Mermin_n)'
        mn = 1/2 * (torch.kron(mermin_rec(angs, n-1, device)[0], A1_new + A2_new) + torch.kron(mermin_rec(angs, n-1, device)[1], A1_new - A2_new))
        mnp = 1/2 * (torch.kron(mermin_rec(angs, n-1, device)[1], A2_new + A1_new) + torch.kron(mermin_rec(angs, n-1, device)[0], A2_new - A1_new))
              
        return [mn,mnp]
    
"""    
    The function calculates the expectation value 
    of the Mermin-Kyshko or Svetlichny observables applied to a state rho"
"""
def get_expectation_value(ineq_type, rho, angs, device):     
    assert ineq_type in ['mer','svet'],  'ineq_type parameter must be mer or svet'

    n = int(np.log2(np.shape(rho)[0]))   # n is the number of qubits

    if ineq_type == 'mer':
        angs = torch.cat(angs, axis=0)
        me = mermin_iter(angs, n, device)[0]   
        expectation = -torch.trace(me @ rho).real

    elif ineq_type == 'svet':
        sv = svetlichny_iter(rho, n, angs, device)
        expectation = -torch.sqrt((torch.trace(sv @ rho).real) ** 2)
        
    return expectation


def get_limits(n_particles):
    
    N = n_particles
    classical = 2**(N-1)
    svet_quantum_limit = classical * np.sqrt(2)

    #print("Svet Quantum Limit: ", svet_quantum_limit)
    #print("Svet Classical Limit: ", classical)

    if N%2==0: 
        mermin_limit=2**(N-2/2)      
        compare_string = 'equal to'
        parity = 'even'
    if N%2!=0:
        mermin_limit=2**((N-1)/2)
        compare_string = 'different from'
        parity = 'odd'

    #print(f"Max Mermin violation: {mermin_limit} - It's {compare_string} Svetlichny when the number of qubits {N} is {parity}")

    return classical, svet_quantum_limit, mermin_limit



if __name__ == "__main__":
    
    device =  torch.device('cuda', 1)
    N = 3
    seeds = 5
    ineq_str_selector = 'svet'

    ghz_state = ghz(N)
    rho = ghz_state.to(torch.complex64).to(device)
    results = []

    for _ in range(seeds):

        def f(ang):
            angles = [ang[4 * i:4 * (i + 1)] for i in range(N)]
            poly_value = get_expectation_value(ineq_str_selector, rho, angles, device)
            return poly_value

        initial_guess = torch.tensor([2 * torch.pi * random.random() for _ in range(4 * N)], dtype=torch.float64, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([initial_guess], lr=0.3)

        # Optimization loop
        for _ in range(200):  # Number of epochs
            def closure():
                optimizer.zero_grad()
                loss = f(initial_guess)
                loss.backward()
                return loss
            
            optimizer.step(closure)
         
            with torch.no_grad():
                initial_guess.clamp_(0, 2 * torch.pi)   # Ensures parameters constraints
                
        fitted_params = initial_guess
        results.append([-f(fitted_params).to('cpu').detach().numpy(), fitted_params.to('cpu').detach().numpy()])

    maximal_pair = max(enumerate(map(itemgetter(0), results)), key=itemgetter(1))
    item = results[maximal_pair[0]]
    violation_value = item[0]

    print(violation_value)