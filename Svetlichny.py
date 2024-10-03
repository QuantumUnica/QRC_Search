import torch
import random
from QRC import QRC_Device


def KroneckerMultiplyList(myList, device):
    result = torch.ones((1, 1), dtype=torch.float32, device=device)
    for x in myList:
        result = torch.kron(result, x.to(device, non_blocking=True))
    return result

def nu(k):
    nu0 = (-1) ** (k * (k + 1) / 2)
    return nu0

def sign(t):
    s = torch.sum(t == 1).item()
    return s


def svet_ineq(rho, N, angles, device):
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device, requires_grad=False)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device, requires_grad=False)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device, requires_grad=False)

    ops = []
    l = torch.tensor([0, 1]).to(device)
    M = torch.cartesian_prod(*[l] * N).to(device)

    S = torch.zeros_like(rho, device=device)

    for A in range(N):
        alpha1 = angles[A][0]
        alpha2 = angles[A][1]
        beta1 = angles[A][2]
        beta2 = angles[A][3]
        A1 = torch.cos(alpha1) * torch.sin(beta1) * X + torch.sin(alpha1) * torch.sin(beta1) * Y + torch.cos(beta1) * Z
        A2 = torch.cos(alpha2) * torch.sin(beta2) * X + torch.sin(alpha2) * torch.sin(beta2) * Y + torch.cos(beta2) * Z
        ops.append([A1, A2])

    for I in M:
        Term = [ops[j][I[j]] for j in range(len(I))]
        T = nu(sign(I)) * KroneckerMultiplyList(Term, device)
        S += T   # Sum terms

    svl = torch.sqrt((torch.trace(S @ rho).real) ** 2)

    return -svl