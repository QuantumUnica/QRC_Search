import torch

"""
     Mathematical utilities
"""
def KroneckerMultiplyList(myList, device):
    result = torch.ones((1, 1), dtype=torch.float64, device=device)
    for x in myList:
        result = torch.kron(result, x.to(device, non_blocking=True))
    return result

def nu(k):
    nu0 = (-1) ** (k * (k + 1) / 2)
    return nu0

def sign(t):
    s = torch.sum(t == 1).item()
    return s

def are_valid_angles(t):
    lower_bound = 0
    upper_bound = 2 * torch.pi
    is_in_range = torch.all((t >= lower_bound) & (t <= upper_bound))
    return is_in_range.item()


"""
    To produce a GHZ state
"""
def proj(vector: torch.Tensor) -> torch.Tensor:
    return torch.outer(vector, vector.conj())

def comp_basis(dim: int) -> torch.Tensor:
    return torch.eye(dim)

def ghz(n_qubits: int) -> torch.Tensor:
    vector = 1 / torch.sqrt(torch.tensor(2.0)) * (comp_basis(2 ** n_qubits)[0] + comp_basis(2 ** n_qubits)[2 ** n_qubits - 1])
    result = proj(vector)
    return result