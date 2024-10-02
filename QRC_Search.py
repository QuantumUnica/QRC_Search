import warnings
warnings.filterwarnings('ignore') 
import os
import json
import socket
import tempfile
import time
import numpy as np
import random
import math as m
import pandas as pd
from operator import itemgetter

import torch
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim

from QRC import QRC_Device

"""
    Math functions
"""
def SVL_aux(rho, N, angles, device):
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device, requires_grad=False)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device, requires_grad=False)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device, requires_grad=False)

    OPS = []
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
        OPS.append([A1, A2])

    for I in M:
        Term = [OPS[j][I[j]] for j in range(len(I))]
        T = nu(sign(I)) * KroneckerMultiplyList(Term, device)
        S += T   # Sum terms

    Svl = torch.sqrt((torch.trace(S @ rho).real) ** 2)

    return -Svl

def KroneckerMultiplyList(myList, device):
    result = torch.ones((1, 1), dtype=torch.float32, device=device)
    for x in myList:
        result = torch.kron(result, x.to(device, non_blocking=True))
    return result

def nu(k, branch=False):
    nu0 = (-1) ** (k * (k + 1) / 2)
    return nu0

def sign(t):
    s = torch.sum(t == 1).item()
    return s




def save_to_tempfile(data, filename):
    """
    Save the data to a file by appending new data to existing content.
    """
    """if os.path.exists(filename):
        # Load existing data
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    else:
        # If file does not exist, initialize an empty list """
    
    existing_data = []

    existing_data.extend(data)

    # Write all data back to the file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir='Results', mode='w') as tmpfile:
        json.dump(existing_data, tmpfile, indent=4)
    os.rename(tmpfile.name, filename)

def partition_dataset(r, world_size, dataset):
    """Divide il dataset tra i processi in modo esclusivo."""
    # Calcolare la dimensione di ogni sottoinsieme
    total_size = len(dataset)
    chunk_size = total_size // world_size
    remainder = total_size % world_size

    # Determinare l'indice iniziale e finale per ciascun processo
    if r < remainder:
        start_idx = r * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = r * chunk_size + remainder
        end_idx = start_idx + chunk_size

    return dataset[start_idx:end_idx]

def worker(shared_list, lock_list, lock_fSave, rank, attempts):

    """Worker function that appends a dictionary to the shared list."""
    # Creazione di un dizionario con dati da aggiungere alla lista

    device = torch.device("cuda", rank) if gpu else torch.device("cpu") # Assign correct device

    for i, _ in enumerate(attempts):
        depth = random.randint(10, 30)
        random_circuit = QRC_Device("Aria1", N, depth, max_gates=2)

        RHO = torch.from_numpy(random_circuit['Ideal RHO']).type(torch.complex64).to(device)
        circuit = random_circuit["Circuit Without Density Matrix"]

        # Definition of violation level criteria
        criterium =0.9*2**(N-1)*np.sqrt(2)
        classical = 2**(N-1)

        results = []

        start_time = time.time()

        # Run seeds searches for each RHO
        for _ in range(seeds):

            def f(ang):
                angles = [ang[4 * i:4 * (i + 1)] for i in range(N)]
                Svl = SVL_aux(RHO, N, angles, device)
                return Svl

            initial_guess = torch.tensor([2 * m.pi * random.random() for _ in range(4 * N)], requires_grad=True, device=device)
            optimizer = optim.Adam([initial_guess], lr=0.3)

            # Optimization loop
            for _ in range(20):  # Number of epochs
                def closure():
                    optimizer.zero_grad()
                    loss = f(initial_guess)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                with torch.no_grad():
                    initial_guess.clamp_(0, 2 * m.pi)   # Ensures parameters constraints

            fitted_params = initial_guess
            results.append([-f(fitted_params).to('cpu').detach().numpy(), fitted_params.to('cpu').detach().numpy()])

        end_time = time.time()
        maximal_pair = max(enumerate(map(itemgetter(0), results)), key=itemgetter(1))
        item = results[maximal_pair[0]]
        violation_value = item[0]

        dict_ = {}
        dict_["Violation"] = str(violation_value)
        dict_["Angles"] = item[1].tolist()
        dict_["Circuit_Instructions"] = [str(i) for i in circuit.instructions]
        dict_["Depth"] = depth

        if rank == 0:
            print(f"Optimization time {int(end_time-start_time)} s. for {seeds} seeds", end='\r')

        with lock_list:  
            shared_list.append(dict_)
            

            """if violation_value > classical:
                batch_violations.append(dict_)
                with violation_lock:
                    total_violations.value += 1
            
            if violation_value > criterium:
                batch_high_violations.append(dict_)
                with violation_lock:
                    total_high_violations.value += 1"""

            # Controlla se è necessario salvare
            if (len(shared_list) % batch_size) == 0 and rank == 0:
                with lock_fSave:
                    save_to_tempfile(list(shared_list), f'Results/data_{N}.json')
                
    

if __name__ == "__main__":
    
    gpu = True
    N = 5                      # Number of qubits
    seeds = 5                  # Number of angle configurations for each quantum state
    n_attempts = 100           # Total number of state analysis attempts
    batch_size = 50             # used to save every 'save_batch_size' attempts
    numbers = list(range(n_attempts))

    rank = int(os.getenv('RANK', -1))
    num_workers = int(os.getenv('WORLD_SIZE', -1))


    if rank == 0:
        
        global_start_time = time.time()
        print("Quantum Limit: ", 2**(N-1)*np.sqrt(2))
        print("Our Criterium of high: ", 0.9*2**(N-1)*np.sqrt(2))
        print("Classical Limit: ", 2**(N-1))
        print()

        for r in range(num_workers):
            current_rank_attempts = partition_dataset(r, num_workers, numbers)
            print(f"Rank {r} is processing {current_rank_attempts}")

    manager = mp.Manager()
    shared_list = manager.list()
    list_lock = manager.Lock()
    fSave_lock = manager.Lock()

    processes = []

    
    for r in range(num_workers):
        current_rank_attempts = partition_dataset(r, num_workers, numbers)
        p = mp.Process(target=worker, args=(shared_list, list_lock, fSave_lock, rank, current_rank_attempts))
        processes.append(p)
        p.start()

    # Aspetta che tutti i worker finiscano
    for p in processes:
        p.join()

    
    if rank==0:
        global_end_time = time.time()
        print(f"\n\nTotal running time {int(global_end_time-global_start_time)} s.")
    
    
        print(f"Shared list content: {len(list(shared_list))}")
        save_to_tempfile(list(shared_list), f'Results/data_{N}.json')

        d = len(pd.read_json(f'Results/data_{N}.json'))
        print(f"Number of States in file: {d}")
