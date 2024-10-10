import warnings
warnings.filterwarnings('ignore') 
import os
import json
import socket
import argparse
import tempfile
import time
import numpy as np
import random
import math as m
import pandas as pd
from operator import itemgetter

import torch
import torch.multiprocessing as mp
from torch import optim

from QRC import QRC_Device
import Inequalities 



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



def worker(shared_state_list, shared_vio, shared_hVio, 
           lock_states, lock_vio, lock_hVio, lock_states_fSave, lock_vio_fSave, lock_hVio_fSave, 
           rank, 
           attempts, 
           gpu, INEQ_TYPE, seeds, batch_size, circ_params, criterium, classical):

    """Worker function that appends a dictionary to the shared list."""
    # Creazione di un dizionario con dati da aggiungere alla lista

    device = torch.device("cuda", rank) if gpu else torch.device("cpu") # Assign correct device
    
    for i, _ in enumerate(attempts):
        if circ_params['D'] == None:
            circ_params['D'] = random.randint(10, 30) 

        random_circuit = QRC_Device(**circ_params)

        rho = torch.from_numpy(random_circuit['Ideal RHO']).type(torch.complex64).to(device)
        circuit = random_circuit["Circuit Without Density Matrix"]
        N = int(np.log2(np.shape(random_circuit['Ideal RHO'])[0]))

        results = []

        start_time = time.time()

        # Run seeds searches for each RHO
        for _ in range(seeds):

            def f(ang):
                angles = [ang[4 * i:4 * (i + 1)] for i in range(N)]
                v = Inequalities.get_expectation_value(INEQ_TYPE, rho, angles, device)
                return v

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
        dict_["Depth"] = circ_params['D']

        with lock_states:  
            shared_state_list.append(dict_)
            
            if (len(shared_state_list) % batch_size) == 0 and rank == 0:
                with lock_states_fSave:
                    save_to_tempfile(list(shared_state_list), f'Results/data_{N}.json')

        
        if violation_value > classical:
            with lock_vio:
                shared_vio.append(dict_)
            
            with lock_vio_fSave:
                save_to_tempfile(list(shared_vio), f'Results/violations_{N}.json')
        
        if violation_value > criterium:
            with lock_hVio:
                shared_hVio.append(dict_)
            
            with lock_hVio_fSave:
                save_to_tempfile(list(shared_hVio), f'Results/high_violations_{N}.json')


        if rank == 0 and (i % 2 == 0):  # Update counters for progress feedback
            with lock_states, lock_vio, lock_hVio:
                n_states = len(list(shared_state_list))
                n_shared_vio = len(list(shared_vio))
                n_shared_hVio = len(list(shared_hVio))

            print("Total states: {} Violations: {} High violations: {} - Optimization time {} s. for {} seeds"
                  .format(n_states, n_shared_vio, n_shared_hVio, int(end_time-start_time), seeds), end='\r')
            

    
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--q_dev", help="Select a quantum device", choices=['Clifford', 'Clifford+T', 'Aria1', 'Forte1', 'Garnet', 'Righetti'], default='Aria1')
    parser.add_argument("--depth", help="Circuit depth", type=int, default=None)
    parser.add_argument("--max_gates", help="Maximum number of gates per layer", type=int, default=2)
    parser.add_argument("--ineq", help="Select Svetlichny or Mermi ineq.", choices=['svet', 'mer'], default='svet')
    parser.add_argument("--n_qubits", help="Number of qubits", type=int, required=True)
    parser.add_argument("--n_attempts", help="Number of attempted searches", type=int, required=True)
    parser.add_argument("--n_seeds", help="Number of seeds for each attempt", type=int, default=5)
    parser.add_argument("--gpu", help="Use True to use the GPU", action='store_true')

    args=parser.parse_args()


    INEQ_TYPE = args.ineq    
    gpu = args.gpu

    N = args.n_qubits                        # Number of qubits
    seeds = args.n_seeds                               # Number of angle configurations for each quantum state
    n_attempts = args.n_attempts            # Total number of state analysis attempts
    batch_size = 50                         # used to save every 'save_batch_size' attempts
    
    global_attempts = list(range(n_attempts))

     # Definition of violation level criteria for non locality
    classical, svet_quantum_limit, mermin_limit = Inequalities.get_limits(N)
    criterium = 0.9 * svet_quantum_limit

    rank = int(os.getenv('RANK', -1))
    num_workers = int(os.getenv('WORLD_SIZE', -1))

    circ_params = {"device_name":args.q_dev, "N":N, "D":args.depth, "max_gates":args.max_gates}

    if rank == 0:     
        global_start_time = time.time()
        print("Svetlichny Quantum Limit: ", svet_quantum_limit)
        print("Mermin Limit: ", mermin_limit)
        print("Our Criterium of high: ", criterium)
        print("Classical Limit: ", classical)
        print()

        for r in range(num_workers):
            current_rank_attempts = partition_dataset(r, num_workers, global_attempts)
            print(f"Rank {r} is processing from {current_rank_attempts[0]} to {current_rank_attempts[-1]}")

    manager = mp.Manager()
    
    # Shared lists
    shared_states = manager.list()
    shared_vio = manager.list()
    shared_hVio = manager.list()
    
    # Lockers for shared lists
    state_lock = manager.Lock()
    vio_lock = manager.Lock()
    hVio_lock = manager.Lock()
    
    # Lockers for file saving
    states_fSave_lock = manager.Lock()
    vio_fSave_lock = manager.Lock()
    hVio_fSave_lock = manager.Lock()

    processes = []

    
    for r in range(num_workers):
        current_rank_attempts = partition_dataset(r, num_workers, global_attempts)
        p = mp.Process(target=worker, args=(shared_states, shared_vio, shared_hVio,
                                             state_lock, vio_lock, hVio_lock,
                                             states_fSave_lock, vio_fSave_lock, hVio_fSave_lock,
                                             rank, current_rank_attempts, gpu, 
                                             INEQ_TYPE, seeds, batch_size, circ_params, criterium, classical))
        processes.append(p)
        p.start()

    # Aspetta che tutti i worker finiscano
    for p in processes:
        p.join()

    
    if rank==0:
        global_end_time = time.time() - global_start_time
        total_execution_time, timeUnit = (global_end_time / 60, f" min.") if global_end_time >= 60 else (global_end_time, f" sec.")
        print(f"\nTotal running time: {round(total_execution_time, 2)} {timeUnit}\n")

        print(f"States: {len(list(shared_states))}")
        print(f"Violations: {len(list(shared_vio))}")
        print(f"High violation: {len(list(shared_hVio))}")
    
        save_to_tempfile(list(shared_states), f'Results/data_{N}.json')
        save_to_tempfile(list(shared_vio), f'Results/violations_{N}.json')
        save_to_tempfile(list(shared_vio), f'Results/high_violations_{N}.json')

        d = pd.read_json(f'Results/data_{N}.json')
        print(f"Number of States in file: {len(d)}")


if __name__ == "__main__":

    main()