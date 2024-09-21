import os
import json
from time import time
import numpy as np
import random
import math as m
from operator import itemgetter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim

from braket.circuits import Circuit, Gate, Noise, Observable, Instruction
from braket.devices import LocalSimulator
from braket.circuits.noise_model import (GateCriteria, NoiseModel, ObservableCriteria)
from braket.circuits.noises import (AmplitudeDamping, BitFlip, Depolarizing,
                                    PauliChannel, PhaseDamping, PhaseFlip, TwoQubitDepolarizing)

from QRC import QRC_Garnet


"""
    Multy GPU management
"""
# Setup for multi-node distributed training
def setup(rank, world_size):
    """
    Initialize the process group for distributed computation across multiple nodes.
    """
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())

def cleanup():
    """
    Cleanup the process group after computation.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def get_attempt_range(rank, world_size, n_attempts):
    """
    Determine the range of attempts to be processed by each device (including CPU if world_size = num_gpu+1).

    Parameters:
    - rank: The rank of the current process.
    - world_size: Total number of devices (GPUs + 1 for CPU).
    - n_attempts: Total number of attempts to be distributed.

    Returns:
    - start: The starting attempt index for this device.
    - end: The ending attempt index (exclusive) for this device.
    """
    attempts_per_device = n_attempts // world_size
    extra_attempts = n_attempts % world_size

    # Calculate start and end index for each device
    start = rank * attempts_per_device + min(rank, extra_attempts)
    end = start + attempts_per_device + (1 if rank < extra_attempts else 0)

    return start, end


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

def optimize(rank, world_size, n_attempts, saving_steps, N, seeds, shared_dictionaries, shared_high_violations, shared_violations, progress_counter, counter_lock):
    """
    Optimization function executed by a process.
    Each process will handle a different range of attempts.
    """
    setup(rank, world_size)
    device = torch.device("cuda", rank) if rank in range(torch.cuda.device_count()) else torch.device("cpu") # Assign correct device
    print(device)

    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9,0)] # Full connectivity

    # Get the range of attempts for this rank
    start_attempt, end_attempt = get_attempt_range(rank, world_size, n_attempts)
    print(f"Rank {rank} processing attempts from {start_attempt} to {end_attempt}")

    # Use shared lists to save dictionaries
    dictionaries = shared_dictionaries
    high_violations = shared_high_violations
    violations = shared_violations

    for j in range(start_attempt, end_attempt):
        depth = random.randint(10, 30)

        # Define noise model
        noise_model = NoiseModel()
        noise_model.add_noise(Depolarizing(0.00007), GateCriteria(Gate.H))
        noise_model.add_noise(Depolarizing(0.00007), GateCriteria(Gate.T))
        noise_model.add_noise(Depolarizing(0.0009), GateCriteria(Gate.CNot))

        random_circuit = QRC_Garnet(N, D=depth, noise_model=noise_model, max_gates=2, connections=connections, Clifford=True)

        RHO = torch.from_numpy(random_circuit['Ideal RHO']).type(torch.complex64).to(device)
        circuit = random_circuit["Circuit Without Density Matrix"]

        # Definition of violation level criteria
        criterium =0.9*2**(N-1)*np.sqrt(2)
        classical = 2**(N-1)

        results = []

        start_time = time()

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

        maximal_pair = max(enumerate(map(itemgetter(0), results)), key=itemgetter(1))
        item = results[maximal_pair[0]]
        violation_value = item[0]

        dict_ = {}
        dict_["Violation"] = str(violation_value)
        dict_["Angles"] = item[1].tolist()
        dict_["Circuit_Instructions"] = [str(i) for i in circuit.instructions]
        dict_["Depth"] = depth

        # Add element to the shared list
        dictionaries.append(dict_)

        if violation_value > classical:
            violations.append(dict_)
            with open(f'Results/violations_{N}', 'w') as fout:
                    json.dump(list(violations), fout)

        if violation_value > criterium:
            high_violations.append(dict_)
            with open(f'Results/high_violations_{N}', 'w') as fout:
                json.dump(list(high_violations), fout)

        if progress_counter.value+1 in saving_steps:
            with open(f'Results/data_{N}', 'w') as fout:
                json.dump(list(dictionaries), fout)


        # Increment the shared attempts progress counter
        with counter_lock:
            progress_counter.value += 1

        print(f"Violations: {len(violations)} - High violations: {len(high_violations)} - {progress_counter.value}/{n_attempts} attempts. Optimization time {int(time()-start_time)} s. for {seeds} seeds", end='\r')

    cleanup()


if __name__ == "__main__":

    # Check if exists, or initialize environment variables needed by torch.distributed

    world_size = int(os.getenv('WORLD_SIZE', torch.cuda.device_count()))    # Total number of GPUs/processes across all nodes
    rank = int(os.getenv('RANK', 0))                                        # Rank of the current process across all nodes
    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', '12355')

    # Ensure torch.distributed gets these environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    N = 10            # Number of qubits
    seeds = 10          # Number of angle configurations for each quantum state
    NATTEMPTS = 35000      # Total number of state analisys attempts

    # Violation criteria (also )
    print("Quantum Limit: ", 2**(N-1)*np.sqrt(2))
    print("Our Criterium of high: ", 0.9*2**(N-1)*np.sqrt(2))
    print("Classical Limit: ", 2**(N-1))
    print()

    save_rate = 10     # used to save every 'save_rate' attempts
    saving_steps = [x for x in range(0, NATTEMPTS, save_rate)]     # Iterention's ids in which to save JSON oput files

    # Manager for shared lists
    with mp.Manager() as manager:
        # Liste condivise tra i worker
        shared_dictionaries = manager.list()
        shared_high_violations = manager.list()
        shared_violations = manager.list()

        progress_counter = manager.Value('i', 0)  # Integer counter, initialized to 0
        counter_lock = manager.Lock()  # Create a lock for the counter

        overall_start_time = time()

        mp.spawn(optimize, args=(world_size, NATTEMPTS, saving_steps, N, seeds,
                                 shared_dictionaries, shared_high_violations, shared_violations, progress_counter, counter_lock),
                 nprocs=world_size, join=True)    # Starts parallel optimization

        print(f"\n\nTotal dictionaries: {len(shared_dictionaries)}")
        print(f"Total violations: {len(shared_violations)}")
        print(f"Total high violations: {len(shared_high_violations)}")

        endTime = (time()-overall_start_time)
        endTime, timeUnit = (endTime/60, f" min.") if endTime >= 60 else (endTime,f" sec.")

        print(f"\nTotal execution time: {round(endTime, 2 ) } {timeUnit}")

