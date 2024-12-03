import random 
import math as m

from braket.circuits import Circuit, Gate
from braket.devices import LocalSimulator
from braket.circuits.noise_model import (GateCriteria, NoiseModel)
from braket.circuits.noises import (Depolarizing)


# Dictionary storing quantum device information
devices_info = {
    "Clifford": {
        # List of available single-qubit gates
        "gate_names_1q": ['h', 's', 'i'], 
        # List of available two-qubit gates
        "gate_names_2q": ['cnot'],  
        # No connectivity restrictions for Clifford devices
        "connectivity": None,
        # Noise model for Clifford device
        "noise_model": NoiseModel()
                         .add_noise(Depolarizing(0.00025), GateCriteria(Gate.H))
                         .add_noise(Depolarizing(0.00025), GateCriteria(Gate.T))
                         .add_noise(Depolarizing(0.0045), GateCriteria(Gate.CNot))
    },
    
    "Clifford+T": {
        "gate_names_1q": ['h', 's', 't', 'i'],
        "gate_names_2q": ['cnot'],
        "connectivity": None,
        "noise_model": NoiseModel()
                         .add_noise(Depolarizing(0.00025), GateCriteria(Gate.H))
                         .add_noise(Depolarizing(0.00025), GateCriteria(Gate.T))
                         .add_noise(Depolarizing(0.0045), GateCriteria(Gate.CNot))
    },
    
    "Aria1": {
        "gate_names_1q": ['gpi', 'gpi2'],  # IonQ-specific single-qubit gates
        "gate_names_2q": ['ms'],  # IonQ-specific multi-qubit gate
        "connectivity": None,  # No specific connectivity required
        "noise_model": NoiseModel()
                         .add_noise(Depolarizing(0.0002), GateCriteria(Gate.MS))
                         .add_noise(Depolarizing(0.002), GateCriteria(Gate.GPi))
                         .add_noise(Depolarizing(0.002), GateCriteria(Gate.GPi2))
    },
    
    "Forte1": {
        "gate_names_1q": ['gpi', 'gpi2'],
        "gate_names_2q": ['zz'],  # ZZ interaction gate
        "connectivity": None,
        "noise_model": NoiseModel()
                         .add_noise(Depolarizing(0.0002), GateCriteria(Gate.ZZ))
                         .add_noise(Depolarizing(0.002), GateCriteria(Gate.GPi))
                         .add_noise(Depolarizing(0.002), GateCriteria(Gate.GPi2))
    },
    
    "Garnet": {
        "gate_names_1q": ['prx'],  # Single-qubit gate specific to Garnet device
        "gate_names_2q": ['cz'],  # Controlled-Z two-qubit gate
        # Connectivity map indicating allowed qubit connections
        "connectivity": None,
        # Noise model for Garnet device
        "noise_model": NoiseModel()
                         .add_noise(Depolarizing(0.004), GateCriteria(Gate.PRx))
                         .add_noise(Depolarizing(0.0002), GateCriteria(Gate.CZ))
    },
    
    "Rigetti": {
        "gate_names_1q": ['rx', 'rz'],  # Rigetti's single-qubit gates
        "gate_names_2q": ['iswap'],  # Two-qubit ISWAP gate
        # Connectivity map defining the qubit connections for Rigetti device
        "connectivity": {0: [1, 7], 1: [0, 2, 8], 2: [1, 3, 9], 3: [2, 4, 10], 
                         4: [3, 5, 11], 5: [4, 6, 12], 6: [5, 13], 7: [0, 8, 14], 
                         8: [1, 7, 9, 15], 9: [2, 8, 10, 16], 10: [3, 9, 11, 17], 
                         11: [4, 10, 12, 18], 12: [5, 11, 13, 19], 13: [6, 12, 20], 
                         14: [7, 15, 21], 15: [8, 14, 22], 16: [9, 17, 23], 
                         17: [10, 16, 18, 24], 18: [11, 17, 19, 25], 19: [12, 18, 20, 26], 
                         20: [13, 19, 27], 21: [14, 22, 28], 22: [15, 21, 23, 29], 
                         23: [16, 22, 24, 30], 24: [17, 23, 25, 31], 25: [18, 24, 26, 32], 
                         26: [19, 25, 33], 27: [20, 34], 28: [21, 29, 35], 29: [22, 28, 30, 36], 
                         30: [23, 29, 31, 37], 31: [24, 30, 32, 38], 32: [25, 31, 33, 39], 
                         33: [26, 32, 34, 40], 34: [27, 33, 41], 35: [28, 36, 42], 
                         36: [29, 35, 37, 43], 37: [30, 36, 38, 44], 38: [31, 37, 39, 45], 
                         39: [32, 38, 40, 46], 40: [33, 39, 41, 47], 41: [34, 40, 48], 
                         42: [35, 43, 49], 43: [36, 42, 44, 50], 44: [37, 43, 45, 51], 
                         45: [38, 44, 46, 52], 46: [39, 45, 47, 53], 47: [40, 46, 48, 54], 
                         48: [41, 47, 55], 49: [42, 50], 50: [43, 49, 51], 
                         51: [44, 50, 52], 52: [45, 51, 53], 53: [46, 52, 54], 
                         54: [47, 53, 55], 55: [48, 54]},
        "noise_model": NoiseModel()
                         .add_noise(Depolarizing(0.02), GateCriteria(Gate.Rx))
                         .add_noise(Depolarizing(0.002), GateCriteria(Gate.Rz))
                         .add_noise(Depolarizing(0.002), GateCriteria(Gate.ISwap))
    },
}

# Función para cargar la información de un dispositivo
def load_device_info(device_name):
    if device_name not in devices_info:
        raise ValueError(f"Device {device_name} not found")
    
    return devices_info[device_name]["gate_names_1q"], devices_info[device_name]["gate_names_2q"], devices_info[device_name].get("connectivity"), devices_info[device_name].get("noise_model")



#######################################
##### Quantum Random Circuit Generator #####
##### Device-Specific Implementation  #####
#######################################

def QRC_Device(device_name, N, D, max_gates):
    """
    Generates a random quantum circuit for a specified quantum device, considering
    its gate set, qubit connectivity, and noise model.

    Args:
        device_name (str): The name of the quantum device to target (e.g., 'IonQ', 'Garnet').
        N (int): The number of qubits for the circuit.
        D (int, optional): The depth of the circuit (number of layers). If None, a random depth is chosen.
        max_gates (int): Maximum number of gates applied per iteration (1 or 2-qubit gates).

    Returns:
        dict: A dictionary containing the ideal and noisy circuits, density matrices, gate counts, and other details.
    """
    
    # Load device-specific information: 1-qubit and 2-qubit gate sets, connectivity, and noise model
    gate_names_1q, gate_names_2q, connectivity, noise_model = load_device_info(device_name)

    # If depth D is not specified, choose a random value between 5 and 12
    if D is None:  
        D = random.randint(5, 12)

    # Initialize the quantum circuit
    circ = Circuit()
    
    # Dictionary to count how many times each 2-qubit gate is used
    gate_counts = {
        "ms": 0,
        "cz": 0,
        "iswap": 0,
        "cnot": 0,
        "zz": 0
    }

    # Process device connectivity and create a bidirectional map (only if connectivity exists)
    if connectivity is not None:
        bidirectional_connections = {}
        for key, values in connectivity.items():
            bidirectional_connections[int(key)] = [int(v) for v in values]

    # Iterate through the layers (depth) of the circuit
    for _ in range(D):
        L = []  # List of operations to apply in this layer
        qubits_numbers = list(range(N))  # Initialize list of available qubits
        choices = []  # List to hold selected gate applications

        while len(qubits_numbers) > 0:
            # Decide randomly whether to apply a one-qubit or two-qubit gate
            if max_gates > 1 and len(qubits_numbers) > 1:
                use_two_qubit_gate = random.choice([True, False])
                
                if use_two_qubit_gate:
                    # Randomly sample two qubits for a 2-qubit gate
                    pair = random.sample(qubits_numbers, 2)
                    q1, q2 = pair[0], pair[1]
                    
                    # Validate connectivity before applying a 2-qubit gate
                    if connectivity is None or (q2 in bidirectional_connections.get(q1, []) or q1 in bidirectional_connections.get(q2, [])):
                        choices.append(pair)
                        qubits_numbers.remove(pair[0])
                        qubits_numbers.remove(pair[1])
                    else:
                        # If no valid connection, fallback to a 1-qubit gate
                        choices.append([qubits_numbers.pop(0)])
                else:
                    # Apply a 1-qubit gate
                    choices.append([qubits_numbers.pop(0)])
            else:
                # If only 1-qubit gates are allowed or not enough qubits for a 2-qubit gate
                choices.append([qubits_numbers.pop(0)])

        # Construct the circuit layer with selected gates
        for qubit_set in choices:
            if len(qubit_set) == 1:  # Apply 1-qubit gate
                qubit = str(qubit_set[0])
                gate = random.choice(gate_names_1q)  # Randomly select a 1-qubit gate

                if gate in {"gpi", "gpi2", "rx", "rz", "prx"}:  # Gates requiring angle parameters
                    angle1 = str(2 * m.pi * (random.random()))
                    angle2 = str(2 * m.pi * (random.random())) if gate == "prx" else ""
                    command = f".{gate}({qubit},{angle1}{','+angle2 if angle2 else ''})"
                else:
                    command = f".{gate}({qubit})"
                
                L.append(command)

            elif len(qubit_set) == 2:  # Apply 2-qubit gate
                q1, q2 = qubit_set
                if connectivity is None or (q1 in bidirectional_connections.get(q2, []) or q2 in bidirectional_connections.get(q1, [])):
                    gate = random.choice(gate_names_2q)  # Randomly select a 2-qubit gate
                    gate_counts[gate] += 1  # Update the gate usage counter

                    if gate == "ms":  # MS gate requires three angle parameters
                        angle1, angle2, angle3 = (str((m.pi / 2) * random.random()) for _ in range(3))
                        command = f".{gate}({q1},{q2},{angle1},{angle2},{angle3})"
                    
                    elif gate == "zz":  # ZZ gate requires one angle parameter
                        angle1 = str((m.pi / 2) * random.random())
                        command = f".{gate}({q1},{q2},{angle1})"
                    
                    else:  # For gates like 'cz', 'cnot', 'iswap'
                        command = f".{gate}({q1},{q2})"
                    
                    L.append(command)

        # Apply the constructed layer of gates to the circuit
        for operation in L:
            eval("circ" + operation)

    # Optional: apply the device's noise model to the circuit (if it exists)
    noisy_circ = noise_model.apply(circ) if noise_model else circ
    circ_without_density = circ.copy()  # Save a copy of the circuit before density matrix conversion
    
    # Convert the circuit to a density matrix form for measurement
    circ.density_matrix(target=range(N))
    noisy_circ.density_matrix(target=range(N))

    # Execute both ideal and noisy circuits on a local simulator (using density matrices)
    device = LocalSimulator(backend="braket_dm")

    # Run ideal circuit simulation
    task = device.run(circ, shots=0)
    result = task.result()
    RHO = result.values[0]  # Extract the ideal state density matrix

    # Run noisy circuit simulation
    noisy_task = device.run(noisy_circ, shots=0)
    noisy_result = noisy_task.result()
    RHO_noisy = noisy_result.values[0]  # Extract the noisy state density matrix

    # Return all relevant information, including the gate counts and circuit depth
    return {
        "Ideal Circuit": circ,
        "Noisy Circuit": noisy_circ,
        "Ideal RHO": RHO,
        "Noisy RHO": RHO_noisy,
        "Gate Counts": gate_counts,
        "Depth": D,
        "Circuit Without Density Matrix": circ_without_density
    }
