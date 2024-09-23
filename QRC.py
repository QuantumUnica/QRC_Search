from braket.circuits import Circuit, Gate, Noise, Observable, Instruction
from braket.devices import LocalSimulator
from braket.circuits.noise_model import (GateCriteria, NoiseModel,
                                         ObservableCriteria)
from braket.circuits.noises import (AmplitudeDamping, BitFlip, Depolarizing,
                                    PauliChannel, PhaseDamping, PhaseFlip,
                                    TwoQubitDepolarizing)
import random
import math as m

def QRC_Garnet(N, D, max_gates, connections, Clifford, noise_model=None):
    # Define gate names
    Gate_Names_1q = ['prx']  # One-qubit gate
    Gate_Names_2q = ['cz']   # Two-qubit gate
    #rotation = ['0', '1.57']  #INTENTO
    # If depth is not specified, choose a random depth
    if D is None:
        D = random.randint(41, 71)

    # Initialize the circuit
    circ = Circuit()

    gate_counts = {
        "prx": 0,
        "cz": 0
    }

    # Create bidirectional connections
    bidirectional_connections = set()
    for (q1, q2) in connections:
        bidirectional_connections.add((q1, q2))
        bidirectional_connections.add((q2, q1))

    # Iterate through the circuit depth
    for _ in range(D):
        L = []
        r = range(N)
        qubits_numbers = list(r)
        choices = []

        # Create choices based on max_gates
        if max_gates == 3:
            while 2 < len(qubits_numbers):
                a = random.randint(1, 3)
                b = random.sample(qubits_numbers, k=a)
                choices.append(b)
                for x in b:
                    qubits_numbers.remove(x)
            while 1 < len(qubits_numbers):
                a = random.randint(1, 2)
                b = random.sample(qubits_numbers, k=a)
                choices.append(b)
                for x in b:
                    qubits_numbers.remove(x)
            if 0 < len(qubits_numbers):
                choices = choices + [[qubits_numbers[0]]]
        elif max_gates == 2:
            while 1 < len(qubits_numbers):
                a = random.randint(1, 2)
                b = random.sample(qubits_numbers, k=a)
                choices.append(b)
                for x in b:
                    qubits_numbers.remove(x)
            if 0 < len(qubits_numbers):
                choices = choices + [[qubits_numbers[0]]]
        elif max_gates == 1:
            while 0 < len(qubits_numbers):
                b = random.sample(qubits_numbers, k=1)
                choices.append(b)
                for x in b:
                    qubits_numbers.remove(x)
        # Generate corresponding random gates
        for x in choices:
            if len(x) == 1:
                a = "."
                b = random.choice(Gate_Names_1q)
                #c = random.choice(rotation) #INTENTO
                gate_counts[b] += 1
                if Clifford == True:
                    if b == "prx":
                    # Apply prx gate with random angle
                        # angle1 = str(2 * (m.pi) * (random.random()))
                        # angle2 = str(2 * (m.pi) * (random.random()))
                        angle1 = str(m.pi/2)
                        angle2 = str(0)
                        quno = str(x[0])
                        c =  a + b + "(" + quno + "," + angle1 + "," + angle2 + ")" #INTENTO
                        #c =  a + b + "(" + quno + "," + angle + "," + '0' + ")"
                    L.append(c)
                else:
                    if b == "prx":
                    # Apply prx gate with random angle
                        angle1 = str(2 * (m.pi) * (random.random()))
                        angle2 = str(2 * (m.pi) * (random.random()))
                        quno = str(x[0])
                        c =  a + b + "(" + quno + "," + angle1 + "," + angle2 + ")" #INTENTO
                        #c =  a + b + "(" + quno + "," + angle + "," + '0' + ")"
                    L.append(c)
            elif len(x) == 2 and tuple(x) in bidirectional_connections:
                # Apply cz gate
                a = "."
                b = random.choice(Gate_Names_2q)
                gate_counts[b] += 1
                if b == 'cz':
                    # Add both directions
                    quno = str(x[0])
                    qdos = str(x[1])
                    c = a + b + "(" + quno + "," + qdos  + ")"
                    c = a + b + "(" + qdos + "," + quno  + ")"
                L.append(c)

        # Add instructions to the circuit
        for instr in L:
            eval("circ" + instr)

    circ_without_density = circ.copy()
#    noisy_circ = noise_model.apply(circ)
#    circ_noise_without_density = noisy_circ.copy()
    # Compute the density matrix for the ideal circuit

    backend= "braket_dm" if noise_model else "default"

    circ.density_matrix(target=range(N))
    device = LocalSimulator(backend=backend)
    task = device.run(circ, shots=0)
    result = task.result()
    RHO = result.values[0]

    # Compute the density matrix for the noisy circuit
    # noisy_circ.density_matrix(target=range(N))
    # noisy_task = device.run(noisy_circ, shots=0)
    # noisy_result = noisy_task.result()
    # RHO_noisy = noisy_result.values[0]

    dic = {
        "Ideal Circuit": circ,
#        "Noisy Circuit": noisy_circ,
#        "Noisy RHO": RHO_noisy,
        "Ideal RHO": RHO,
        "Depth": D,
        "Gate Counts": gate_counts,
        "Circuit Without Density Matrix": circ_without_density,
        # "Circuit Noise Without Density Matrix": circ_noise_without_density
    }

    return dic

