import main_cd as cd
import numpy as np

if __name__ == '__main__':
    ## PROTOCOL
    protocol = 'srs'  # Currently only 'srs' has been debugged

    ## TOPOLOGY
    # Use any function main.adjacency_*() to define a topology.
    # Here we use a squared lattice (with hard boundary conditions)
    # with 9 nodes as an example.
    l = 3
    n = int(l * l)
    A = cd.adjacency_squared_hard(l)
    topology = 'squared_hard'

    ## HARDWARE
    p_gen = 1  # Probability of successful entanglement generation
    p_swap = 1  # Probability of successful swap
    qbits_per_channel = 1  # Number of qubits per node per physical neighbor

    ## SOFTWARE
    q_swap = 0.12  # Probability of performing swaps in the SRS protocol
    max_links_swapped = 4  #  Maximum number of elementary links swapped
    p_cons = 0  # Probability of virtual neighbors consuming a link per time step
    F_app = 0  # Minimum fidelity required by the background application

    ## CUTOFF
    # The cutoff is here chosen arbitrarily. To find a physically meaningful value,
    # one should use the coherence time of the qubits and the fidelity of newly
    # generated entangled links. A valid approach is to use a worst-case model
    # as in Iñesta et al. 'Optimal entanglement distribution policies in homogeneous
    # repeater chains with cutoffs', 2023.
    cutoff = 20

    ## SIMULATION
    data_type = 'avg'  # Store only average (and std) values instead of all simulation data
    N_samples = 1  #  Number of samples
    total_time = int(cutoff * 5)  #  Simulation time
    plot_nodes = [0, 4, 5]  # We will plot the time evolution of these nodes
    randomseed = 0
    np.random.seed(randomseed)
    cd.simulation_cd(protocol, A, p_gen, q_swap, p_swap,
                  p_cons, cutoff, max_links_swapped,
                  qbits_per_channel, N_samples,
                  total_time,
                  return_data=data_type)