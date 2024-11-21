# Optimizing CD protocols

Optimization of the parameters of a protocol for continuous delivery (CD) of entanglement in a quantum network.

The code in this repository was used to generate the results shown in our paper 'Performance metrics for the continuous distribution of entanglement in multi-user quantum networks' ([arXiv:2307.01406](https://arxiv.org/abs/2307.01406)). The data shown in the paper can be found in our [data repository](http://doi.org/10.4121/75ccbe86-76dd-4188-8c34-f6e012b1373a.v1).

---
## Basic usage

To install our code as a Python package, run `pip install git+https://github.com/AlvaroGI/optimizing-cd-protocols.git@package`.

For a tutorial, see `TUTORIAL.ipynb`. 

For further details on the physical model, see below and also our [paper](https://arxiv.org/abs/2307.01406).

---
## Network assumptions

Some general assumptions:
 - Homogeneous network: every node and virtual link has the same properties (i.e., same probability of successful entanglement generation, same probability of successful swap, and same coherence time). This assumption simplifies the analysis but can be easily removed by modifying the code.
 - Arbitrary topology given by adjacency matrix.
 - Classical communication is *not* instantaneous.
 - Four basic operations:
	- GEN: heralded generation of entanglement between neighbors with probability $`p`$. Sequential attempts over a single physical channel.
	- SWAP: succeeds with probability $`p_\mathrm{s}`$.
	- CUT: remove entangled links older than $`t_\mathrm{cut}`$.
	- CONS: consume an entangled link in some application. When two nodes are virtual neighbors (i.e., they share at least one entangled link), they consume one entangled link per time step with probability $`p_\mathrm{cons}`$.
 - The maximum number of elementary links that can be swapped to form a longer link is $`M`$ (also called maximum entangled link length).
 - Each node has $`r`$ qubits to generate entanglement with each physical neighbor. Hence, each node can hold at most $`r \times \{\mathrm{number of physical neighbors}\}`$.

Here, we consider a basic CD protocol in which the nodes operate synchronously over discrete time slots. The protocol is parametrized by a single parameter. Our goal is to optimize the value of such parameter. Below, we describe the protocol in detail (see also our paper for a detailed discussion).
Note that our methods remain general and our code can be extended to more complex protocols.

---
## The SRS protocol

In the Single Random Swap (SRS) protocol, in each time slot:
 0. Existing links get older (i.e., add +1 to all link ages).
 1. CUT is applied to every link.
 2. GEN is attempted at every physical link (if qubits are available).
 3. SWAPS are performed as follows. Each node does the following (in the simulation, nodes do this in a random order):
 	1. Pick an occupied qubit that has not been picked in this time step, at random.
 	2. Pick an occupied qubit (that has not been picked in this time step) connected to a different node (that is not a physical neighbor of the original node), at random. If not possible, go to step 4.
 	3. With probability $`q`$, perform a swap on both qubits, which succeeds with probability $`p_\mathrm{s}`$.
 4. Classical communication: every node gains complete knowledge about the state of their qubits. (In the simulation we do not need to take any action).
 5. Remove links that are too long.
 6. CONS happens between each pair of virtual neighbors.

We also implemented a node-dependent SRS protocol (NDSRS), where the parameter $`q`$ is node-dependent. For further details, see `VALIDATION-node-dependent-srs.ipynb`.

---
## Files

 - `TUTORIAL.ipynb`: tutorial on how to run a simulation of a CD protocol and how to use our main functions.

 - `run_cd.py`: runs many realizations of the simulation for a specific set of parameters.

### Files that generate the results shown in the paper:
These files require some data to be generated in advance. The data can be generated locally or downloaded from our data repository (DOI: 10.4121/75ccbe86-76dd-4188-8c34-f6e012b1373a). If downloaded from the data repository, the data files must be placed in `\data-srs\avg\tree\`.

 - `MANUSCRIPT-noswaps.ipynb`: analysis of the performance metrics in the absence of swaps (analytical results). The file produces Figures 4, 10, and 11 from the paper.
 - `MANUSCRIPT-srs-tree.ipynb`: analysis of the performance of the SRS protocol in a (2,3)-tree. The file produces Figures 5, 7, 8, and 13 from the paper.
 - `MANUSCRIPT-srs-tree-Fapp-pcons.ipynb`: analysis of the influence of the fidelity requirements and the consumption rate on the SRS protocol performance in a (2,3)-tree. The file produces Figure 6 from the paper.
 - `MANUSCRIPT-srs-tree-variations.ipynb`: analysis of the performance of the SRS protocol in different trees. The file produces Figure 14 from the paper.
 - `MANUSCRIPT-data-blocking.ipynb`: here we explain why data blocking was not useful in our work (briefly discussed also in Appendix D of the paper).

### Other files:

 - `main-cd.py`: main functions.

 - `main-analytical.py`: main functions with the analytical results.

 - `validation/`: to run any of these tests, the outer directory should be added to the path (or simply take the validation file out of this folder).
   - `validation_GEN.ipynb`: checks that main.generate_all_links() works.
   - `validation_CUT.ipynb`: checks that main.advance_time() and main.cutoffs() work.
   - `validation_CONS.ipynb`: checks that main.consume_fixed_rate() works.
   - `validation_SWAP.ipynb`: checks that main.swap() works.
   - `validation_find-steady-state.ipynb`: checks that main.find_steady_state(), which finds the steady state of a function, works.









