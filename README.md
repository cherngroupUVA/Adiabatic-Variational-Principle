# Exact Simulations of the Spinless t–V Model

This repository contains exact numerical simulation code and data for studying charge-density-wave (CDW) ordering and domain-coarsening dynamics in the spinless t–V model on both square and triangular lattices.

The simulations evolve the local charge-density configuration and compute the corresponding thermodynamic forces from the microscopic electronic model. The resulting data can be used as exact-simulation benchmarks for analyzing CDW dynamics and for comparison with machine-learning models.

## Repository Structure

```text
.
├── code/
│   ├── square-lattice.jl
│   ├── triangle-lattice.jl
│   ├── lattice.csv
│   └── cdw-pattern.csv
│
├── data/
│   ├── square_lattice/
│   │   └── SquareLattice.zip
│   │
│   └── triangle_lattice/
│       └── TriangularLattice.zip
│
└── README.md
```

## Code

### `code/square-lattice.jl`

Exact simulation of the spinless t–V model on a square lattice.

The square-lattice implementation uses four nearest neighbors for each site:

- left,
- right,
- top,
- bottom.

Periodic boundary conditions are applied in both spatial directions.

### `code/triangle-lattice.jl`

Exact simulation of the spinless t–V model on a triangular lattice.

This implementation does not hard-code the triangular-lattice connectivity. Instead, it reads the lattice connectivity from

```text
lattice.csv
```

and the CDW initialization pattern from

```text
cdw-pattern.csv
```

The number of lattice sites and the coordination number are inferred directly from `lattice.csv`.

---

## Requirements

The simulations are written in Julia and use GPU acceleration through CUDA.

Required Julia packages are

```julia
using CUDA
using DelimitedFiles
using LinearAlgebra
using Random
```

A CUDA-compatible NVIDIA GPU and a working CUDA.jl installation are required.

---

# Square-Lattice Simulation

## Default Parameters

The default parameters used in `square-lattice.jl` are

```julia
dim   = 64      # lattice size: 64 × 64
fil   = 0.5     # filling factor
kT    = 2e-1    # electronic temperature
tnn   = 2.0     # nearest-neighbor hopping
vnn   = 1.0     # nearest-neighbor repulsion
W     = 0.0     # disorder strength

stp   = 5e-0    # step size for convergence
dt    = 5e-3    # time step for order-parameter dynamics
```

The system size is therefore

```text
64 × 64 = 4096 sites.
```

The simulation is performed at half filling.

## Initialization

The initial charge-density configuration is generated from a random onsite potential.

For a target charge-density configuration, the corresponding onsite potential is obtained iteratively. The conjugate thermodynamic force is then calculated from the converged solution.

A self-consistent calculation is first used to determine the equilibrium chemical potential and CDW order parameter.

## Dynamics

The main dynamical loop runs for

```text
10,000 steps
```

The density configuration is updated according to the calculated thermodynamic force together with a small random perturbation.

The density and force are stored every 10 dynamical steps.

## Running the Square-Lattice Simulation

The program takes one integer command-line argument. This integer is used as

1. the CUDA random seed, and
2. the simulation-run index used in the output filenames.

For example,

```bash
julia square-lattice.jl 1
```

generates

```text
density-1.txt
force-1.txt
```

To save the output directly into the square-lattice data directory, one convenient workflow is

```bash
cd data/square_lattice
julia ../../code/square-lattice.jl 1
```

Independent runs can be generated using different seeds:

```bash
julia ../../code/square-lattice.jl 1
julia ../../code/square-lattice.jl 2
julia ../../code/square-lattice.jl 3
```

## Square-Lattice Output

### `density-*.txt`

Contains the local electron density at each saved simulation snapshot.

For the current lattice,

```text
N = 64 × 64 = 4096
```

density values are stored per snapshot.

The data can be reshaped for analysis as

```python
(num_snapshots, 64 * 64)
```

### `force-*.txt`

Contains the thermodynamic force corresponding to the density configurations in the matching `density-*.txt` file.

For example,

```text
density-1.txt
force-1.txt
```

belong to the same independent simulation run.

---

# Triangular-Lattice Simulation

## Lattice Connectivity Input

The triangular-lattice program explicitly imports

```text
lattice.csv
```

through

```julia
lat = CuArray(readdlm("lattice.csv", ',', Int64))
```

Each row of `lattice.csv` corresponds to one lattice site, and each entry in that row gives the index of one connected neighboring site.

The supplied `lattice.csv` has

```text
3969 rows × 6 columns
```

so the program interprets it as

```text
nSt  = 3969 lattice sites
ncor = 6 nearest neighbors per site
dim  = sqrt(3969) = 63
```

Therefore, the supplied connectivity file corresponds to a `63 × 63` triangular lattice with coordination number 6.

The program determines these quantities automatically using

```julia
ncor = size(lat)[2]
nSt  = size(lat)[1]
dim  = Int64(sqrt(nSt))
```

This connectivity table is then used when constructing the hopping Hamiltonian, evaluating the nearest-neighbor interaction contribution, and calculating the thermodynamic force.

## CDW Pattern Input

The triangular-lattice program also explicitly imports

```text
cdw-pattern.csv
```

through

```julia
cdw = CuArray(readdlm("cdw-pattern.csv", Int64))
```

The supplied `cdw-pattern.csv` contains

```text
3969 entries
```

which matches the 3969 lattice sites defined by `lattice.csv`.

The entries take the values

```text
0, 1, 2
```

and form a repeating three-sublattice pattern on the `63 × 63` triangular lattice. When reshaped into the lattice geometry, the pattern begins as

```text
0 1 2 0 1 2 ...
2 0 1 2 0 1 ...
1 2 0 1 2 0 ...
0 1 2 0 1 2 ...
...
```

The Julia program uses this file to initialize the density through

```julia
nn[idx] = fil * cdw[idx]
```

with

```julia
fil = 1/3
```

so the corresponding initial density values are

```text
0, 1/3, 2/3
```

on the three sublattices.

Thus, `cdw-pattern.csv` specifies the three-sublattice CDW reference pattern used to initialize the self-consistent triangular-lattice calculation.

Both `lattice.csv` and `cdw-pattern.csv` must be available in the working directory when `triangle-lattice.jl` is executed.

## Default Parameters

The default parameters used in `triangle-lattice.jl` are

```julia
fil   = 1/3     # filling factor
kT    = 4e-1    # electronic temperature
tnn   = 1.0     # nearest-neighbor hopping
vnn   = 2.0     # nearest-neighbor repulsion
W     = 0.0     # disorder strength

stp   = 8e-0    # step size for convergence
dt    = 1e-2    # time step for order-parameter dynamics
```

For the supplied `lattice.csv`, the simulation size is

```text
63 × 63 = 3969 sites.
```

## Chemical Potential and Self-Consistent Calculation

The triangular-lattice implementation determines the chemical potential numerically for the target filling.

A self-consistent calculation is then used to obtain the equilibrium charge-density configuration and CDW order parameter before the dynamical simulation starts.

## Dynamics

The main dynamical loop runs for

```text
10,000 steps.
```

In the current triangular-lattice implementation, the target density is updated according to the calculated thermodynamic force:

```julia
target += lnn * dt
```

The density, force, and single-particle energy spectrum are stored every 100 dynamical steps.

## Running the Triangular-Lattice Simulation

Because the program reads `lattice.csv` and `cdw-pattern.csv` from the current working directory, a convenient organization is to keep all four files together inside `code/`:

```text
code/
├── triangle-lattice.jl
├── lattice.csv
└── cdw-pattern.csv
```

Then run

```bash
cd code
julia triangle-lattice.jl 1
```

This generates files such as

```text
density-1.txt
force-1.txt
dos-1.txt
```

The resulting files can then be moved to

```text
data/triangle_lattice/
```

Independent runs can be generated using different random seeds:

```bash
julia triangle-lattice.jl 1
julia triangle-lattice.jl 2
julia triangle-lattice.jl 3
```

## Triangular-Lattice Output

### `density-*.txt`

Contains the local electron density for each saved snapshot.

For the supplied lattice, each configuration contains

```text
3969 density values.
```

### `force-*.txt`

Contains the corresponding thermodynamic force for each saved density configuration.

### `dos-*.txt`

Contains the single-particle eigenvalue spectrum stored together with the density and force snapshots.

For a given run index,

```text
density-1.txt
force-1.txt
dos-1.txt
```

belong to the same simulation run.

---

# Random Seeds and Independent Runs

Both simulation programs use the first command-line argument as the CUDA random seed:

```julia
CUDA.seed!(parse(Int, ARGS[1]))
```

Different integer arguments therefore generate independent simulations.

The same integer is also used in the output filenames, which makes it straightforward to associate each data file with the corresponding simulation run.

---

# Data Organization

Simulation results are organized according to lattice geometry:

```text
data/
├── square_lattice/
│   └── square-lattice exact simulation data
│
└── triangle_lattice/
    └── triangular-lattice exact simulation data
```

This separation makes subsequent analysis, visualization, and machine-learning benchmarking easier.

---

# Typical Workflow

A typical workflow is

1. Select the lattice geometry.
2. For the triangular lattice, make sure `lattice.csv` and `cdw-pattern.csv` are available in the working directory.
3. Select a random seed for an independent simulation run.
4. Run the corresponding Julia program.
5. Store the generated data in the corresponding lattice folder.
6. Reshape and analyze the density configurations.
7. Construct the appropriate CDW order parameter if needed.
8. Compute correlation functions, characteristic length scales, or other observables.
9. Compare the exact simulation with machine-learning predictions.

---

# Notes

- `square-lattice.jl` implements the square-lattice t–V simulation.
- `triangle-lattice.jl` implements the triangular-lattice t–V simulation.
- The square-lattice simulation uses four nearest neighbors with periodic boundary conditions.
- The triangular-lattice simulation reads its connectivity from `lattice.csv`.
- `cdw-pattern.csv` contains a 3969-site three-sublattice pattern with labels `0`, `1`, and `2`, used to initialize the triangular-lattice self-consistent calculation.
- The supplied triangular-lattice connectivity file contains 3969 sites with 6 neighbors per site, corresponding to a `63 × 63` triangular lattice.
- The square-lattice simulation uses filling `1/2`.
- The triangular-lattice simulation uses filling `1/3`.
- Both programs run for 10,000 dynamical steps in their current form.
- Square-lattice density and force snapshots are stored every 10 steps.
- Triangular-lattice density, force, and energy-spectrum snapshots are stored every 100 steps.
- Different runs are distinguished by the integer random seed passed through the command line.

---

# Reference

This repository contains exact simulation code and data associated with the study of charge-density-wave dynamics in the spinless t–V model.

If you use this code or the accompanying data in research, please cite the corresponding manuscript/publication.
