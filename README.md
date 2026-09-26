# Exact Simulations of the Spinless t–V Model

This repository contains exact numerical simulation code and a small sample of simulation data for studying charge-density-wave (CDW) ordering and domain-coarsening dynamics in the spinless t–V model on square and triangular lattices.

The full simulation data sets are much larger than what is included here. To keep the repository compact, only **three independent runs** are provided for each lattice geometry as representative samples.

## Repository Structure

```text
.
├── code/
│   ├── square-lattice simulation code
│   ├── triangle-lattice simulation code
│   ├── lattice.csv
│   └── cdw-pattern.csv
│
├── data/
│   ├── square_lattice/
│   │   └── 11-part compressed archive for 3 independent runs
│   │
│   └── triangle_lattice/
│       └── data from 3 independent runs
│
└── README.md
```

## Simulation Code

The Julia programs perform exact simulations of the spinless t–V model and generate the local charge-density configurations and their corresponding thermodynamic forces.

The simulations use

```julia
using CUDA
using DelimitedFiles
using LinearAlgebra
using Random
```

and therefore require Julia, CUDA.jl, and a CUDA-compatible NVIDIA GPU.

---

# Square-Lattice Simulation

The square-lattice simulation uses a `64 × 64` lattice with periodic boundary conditions and four nearest neighbors per site.

## Parameters

The current square-lattice calculation uses

```julia
dim   = 64      # system size: 64 × 64
fil   = 0.5     # filling factor
kT    = 2e-1    # electronic temperature
tnn   = 2.0     # nearest-neighbor hopping
vnn   = 1.0     # nearest-neighbor repulsion
W     = 0.0     # disorder strength

stp   = 5e-0    # convergence step
dt    = 5e-3    # time step for order-parameter dynamics
```

The main dynamical simulation runs for 10,000 steps. Density and force snapshots are stored every 10 dynamical steps.

The initial charge-density configuration is generated from a random onsite potential. For each target density configuration, the corresponding onsite potential is determined iteratively, and the thermodynamic force is obtained from the converged solution.

## Output

The square-lattice simulation generates

```text
density-*.txt
force-*.txt
```

where `*` denotes the simulation-run index.

Each density snapshot contains

```text
64 × 64 = 4096
```

site-resolved charge-density values.

The corresponding `force-*.txt` file uses the same site ordering and saved time steps.

## Sample Data Included in This Repository

Only **three independent square-lattice runs** are included as sample data.

Because these files are still relatively large, the three runs have been compressed into a **split archive containing 11 compressed parts**.

All 11 parts must be kept together in the same directory before extraction. The archive should be extracted starting from

```text
square_lattice.zip
```

using an archive program that supports split ZIP files.

After extraction, the square-lattice sample data contain the corresponding `density-*.txt` and `force-*.txt` files for the three retained independent runs.

The compressed archive is provided only as a compact sample of the exact simulation data; it is not the complete square-lattice data set.

---

# Triangular-Lattice Simulation

The triangular-lattice simulation uses a `63 × 63` lattice and a three-sublattice CDW pattern.

The reference simulation setup uses

```julia
dim     = 63     # system size: 63 × 63
tnn     = 1.0    # nearest-neighbor hopping
vnn     = 2.0    # nearest-neighbor repulsion
dt      = 1e-2   # time step
kT      = 4e-1   # electronic temperature
filling = 1/3    # filling factor
```

The triangular-lattice implementation reads its lattice connectivity and CDW initialization pattern from two input files:

```text
lattice.csv
cdw-pattern.csv
```

## `lattice.csv`

The supplied `lattice.csv` contains the connectivity information for the triangular lattice.

It has

```text
3969 rows × 6 columns
```

corresponding to

```text
63 × 63 = 3969 lattice sites
```

with six nearest neighbors per site.

The Julia program reads this file with

```julia
lat = CuArray(readdlm("lattice.csv", ',', Int64))
```

and obtains the system size and coordination number from the dimensions of the imported array.

## `cdw-pattern.csv`

The supplied `cdw-pattern.csv` contains 3969 entries and defines the three-sublattice CDW reference pattern used to initialize the self-consistent calculation.

The entries take the values

```text
0, 1, 2
```

in a repeating three-sublattice pattern.

The Julia program reads this file and initializes the density according to

```julia
nn[idx] = fil * cdw[idx]
```

with

```julia
fil = 1/3
```

so the initial density values on the three sublattices are

```text
0, 1/3, 2/3
```

## Output

The triangular-lattice simulation generates files such as

```text
density-*.txt
force-*.txt
dos-*.txt
```

The `density-*.txt` files contain the site-resolved charge density, and the corresponding `force-*.txt` files contain the thermodynamic forces using the same ordering.

The `dos-*.txt` files contain the single-particle eigenvalue spectrum stored during the simulation.

## Sample Data Included in This Repository

Only **three independent triangular-lattice runs** are included as sample data.

Because the triangular-lattice sample is small enough for direct storage in the repository, these files are uploaded directly under

```text
data/triangle_lattice/
```

without additional compression.

These three runs are representative examples only and do not constitute the complete triangular-lattice simulation data set.

---

# Random Seeds and Independent Runs

The simulation programs use the first command-line argument as the CUDA random seed:

```julia
CUDA.seed!(parse(Int, ARGS[1]))
```

Different integer seeds therefore generate independent simulation runs.

The same run index is also used in the output filenames, for example

```text
density-1.txt
force-1.txt
```

for one run and

```text
density-2.txt
force-2.txt
```

for another.

---

# Data Format

## `density-*.txt`

Contains the local electron density at the saved simulation times.

For a lattice with `N` sites, each saved configuration contains `N` density values.

For the square lattice,

```text
N = 4096
```

and for the triangular lattice,

```text
N = 3969.
```

The data can be reshaped during analysis into the form

```python
(num_snapshots, num_sites)
```

## `force-*.txt`

Contains the thermodynamic force corresponding to each saved density configuration.

A matching pair such as

```text
density-1.txt
force-1.txt
```

belongs to the same simulation run and uses the same snapshot and site ordering.

## `dos-*.txt`

For the triangular-lattice simulation, these files contain the single-particle eigenvalue spectrum saved together with the density and force data.

---

# Typical Workflow

A typical workflow is:

1. Select either the square- or triangular-lattice simulation.
2. Choose a random seed for an independent run.
3. For the triangular lattice, ensure that `lattice.csv` and `cdw-pattern.csv` are available in the working directory.
4. Run the corresponding Julia simulation.
5. Store the generated files in the appropriate data directory.
6. Reshape the density configurations for analysis.
7. Construct the corresponding CDW order parameter if needed.
8. Calculate quantities such as correlation functions and characteristic length scales.
9. Compare the exact simulations with machine-learning predictions or other dynamical simulations.

---

# Notes on the Included Data

The data in this GitHub repository are intended as **small representative samples** of the exact simulations.

- Three independent runs are provided for the square lattice.
- Three independent runs are provided for the triangular lattice.
- The square-lattice samples are distributed as an 11-part split ZIP archive because of their larger file sizes.
- All square-lattice archive parts must be downloaded before extracting `square_lattice.zip`.
- The triangular-lattice samples are stored directly in the repository.
- The complete simulation data sets are not included in this repository.

---

# Reference

This repository contains exact simulation code and representative data associated with studies of charge-density-wave dynamics in the spinless t–V model.

If you use the code or data in research, please cite the corresponding manuscript/publication.
