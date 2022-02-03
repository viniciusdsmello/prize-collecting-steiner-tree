# Prize Collecting Steiner Tree based on Ant Colony Optimization

Prize-Collecting Steiner Tree (PCSTP) [1] problem solved with Ant-Colony Optimization for the discipline of Metaheuristics in Combinatorial Optimization of master degree at PESC - UFRJ.

In this Project an ACO algorithm was developed and compared with NetworkX Implementation [2] and also a Greedy Heuristic based on shortest path between terminals pairs.

# Repository Structure
```
.
├── notebooks      # Directory with Jupyter Notebooks files
├── data           # Directory with PCSTP instances
├── experiments    # Directory with scripts to run experiments in parallel
├── pcstp          # Base directory for all implementation
│   ├── instances  # Module with scripts to parse instance files, and also generate instances.
│   ├── solver     # Module with ACO and Greedy Heuristic implementation
│   ├── utils      # Module with auxiliary files such as scripts to draw graphs and also to preprocess graphs before calling the solvers.
└── README.md
```

# Refereces
1. The Prize-Collecting Steiner Tree Problem - https://homepage.univie.ac.at/ivana.ljubic/research/pcstp/#:~:text=Definition%3A,needed%20to%20establish%20the%20network.
2. NetworkX Approximation to Steiner Tree - https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.steinertree.steiner_tree.html
