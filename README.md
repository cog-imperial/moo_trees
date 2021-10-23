# MOO_TREES
This repository contains scripts for the multi-objective extension of ENTMOOT featured in: <cite-article>.

Please cite this work as:
```
@article{thebelt2021mootrees,
  title   =   {{Multi-objective Constrained Optimization for Energy Applications via Tree Ensembles}},
  author  =   {Thebelt, Alexander and Tsay, Calvin and Lee, Robert M. and Walz, David and Tranter, Tom and Misener, Ruth},
  journal =   {Applied Energy},
  volume  =   {},
  year    =   {2021}
}
```

## Dependencies
* python >= 3.7.4
* numpy >= 1.20.3
* scipy >= 1.6.3
* gurobipy >= 9.1.2
* pyaml >= 20.4.0
* scikit-learn >= 0.24.2
* lightgbm >= 3.2.1
* pybamm >= 0.4.0

For PyBaMM please install this branch `https://github.com/pybamm-team/PyBaMM/tree/issue-1575-discharged_energy`, which
allows direct access to the `discarged_energy` variable. The following command will install the right branch:

`pip install git+https://github.com/pybamm-team/PyBaMM.git@issue-1575-discharged_energy`

## Installing Gurobi
The solver software [Gurobi](https://www.gurobi.com) is required to run the examples. Gurobi is a commercial mathematical optimization solver and free of charge for academic research. It is available on Linux, Windows and Mac OS. 

Please follow the instructions to obtain a [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/). Once Gurobi is installed on your system, follow the steps to setup the Python interface [gurobipy](https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html).

## Running Experiments
This repo includes the two benchmark problems: (i) windfarm layout optimization which was adapted from 
[here](https://www.sciencedirect.com/science/article/pii/S1364032116303458), and (ii) battery optimization 
which uses [PyBaMM](https://github.com/pybamm-team/PyBaMM) to simulate different configurations.

To run experiments please first execute `create_init` to generate all initial points for 25
different random seeds for both benchmarks which will be stored in `moo_results/bb_init.json`. A directory `moo_results` will be created if it doesn't exist already.

Afterwards, you can call `main.py` to run experiments:

e.g. `python main.py Windfarm 101 10` runs the windfarm benchmark for random seed 101 and evaluation budget 10.

## Authors
* **[Alexander Thebelt](https://optimisation.doc.ic.ac.uk/person/alexander-thebelt/)** ([ThebTron](https://github.com/ThebTron)) - Imperial College London
* **[Calvin Tsay](https://www.imperial.ac.uk/people/c.tsay)** ([tsaycal](https://github.com/tsaycal)) - Imperial College London
* Robert M. Lee - BASF SE
* **[Nathan Sudermann-Merx](https://www.mannheim.dhbw.de/profile/sudermann-merx)** ([spiralulam](https://github.com/spiralulam)) - Cooperative State University Mannheim
* **[David Walz](https://www.linkedin.com/in/walzds/?originalSubdomain=de)** ([DavidWalz](https://github.com/DavidWalz)) - BASF SE
* **[Tom Tranter](https://www.mannheim.dhbw.de/profile/sudermann-merx)** ([TomTranter](https://github.com/TomTranter)) - Electrochemical Innovation Lab UCL
* **[Ruth Misener](http://wp.doc.ic.ac.uk/rmisener/)** ([rmisener](https://github.com/rmisener)) - Imperial College London

## License
This repository is released under the BSD 3-Clause License. Please refer to the [LICENSE](https://github.com/cog-imperial/moo_trees/blob/main/LICENSE) file for details.

## Acknowledgements
This work was supported by BASF SE, Ludwigshafen am Rhein, EPSRC Research Fellowships to RM (EP/P016871/1) and CT (EP/T001577/1), and an Imperial College Research Fellowship to CT. TT acknowledges funding from the EPSRC Faraday Institution Multiscale Modelling Project (EP/S003053/1, FIRG003).

