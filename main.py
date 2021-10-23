from argparse import ArgumentParser
from pathlib import Path
from benchmarks.moo_bb_func import get_bb_func

parser = ArgumentParser()
parser.add_argument("problem", type=str)
parser.add_argument("rnd", type=int)
parser.add_argument("n_calls", type=int)
parser.add_argument("-dim", default=3, type=int)
parser.add_argument("-kappa", default=1.96, type=float)

args = parser.parse_args()

# get initial points
import json

with open("moo_results/bb_init.json") as json_file:
    x_init = json.load(json_file)[args.problem][str(args.rnd)]

# for the windfarm case we have a constraint problem
if args.problem == "Windfarm":
    bb_func = get_bb_func(args.problem)
    opt_core = bb_func.get_opt_core()

    from entmoot_moo.optimizer import Optimizer
    opt = Optimizer(
        bb_func.bounds,
        model="ENTING",
        model_unc="BDD",
        random_state=args.rnd,
        kappa=args.kappa,
        opt_core=opt_core
    )

    import numpy as np


    for idx, x in enumerate(x_init):
        print(f"{x[32:]}")

        # tell solver initial points
        next_y = bb_func(x)
        opt.tell(x, next_y)

    for _ in range(args.n_calls):
        next_x = opt.ask()
        next_x = np.asarray(next_x)

        for idx, x in enumerate(next_x):
            print(f"{idx}: {x}")

        if isinstance(next_x[0], list):
            next_y = [bb_func(x) for x in next_x]

            for i, y in enumerate(next_y):
                opt.tell(next_x[i], y)
        else:
            next_y = bb_func(next_x)
            opt.tell(next_x, next_y)

    data_x = opt.X
    data_y = opt.y

# battery example requires certain constraints
elif args.problem == "Battery":
    bb_func = get_bb_func(args.problem)
    opt_core = bb_func.get_opt_core()

    from entmoot_moo.optimizer import Optimizer
    opt = Optimizer(
        bb_func.bounds,
        model="ENTING",
        model_unc="BDD",
        random_state=args.rnd,
        kappa=args.kappa,
        opt_core=opt_core
    )

    import numpy as np


    for idx, x in enumerate(x_init):
        print(f"{x}")

        # tell solver initial points
        next_y = bb_func(x)
        opt.tell(x, next_y)

    for _ in range(args.n_calls):
        next_x = opt.ask()
        next_x = np.asarray(next_x)

        for idx, x in enumerate(next_x):
            print(f"{idx}: {x}")

        if isinstance(next_x[0], list):
            next_y = [bb_func(x) for x in next_x]

            for i, y in enumerate(next_y):
                opt.tell(next_x[i], y)
        else:
            next_y = bb_func(next_x)
            opt.tell(next_x, next_y)

    data_x = opt.X
    data_y = opt.y

# save results
from pathlib import Path
import os

# collect run info and results
file_name = "_".join([str(args.rnd), args.problem, "entmoot"]) + ".json"
res_dict = {
    "algo": "entmoot",
    "problem": args.problem,
    "rnd": args.rnd,
    "n_calls": args.n_calls,
    "dim": args.dim,
    "kappa": args.kappa
}

res_dict['data_y'] = []
res_dict['data_x'] = []

# format ndarray to serializable python objects
for itr in range(len(data_y)):
    res_dict['data_y'].append([float(y) for y in data_y[itr]])
    res_dict['data_x'].append([float(x) for x in data_x[itr]])

# create parent dir if it doesn't exist
parent_path = Path("moo_results") / (args.problem + '_' + str(args.n_calls))
if not os.path.isdir(parent_path):
    os.mkdir(parent_path)

# save file
full_path = parent_path / file_name
with open(full_path, 'w') as json_file:
    json.dump(res_dict, json_file, indent=4)