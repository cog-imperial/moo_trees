from benchmarks.moo_bb_func import get_bb_func
from entmoot_moo.optimizer import Optimizer

# iterate through different problem names to
for bb_name in ['Battery','Windfarm']:
    bb_func = get_bb_func(bb_name)

    # we consider 25 different rnd states in this example
    for itr in range(25):
        seed = 101 + itr

        # consider the windfarm example
        if bb_func.name == 'Windfarm':
            opt_core = bb_func.get_opt_core()
            opt = Optimizer(
                bb_func.bounds,
                model="ENTING",
                model_unc="BDD",
                random_state=seed,
                opt_core=opt_core
            )

            # generate initial points
            x_init = []
            for i_init in range(16):
                # gets a model core with all problem constraints
                temp_opt_core = bb_func.get_opt_core()

                # pertubation in placement of first turbine to generate varying solutions
                if i_init == 0:
                    import random
                    random.seed(seed)
                    pert = random.uniform(0, 0.5)

                    temp_opt_core.addConstr(
                        temp_opt_core._cont_var_dict[0] == 0.5 + pert
                    )
                    temp_opt_core.addConstr(
                        temp_opt_core._cont_var_dict[16] == 0.5 + pert
                    )

                # activate different turbines at every iteration
                for var in range(0, i_init + 1):
                    temp_opt_core.addConstr(
                        temp_opt_core._cont_var_dict[var + 32] == 1
                    )

                for var in range(i_init + 1, 16):
                    temp_opt_core.addConstr(
                        temp_opt_core._cont_var_dict[var + 32] == 0
                    )

                # solve the problem with maximized distance between points
                x_init.append(
                    opt.ask_feas_samples(
                        1,
                        opt_core=temp_opt_core,
                        init_samples=x_init
                    )
                )
        elif bb_func.name == 'Battery':
            opt_core = bb_func.get_opt_core()
            opt = Optimizer(
                bb_func.bounds,
                model="ENTING",
                model_unc="BDD",
                random_state=seed,
                opt_core=opt_core
            )

            # generate initial points
            x_init = []
            import random
            random.seed(seed)

            for i_init in range(10):
                temp_opt_core = bb_func.get_opt_core()

                # pick param set and randomized c-rate
                if i_init == 0 or i_init == 5:
                    param_set = 0
                    set_ub_c_rate = 3.2
                elif i_init == 1 or i_init == 6:
                    param_set = 1
                    set_ub_c_rate = 2.2
                elif i_init == 2 or i_init == 7:
                    param_set = 2
                    set_ub_c_rate = 8.2
                elif i_init == 3 or i_init == 8:
                    param_set = 3
                    set_ub_c_rate = 5.2
                elif i_init == 4 or i_init == 9:
                    param_set = 4
                    set_ub_c_rate = 8.2

                lb_c_rate = bb_func.bounds[1][0]
                ub_c_rate = bb_func.bounds[1][1]

                c_rate = random.uniform(
                    lb_c_rate, int(set_ub_c_rate))
                range_c_rate = ub_c_rate - lb_c_rate

                temp_opt_core.addConstr(
                    temp_opt_core._cat_var_dict[0][param_set] == 1
                )

                temp_opt_core.addConstr(
                    temp_opt_core._cont_var_dict[1] == \
                    (c_rate - lb_c_rate) / range_c_rate
                )

                print(f"* * * param_set: {param_set}")
                print(f"* * * c_rate: {c_rate}")

                x_init.append(
                    opt.ask_feas_samples(
                        1,
                        opt_core=temp_opt_core,
                        init_samples=x_init
                    )
                )


        trafo_x_init = []
        for x in x_init:
            x_temp = [float(xi) for xi in x]
            trafo_x_init.append(x_temp)

        import os
        if not os.path.exists('moo_results'):
            os.makedirs('moo_results')
            f = open("moo_results/bb_init.json", "x")
            f.write("{}")
            f.close()

        import json
        with open("moo_results/bb_init.json") as json_file:
            init_dict = json.load(json_file)

        if bb_func.name not in init_dict.keys():
            init_dict[bb_func.name] = {}

        init_dict[bb_func.name][seed] = trafo_x_init

        with open("moo_results/bb_init.json", 'w') as json_file:
            json.dump(init_dict, json_file, indent=4)

        print(f"... save rnd seed: {seed}")

