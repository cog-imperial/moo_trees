import numpy as np
from benchmarks.moo_windfarm import windfarm

def get_bb_func(name):
    if name == 'Windfarm':
        return Wind()

    elif name == 'Battery':
        return Battery()

class MultiBB:

    def __call__(self, x):
        return [self.f1(x), self.f2(x)]

class Wind(MultiBB):
    def __init__ (self):
        self.name = 'Windfarm'

    @property
    def r(self):
        sigma = 2.
        return np.array([[-1.0 - (1e-5), 0.0 + (1e-5)],[-1.0 - (1e-5), 0.0 + (1e-5)]])

    @property
    def bounds(self):
        bnds = [[0., 3900.]] * 32 + [[0, 1]] * 16
        return bnds

    def __call__(self, x):
        x_vals = x[0:16]
        y_vals = x[16:32]
        bool_vals = np.asarray(np.rint(x[32:48]), dtype=bool)

        return windfarm(x=x_vals, y=y_vals, ispresent=bool_vals)

    def get_opt_core(self):
        from entmoot_moo.models.opt_model_utils import get_opt_core
        from entmoot_moo.space.space import Space

        # init space and opt_core
        bnds = self.bounds
        temp_space = Space(bnds)
        temp_core = get_opt_core(temp_space)

        ## add ordering
        for var in range(32, 47):
            temp_core.addConstr(
                temp_core._cont_var_dict[var] >= temp_core._cont_var_dict[var + 1]
            )

        ## only optimize turbines that are active
        for var in range(0, 16):
            temp_bin = temp_core._cont_var_dict[var + 32]
            temp_x = temp_core._cont_var_dict[var]
            temp_y = temp_core._cont_var_dict[var + 16]

            temp_core.addConstr((temp_bin == 1) >> (temp_x >= 0))
            temp_core.addConstr((temp_bin == 0) >> (temp_x <= 0))

            temp_core.addConstr((temp_bin == 1) >> (temp_y >= 0))
            temp_core.addConstr((temp_bin == 0) >> (temp_y <= 0))

        ## minimum one turbine
        from gurobipy import quicksum
        temp_core.addConstr(
            quicksum(temp_core._cont_var_dict[var] for var in range(32, 48)) >= 1
        )

        ## minimum distance
        for i in range(0, 15):
            for j in range(i + 1, 16):
                temp_x1 = temp_core._cont_var_dict[i]
                temp_x2 = temp_core._cont_var_dict[j]

                temp_y1 = temp_core._cont_var_dict[i + 16]
                temp_y2 = temp_core._cont_var_dict[j + 16]

                temp_dist = temp_core.addVar(name=f"min_dist: {i}_{j}",
                                             vtype='C')

                temp_core.addConstr(
                    (temp_x1 - temp_x2) * (temp_x1 - temp_x2) + \
                    (temp_y1 - temp_y2) * (temp_y1 - temp_y2) >= \
                    temp_dist * temp_dist
                )

                temp_bin1 = temp_core._cont_var_dict[i + 32]
                temp_bin2 = temp_core._cont_var_dict[j + 32]

                extra_bin = temp_core.addVar(name=f"extr_bin {j}_{j}",
                                             vtype='B')

                temp_core.addConstr(temp_bin1 * temp_bin2 == extra_bin)

                temp_core.addConstr((extra_bin == 1) >> (temp_dist >= 0.25 + 0.0001))

        # opt_core parameters
        temp_core.Params.NonConvex = 2
        temp_core.Params.TimeLimit = 100

        return temp_core


class Battery:
    def __init__ (self):
        self.name = 'Battery'

    @property
    def r(self):
        sigma = 2.
        return np.array([[-1.0 - (1e-5), 0.0 + (1e-5)],[-1.0 - (1e-5), 0.0 + (1e-5)]])

    @property
    def bounds(self):
        bnds = [
            (0,1,2,3,4),  # name of parameter set
            (0.5, 8.2),   # C rate
            (0.2,0.7),    # Negative electrode porosity
            (0.2,0.7),    # Negative electrode active material volume fraction
            (1e-6, 20e-6),# Negative particle radius [m]
            (0.2, 0.7),   # Positive electrode porosity
            (0.2, 0.7),   # Positive electrode active material volume fraction
            (1e-6, 20e-6),# Positive particle radius [m]
            (0.5, 2.0),   # negative size scaling
            (0.5, 2.0)    # positive size scaling
        ]
        return bnds

    def __call__(self, x):
        from benchmarks.moo_battery import run_battery_simulation
        y = run_battery_simulation(x)
        return y

    def get_opt_core(self):
        from entmoot_moo.models.opt_model_utils import get_opt_core
        from entmoot_moo.space.space import Space

        # init space and opt_core
        bnds = self.bounds
        temp_space = Space(bnds)
        temp_core = get_opt_core(temp_space)

        # only one category is active
        temp_core.addConstr(
            sum([temp_core._cat_var_dict[0][cat]
                 for cat in temp_core._cat_var_dict[0].keys()]) == 1)

        ## add constraints for porosity - active material fractions
        ## sum is always <= 0.95 to ensure theres at least 5 % left for binder
        # negative electrode
        min_poro = self.bounds[2][0]
        max_poro = self.bounds[2][1]
        range_poro = max_poro - min_poro

        min_act = self.bounds[3][0]
        max_act = self.bounds[3][1]
        range_act = max_act - min_act

        temp_core.addConstr(
            range_poro*temp_core._cont_var_dict[2] + min_poro + \
            range_act*temp_core._cont_var_dict[3] + min_act \
            <= 0.95
        )

        # positive electrode
        min_poro = self.bounds[5][0]
        max_poro = self.bounds[6][1]
        range_poro = max_poro - min_poro

        min_act = self.bounds[5][0]
        max_act = self.bounds[6][1]
        range_act = max_act - min_act

        temp_core.addConstr(
            range_poro*temp_core._cont_var_dict[5] + min_poro + \
            range_act*temp_core._cont_var_dict[6] + min_act \
            <= 0.95
        )

        ## add constraints for variable C-rate ranges,
        ## i.e. different parameter sets are considered for different C-rate
        ## ranges

        min_c = self.bounds[1][0]
        max_c = self.bounds[1][1]
        range_c = max_c - min_c

        # 0: 'Ai2020' ranges -> (0.5, 3.2)
        temp_core.addConstr(
            (temp_core._cat_var_dict[0][0] == 1) >> \
            (temp_core._cont_var_dict[1] <= (3.2 - min_c) / range_c ))

        # 1: 'Chen2020' ranges -> (0.5, 2.2)
        temp_core.addConstr(
            (temp_core._cat_var_dict[0][1] == 1) >> \
            (temp_core._cont_var_dict[1] <= (2.2 - min_c) / range_c ))

        # 0: 'Ecker2015' ranges -> (0.5, 8.2)
        temp_core.addConstr(
            (temp_core._cat_var_dict[0][2] == 1) >> \
            (temp_core._cont_var_dict[1] <= (8.2 - min_c) / range_c ))

        # 0: 'Marquis2019' ranges -> (0.5, 5.2)
        temp_core.addConstr(
            (temp_core._cat_var_dict[0][3] == 1) >> \
            (temp_core._cont_var_dict[1] <= (5.2 - min_c) / range_c ))

        # 0: 'Yang2017' ranges -> (0.5, 8.2)
        temp_core.addConstr(
            (temp_core._cat_var_dict[0][4] == 1) >> \
            (temp_core._cont_var_dict[1] <= (8.2 - min_c) / range_c ))

        # opt_core parameters
        temp_core.Params.NonConvex = 2
        temp_core.Params.TimeLimit = 100

        return temp_core