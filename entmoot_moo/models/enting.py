from entmoot_moo.models.model_utils import TreeEnsemble
from timeit import default_timer
import numpy as np


class Enting:
    def __init__(self, space,
                 acq_func="LCB",
                 kappa=1.96,
                 model_unc="BDD",
                 random_state=None,
                 param=None):
        self._space = space
        self._random_state = random_state

        self._acq_func = acq_func
        self._kappa = kappa
        self._n_obj = 2

        if not param:
            self.param = {}

        # setup model uncertainty measure
        from entmoot_moo.models.model_unc import \
            DistanceBasedExploration

        allowed_model_unc = ['BDD', 'L1BDD']
        assert model_unc in allowed_model_unc, \
            f"'{model_unc}' not supported, please pick one of: '{allowed_model_unc}'"

        if model_unc == "BDD":
            self._model_unc = \
                DistanceBasedExploration(
                    space,
                    metric_cont="sq_euclidean",
                    metric_cat="goodall4",
                )

        elif model_unc == "L1BDD":
            self._model_unc = \
                DistanceBasedExploration(
                    space,
                    metric_cont="manhattan",
                    metric_cat="goodall4",
                )

    def reset(self):
        return self

    def fit(self, X, y):
        self._X = X
        self._y = y

        # define tree ensemble
        print(f"\n* train new tree model")
        self._tree = []

        for obj_idx in range(self._n_obj):

            # slice targets based on n_obj
            if self._n_obj > 1:
                obj_y = self._y[:, obj_idx]
            else:
                obj_y = self._y

            # train tree models
            tree_model = TreeEnsemble(
                self._X, obj_y,
                random_state=self._random_state,
                cat_idx=self._space.cat_idx,
                n_trees=400,
                dump_model=False)

            self._tree.append(tree_model)

        # update model uncertainty
        self._model_unc.update(self._X, self._y)

    def propose(self, n_propose=1, weights=None, opt_core=None):
        from entmoot_moo.models.opt_model_utils import \
            get_multi_obj_weight_samples

        np.random.seed(self._random_state)

        if self._n_obj > 1:
            # check if weights are provided
            if not weights:
                weights = get_multi_obj_weight_samples(
                    self._n_obj, n_propose)
        else:
            weights =[None]

        next_x_list = []

        for w in weights:
            # build gurobi model
            self._opt_model = self._build_opt_model(
                weights=w,
                add_points=next_x_list,
                opt_core=opt_core)

            # solve gurobi model
            start_time = default_timer()
            print(f"\n<> <> <> GUROBI SOLVE")
            self._opt_model.optimize()

            while self._opt_model.SolCount < 1:
                self._opt_model.Params.TimeLimit *= 2.0
                self._opt_model.optimize()

                # stop if no solution is found after 1hr
                if self._opt_model.Params.TimeLimit > 3600:
                    raise RuntimeError(
                        f"gurobi solver didn't find a solution after "
                        f"'{self._opt_model.Params.TimeLimit} s'"
                    )
            print(f"<> <> <> GUROBI SOLVE\n")
            runtime = default_timer() - start_time
            print(f"* * * time gurobi solve: {round(runtime)} s")

            # print mu and alpha
            from entmoot_moo.models.opt_model_utils import get_gbm_model_mu_val
            mu = get_gbm_model_mu_val(self._opt_model, 0)
            alpha = self._opt_model._alpha.x

            print(f"**** mu: {mu}")
            print(f"**** alpha: {alpha}")

            # extract solution
            next_x = np.empty(self._opt_model._n_feat)

            for cont_idx in self._opt_model._cont_var_dict.keys():
                next_x[cont_idx] = self._opt_model._cont_var_dict[cont_idx].x

            for cat_idx in self._opt_model._cat_var_dict.keys():
                active_cats = [cat
                               for cat in self._opt_model._cat_var_dict[cat_idx].keys()
                               if int(
                        round(self._opt_model._cat_var_dict[cat_idx][cat].x)
                    ) == 1]
                next_x[cat_idx] = active_cats[0]
            next_x_list.append(next_x)

        return next_x_list

    def _build_opt_model(self, weights=None, add_points=None, opt_core=None):
        # build opt_model core
        from entmoot_moo.models.opt_model_utils import \
            get_opt_core, add_gbm_to_opt_model, add_acq_to_opt_model, \
            get_gbm_model_mu, get_gbm_model_multi_obj_mu

        opt_model = get_opt_core(self._space, opt_core=opt_core)

        ## set log parameters
        opt_model.Params.LogToConsole = 1
        opt_model.Params.TimeLimit = 100

        # add tree logic to opt_model
        gbm_model_dict = {
            obj_idx: tree._gbm_model
            for obj_idx, tree in enumerate(self._tree)
        }

        add_gbm_to_opt_model(self._space,
                             gbm_model_dict,
                             opt_model,
                             z_as_bin=False)

        # add uncertainty to opt_model
        self._model_unc.add_to_opt_model(opt_model, add_points=add_points)

        # add acq function
        if self._n_obj > 1:
            model_mu = get_gbm_model_multi_obj_mu(opt_model, self._y)
            model_unc = self._model_unc.get_opt_model_obj(opt_model)
        else:
            model_mu = get_gbm_model_mu(opt_model, self._y, norm=True)
            model_unc = self._model_unc.get_opt_model_obj(opt_model)


        add_acq_to_opt_model(opt_model, model_mu, model_unc,
                             n_obj=self._n_obj,
                             weights=weights,
                             kappa=self._kappa,
                             acq_func=self._acq_func)
        return opt_model
