import numpy as np


# TODO: add arg types to all function inputs


class Optimizer:

    def __init__(
            self,
            dims,
            model="ENTING",
            model_unc="BDD",
            acq_func="LCB",
            kappa=1.96,
            random_state=None,
            opt_core=None,
    ):

        # save problem information
        from entmoot_moo.space.space import Space
        self.space = Space(dims)
        self.random_state = random_state
        self._opt_core = opt_core

        # store dataset, i.e. we assume 2 objectives
        self.X = []
        self.y = np.zeros((0, 2))

        # init model and model_cache
        from entmoot_moo.utils import check_model
        self.base_model = \
            check_model(model,
                        space=self.space,
                        random_state=self.random_state,
                        model_unc=model_unc,
                        kappa=kappa,
                        acq_func=acq_func)

    def ask_feas_samples(self, n_points=1, opt_core=None, init_samples=None):
        from entmoot_moo.models.opt_model_utils import get_opt_core_copy
        from entmoot_moo.models.model_unc import \
            DistanceBasedExploration

        if not opt_core:
            opt_core = self._opt_core
        opt_core.update()

        model_unc = DistanceBasedExploration(
            self.space,
            metric_cont="manhattan",
            metric_cat="goodall4",
        )

        if init_samples:
            x_samples = init_samples
        else:
            x_samples = []

        new_x_samples = []
        for _ in range(n_points):
            opt_core = get_opt_core_copy(opt_core)

            if x_samples:
                # transform data

                x_trafo = np.asarray(
                    self.space.transform(x_samples)
                )

                model_unc.update(x_trafo, None)
                model_unc.add_to_opt_model(opt_core)
                ob_expr = model_unc.get_opt_model_obj(opt_core)
                opt_core.setObjective(ob_expr)

            else:
                x_trafo = []

            # optimize core
            opt_core.optimize()
            while opt_core.SolCount < 1:
                opt_core.Params.TimeLimit *= 2.0
                opt_core.optimize()

                # stop if no solution is found after 1hr
                if opt_core.Params.TimeLimit > 3600:
                    raise RuntimeError(
                        f"gurobi solver didn't find a solution after "
                        f"'{opt_core.Params.TimeLimit} s'"
                    )

            # extract solution
            next_x = np.empty(opt_core._n_feat)

            for cont_idx in opt_core._cont_var_dict.keys():
                next_x[cont_idx] = opt_core._cont_var_dict[cont_idx].x

            for cat_idx in opt_core._cat_var_dict.keys():
                active_cats = [cat
                               for cat in opt_core._cat_var_dict[cat_idx].keys()
                               if int(
                        round(opt_core._cat_var_dict[cat_idx][cat].x)
                    ) == 1]
                next_x[cat_idx] = active_cats[0]


            # inverse_transform next_x
            for next_tx in [next_x]:
                trafo_lb = [d.transformed_bounds[0]
                            for d in self.space.dimensions]
                trafo_ub = [d.transformed_bounds[1]
                            for d in self.space.dimensions]

                for idx in self.space.cat_idx:
                    trafo_lb[idx] = 0
                    trafo_ub[idx] = len(self.space.dimensions[idx].categories) - 1

                clipped_next = np.clip(next_tx, trafo_lb, trafo_ub)
                next_x = self.space.inverse_transform(
                    clipped_next.reshape((1, -1))
                )[0]
                new_x_samples.append(next_x)

        if len(new_x_samples) == 1:
            return new_x_samples[0]
        else:
            return new_x_samples

    def ask(self):
        model = self.base_model.reset()

        # transform data
        x_trafo = np.asarray(
            self.space.transform(self.X)
        )

        # standardize targets
        mu = np.median(self.y, axis=0)
        sigma = self.y.std(axis=0)
        sigma[sigma < 1e-6] = 1.0
        y_trafo = (self.y - mu) / sigma


        # fit model and propose next points
        model.fit(x_trafo, y_trafo)

        if self._opt_core:
            from entmoot_moo.models.opt_model_utils import get_opt_core_copy
            self._opt_core.update()
            opt_core = get_opt_core_copy(self._opt_core)
        else:
            opt_core = None

        next_tx_list = model.propose(opt_core=opt_core)

        # inverse_transform next_x
        next_x_list = []
        for next_tx in next_tx_list:
            trafo_lb = [d.transformed_bounds[0]
                        for d in self.space.dimensions]
            trafo_ub = [d.transformed_bounds[1]
                        for d in self.space.dimensions]

            for idx in self.space.cat_idx:
                trafo_lb[idx] = 0
                trafo_ub[idx] = len(self.space.dimensions[idx].categories) - 1

            clipped_next = np.clip(next_tx, trafo_lb, trafo_ub)
            next_x = self.space.inverse_transform(
                clipped_next.reshape((1, -1))
            )[0]
            next_x_list.append(next_x)

        if len(next_x_list) == 1:
            return next_x_list[0]
        else:
            return next_x_list

    def tell(self, X, y):
        # check inputs
        # TODO: check inputs

        # add new points to data
        self.X.append(X)

        # check dim for multi-obj
        assert len(y) == 2, \
            f" provide 'y' of dimension '2'"

        self.y = np.vstack([self.y, y])
