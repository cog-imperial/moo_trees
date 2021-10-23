from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

import numpy as np

class DistanceBasedUnc(ABC):

    def __init__(self,
                 space,
                 metric_cont='sq_euclidean',
                 metric_cat='goodall4'):

        self._space = space
        self.unc_type = 'distance'

        # set distance metric for cont variables
        if metric_cont == 'sq_euclidean':
            self.cont_dist_metric = \
                SquaredEuclidean(self._space)

        elif metric_cont == 'manhattan':
            self.cont_dist_metric = \
                Manhattan(self._space)

        # set similarity metric for cat variables
        if metric_cat == "overlap":
            self.cat_sim_metric = \
                Overlap(self._space)

        elif metric_cat == "goodall4":
            self.cat_sim_metric = \
                Goodall4(self._space)

    def update(self, Xi, yi):

        # update data set attributes
        self._Xi = Xi
        self._yi = yi

        # update cat_sim_metric with new data
        self.cat_sim_metric.update(self._Xi)

    @abstractmethod
    def add_to_opt_model(self, model, add_points=None):

        pass

    @abstractmethod
    def get_opt_model_obj(self, model):

        pass


class DistanceMetric(ABC):

    def __init__(self, space):
        self._space = space

    @abstractmethod
    def get_distance(self, X1, X2):
        pass

    def get_max_space_scaled_dist(self, ref_points, x_means, x_stddev, model):
        # computes maximum distance in search space
        lb = np.asarray(
            [model._cont_var_dict[i].lb for i in model._cont_var_dict.keys()]
        )
        ub = np.asarray(
            [model._cont_var_dict[i].ub for i in model._cont_var_dict.keys()]
        )

        lb_std = np.divide(lb - x_means, x_stddev)
        ub_std = np.divide(ub - x_means, x_stddev)

        max_dist = self.get_distance(
            lb_std,
            ub_std
        )
        return max_dist


class SquaredEuclidean(DistanceMetric):

    @staticmethod
    def get_distance(X1, X2):
        if X1.ndim == 1:
            dist = np.sum((X1 - X2) ** 2)
        else:
            dist = np.sum((X1 - X2) ** 2, axis=1)
        return dist

    def add_exploration_to_opt_model(self,
                                     ref_points, model,
                                     add_rhs=None):

        from gurobipy import quicksum

        # variable alpha captures distance measure
        model._alpha = \
            model.addVar(
                lb=0,
                ub=1,
                name="alpha",
                vtype='C'
            )

        def distance_ref_point_i(model, xi_ref, add_rhs=None):
            # function returns constraints capturing the standardized
            # exploration distance
            c_x = model._cont_var_dict
            alpha = model._alpha

            diff_to_ref_point_i = quicksum(
                ((xi_ref[idx] - c_x[cont_idx]) *
                 (xi_ref[idx] - c_x[cont_idx]))
                for idx, cont_idx in enumerate(model._cont_var_dict.keys())
            )

            if add_rhs is None:
                return alpha * float(model._n_feat) <= diff_to_ref_point_i
            else:
                return alpha * float(model._n_feat) <= diff_to_ref_point_i + add_rhs

        # add exploration distances as quadratic constraints to the model
        for i in range(len(ref_points)):
            if add_rhs:
                temp_rhs = add_rhs[i]
            else:
                temp_rhs = None

            model.addQConstr(
                distance_ref_point_i(
                    model, ref_points[i], temp_rhs
                ),
                name=f"unc_const_{i}"
            )
        model.update()

class Manhattan(DistanceMetric):

    @staticmethod
    def get_distance(X1, X2):
        if X1.ndim == 1:
            dist = np.sum(np.abs(X1 - X2))
        else:
            dist = np.sum(np.abs(X1 - X2), axis=1)
        return dist

    def add_exploration_to_opt_model(self,
                                     ref_points, model,
                                     add_rhs=None):

        from gurobipy import GRB, quicksum

        # two sets of variables are used to capture positive and negative
        # parts of manhattan distance
        model._c_x_aux_pos = \
            model.addVars(range(len(ref_points)),
                          model._cont_var_dict.keys(),
                          name="man_x_aux_pos", vtype='C')

        model._c_x_aux_neg = \
            model.addVars(range(len(ref_points)),
                          model._cont_var_dict.keys(),
                          name="man_x_aux_neg", vtype='C')

        # variable alpha captures distance measure
        model._alpha = \
            model.addVar(lb=0,
                         ub=1,
                         name="alpha", vtype='C')

        def distance_ref_point_i_for_feat_j(
                model, xi_ref, i_ref, feat_j, i_var):
            # function returns constraints capturing the standardized
            # exploration distance
            c_x = model._cont_var_dict

            diff_to_ref_point_i = \
                (xi_ref[feat_j] - c_x[i_var])
            return diff_to_ref_point_i == model._c_x_aux_pos[i_ref, i_var] - \
                   model._c_x_aux_neg[i_ref, i_var]

        for i_ref in range(len(ref_points)):
            for feat_j, cont_idx in enumerate(model._cont_var_dict.keys()):
                # add constraints to capture distances in variables
                # _c_x_aux_pos and _c_x_aux_neg
                model.addConstr(
                    distance_ref_point_i_for_feat_j(model,
                                                    ref_points[i_ref], i_ref, feat_j, cont_idx),
                    name=f"unc_const_feat_{cont_idx}_{i_ref}"
                )
                # add sos constraints that allow only one of the +/- vars,
                # i.e. _c_x_aux_pos / _c_x_aux_neg to be active
                model.addSOS(GRB.SOS_TYPE1,
                             [model._c_x_aux_pos[i_ref, cont_idx],
                              model._c_x_aux_neg[i_ref, cont_idx]])

            # add exploration distances as linear constraints to the model
            if add_rhs is None:
                model.addConstr(
                    model._alpha * float(model._n_feat) <= quicksum(
                        (model._c_x_aux_pos[i_ref, j] +
                         model._c_x_aux_neg[i_ref, j])
                        for j in model._cont_var_dict.keys()
                    ),
                    name=f"man_alpha_sum"
                )
            else:
                model.addConstr(
                    model._alpha * float(model._n_feat) <= quicksum(
                        (model._c_x_aux_pos[i_ref, j] +
                         model._c_x_aux_neg[i_ref, j])
                        for j in model._cont_var_dict.keys()
                    ) + add_rhs[i_ref],
                    name=f"alpha_sum"
                )
        model.update()


class NonSimilarityMetric:
    def __init__(self, space):
        self._space = space

    def update(self, Xi):
        self._Xi_cat = Xi[:, self._space.cat_idx].astype(int)

    def get_gurobi_model_rhs(self, model):
        if self._space.cat_idx:
            constr_list = []
            for X in self._Xi_cat:
                temp_constr = 0

                for cat_idx in range(len(X)):
                    for cat in range(len(self._cat_mat[cat_idx])):
                        temp_cat_mat = self._cat_mat[cat_idx][cat, X[cat_idx]]
                        temp_constr += \
                            (1 - temp_cat_mat) * \
                            model._cat_var_dict[self._space.cat_idx[cat_idx]][cat]

                constr_list.append(temp_constr)

            return constr_list
        else:
            return None

    def get_similarity(self, X1, X2):
        # create zero vector that contains similarities for all data points
        sim_vec = np.zeros(X1.shape[0])

        # iterate through cat features and populate sim_mat entries
        for cat_id in range(X1.shape[1]):
            # compute individual similarities
            sim_vec += \
                self._cat_mat[cat_id][int(X2[cat_id]), X1[:, cat_id]]
        return sim_vec


class Overlap(NonSimilarityMetric):

    def update(self, Xi):
        super().update(Xi)

        # generate matrix rules
        cat_mat_rule = \
            lambda i, j, cat_idx: 1 if i == j else 0

        self._cat_mat = []

        for idx, cat_idx in enumerate(self._space.cat_idx):
            d = self._space.dimensions[cat_idx]
            n_cat = len(d.categories)

            # populate matrix
            temp_mat = np.fromfunction(
                np.vectorize(cat_mat_rule, ), (n_cat, n_cat),
                dtype=int, cat_idx=idx
            )

            self._cat_mat.append(temp_mat)

    def get_similarity(self, X1, X2):
        dist_cat = np.sum(X1 == X2, axis=1)
        return dist_cat


class Goodall4(NonSimilarityMetric):

    def update(self, Xi):
        super().update(Xi)

        def get_pk2(cat_rows, cat):
            count_cat = np.sum(cat_rows == cat)
            n_rows = len(cat_rows)
            return (count_cat * (count_cat - 1)) / (n_rows * (n_rows - 1))

        # generate matrix rules
        cat_mat_rule = \
            lambda i, j, cat_idx: get_pk2(self._Xi_cat[:, cat_idx], i) \
                if i == j else 0

        self._cat_mat = []

        for idx, cat_idx in enumerate(self._space.cat_idx):
            d = self._space.dimensions[cat_idx]
            n_cat = len(d.categories)

            # populate matrix
            temp_mat = np.fromfunction(
                np.vectorize(cat_mat_rule, ), (n_cat, n_cat), dtype=int, cat_idx=idx
            )
            self._cat_mat.append(temp_mat)

class DistanceBasedExploration(DistanceBasedUnc):

    def __init__(self,
                 space,
                 metric_cont='sq_euclidean',
                 metric_cat='overlap'):
        super().__init__(space, metric_cont, metric_cat)

    def set_params(self, **kwargs):
        pass

    def update(self, Xi, yi):
        super().update(Xi, yi)

        self._ref_points_cont = Xi[:, self._space.cont_idx]
        self._ref_points_cat = Xi[:, self._space.cat_idx]

    def get_opt_model_obj(self, model):
        # negative contributation of alpha requires non-convex flag in gurobi.
        model.Params.NonConvex = 2
        return -model._alpha

    def add_to_opt_model(self, model, add_points=None):
        if add_points:
            ref_points = \
                np.vstack([self._ref_points_cont, add_points])
        else:
            ref_points = self._ref_points_cont

        # first get rhs of model constraints based on cat values
        cat_rhs = self.cat_sim_metric.get_gurobi_model_rhs(model)

        self.cont_dist_metric.add_exploration_to_opt_model(
            ref_points,
            model,
            add_rhs=cat_rhs
        )