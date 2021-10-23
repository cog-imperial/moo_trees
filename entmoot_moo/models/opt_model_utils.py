from gurobipy import GRB, quicksum
import gurobipy as gp
import numpy as np
from entmoot_moo.space.space import Integer

def get_opt_core_copy(opt_core):
    new_opt_core = opt_core.copy()
    new_opt_core._n_feat = opt_core._n_feat

    # transfer var dicts
    new_opt_core._cont_var_dict = {}
    new_opt_core._cat_var_dict = {}

    ## transfer cont_var_dict
    for var in opt_core._cont_var_dict.keys():
        var_name = opt_core._cont_var_dict[var].VarName

        new_opt_core._cont_var_dict[var] = \
            new_opt_core.getVarByName(var_name)

    ## transfer cat_var_dict
    for var in opt_core._cat_var_dict.keys():
        for cat in opt_core._cat_var_dict[var].keys():
            var_name = opt_core._cat_var_dict[var][cat].VarName

            if var not in new_opt_core._cat_var_dict.keys():
                new_opt_core._cat_var_dict[var] = {}

            new_opt_core._cat_var_dict[var][cat] = \
                new_opt_core.getVarByName(var_name)

    return new_opt_core


def get_gbm_model_multi_obj_mu(opt_model, yi):
    # normalize mu based on targets
    y_min = np.min(yi, axis=0)
    y_max = np.max(yi, axis=0)
    y_range = y_max - y_min

    # collect weighted sum for every label
    ob_expr = {}
    for label in opt_model._gbm_set:
        weighted_sum = quicksum(
            opt_model._leaf_weight(label, tree, leaf) *
            opt_model._z_l[label, tree, leaf]
            for tree, leaf in label_leaf_index(opt_model, label)
        )
        # normalized mu contribution
        ob_expr[label] = (weighted_sum - y_min[label]) / y_range[label]
    return ob_expr


def get_gbm_model_multi_obj_mu_val(opt_model):
    return opt_model._mu.x


def get_gbm_model_mu(opt_model, yi, norm=False):
    # normalize mu based on targets
    y_min = np.min(yi)
    y_max = np.max(yi)
    y_range = y_max - y_min

    weighted_sum = quicksum(
        opt_model._leaf_weight(label, tree, leaf) * \
        opt_model._z_l[label, tree, leaf]
        for label, tree, leaf in leaf_index(opt_model)
    )
    if norm:
        return (weighted_sum - y_min) / y_range
    else:
        return weighted_sum


def get_gbm_model_mu_val(opt_model, label):
    temp_sum = sum(opt_model._leaf_weight(label, tree, leaf) * \
                   round(opt_model._z_l[label, tree, leaf].x, 1)
                   for temp_label, tree, leaf in leaf_index(opt_model)
                   if temp_label == label)
    return temp_sum


def get_multi_obj_weight_grid(n_obj, n_weights):
    # adapted from https://github.com/basf/mopti/blob/main/opti/sampling/simplex.py
    # Nijenhuis and Wilf, Combinatorial Algorithms, Chapter 5, Academic Press, 1978.
    from scipy.special import comb
    m = n_obj
    n = n_weights - 1
    L = comb(n_obj - 1 + n_weights - 1, n_obj - 1, exact=True)

    x = np.zeros(m, dtype=int)
    x[-1] = n

    out = np.empty((L, m), dtype=int)
    out[0] = x

    h = m
    for i in range(1, L):
        h -= 1

        val = x[h]
        x[h] = 0
        x[h - 1] += 1
        x[-1] = val - 1

        if val != 1:
            h = m

        out[i] = x

    return out / n

def get_multi_obj_weight_samples(n_obj, n_weights):
    s = np.random.standard_exponential((n_weights, n_obj))
    return (s.T / s.sum(axis=1)).T


def add_acq_to_opt_model(opt_model, model_mu, model_unc,
                         acq_func='LCB',
                         kappa=1.96,
                         n_obj=1,
                         weights=None):

    if n_obj > 1:
        opt_model._mu = opt_model.addVar(
            name='mu', vtype='C', lb=-GRB.INFINITY, ub=GRB.INFINITY)

        for obj_idx in model_mu.keys():
            temp_constr = \
                opt_model.addConstr(
                    opt_model._mu >= weights[obj_idx] * model_mu[obj_idx],
                    name=f"multi-obj_{obj_idx}"
                )
            opt_model.update()
        proc_mu = opt_model._mu
    else:
        proc_mu = model_mu

    if acq_func == "LCB":
        ob_expr = quicksum((proc_mu, kappa * model_unc))
        opt_model.setObjective(ob_expr, GRB.MINIMIZE)

    opt_model.update()


def get_opt_core(space, opt_core=None):
    if opt_core is None:
        model = gp.Model()
        model._cont_var_dict = {}
        model._cat_var_dict = {}
    else:
        return opt_core

    for idx, d in enumerate(space.dimensions):
        var_name = '_'.join(['x', str(idx)])

        if idx in space.cont_idx:

            lb = d.transformed_bounds[0]
            ub = d.transformed_bounds[1]

            if isinstance(d, Integer) and \
                d.prior in ["uniform"] and \
                (lb == 0 and ub == 1):
                # define binary vars
                model._cont_var_dict[idx] = \
                    model.addVar(name=var_name,
                                 vtype='B')
            else:
                # define continuous vars
                model._cont_var_dict[idx] = \
                    model.addVar(lb=lb,
                                 ub=ub,
                                 name=var_name,
                                 vtype='C')

        elif idx in space.cat_idx:
            # define categorical vars
            model._cat_var_dict[idx] = {}
            for cat in d.transform(d.bounds):
                model._cat_var_dict[idx][cat] = \
                    model.addVar(name=f"{var_name}_{cat}",
                                 vtype=GRB.BINARY)
    model._n_feat = \
        len(model._cont_var_dict) + len(model._cat_var_dict)

    model.update()
    return model


def get_max_var_range(bnd_dict):
    var_range = \
        [bnd_dict[var][1] - bnd_dict[var][0]
         for var in bnd_dict.keys()]

    if var_range:
        return max(var_range)
    else:
        return 1.0


def get_all_var_range(bnd_dict):
    return [bnd_dict[var][1] - bnd_dict[var][0]
            for var in sorted(bnd_dict.keys())]


def get_opt_cat_vals(model):
    cat_dict = {}
    # cat features
    for i in model._cat_var_dict.keys():
        cat = \
            [key
             for key in model._cat_var_dict[i].keys()
             if int(round(model._cat_var_dict[i][key].x, 1)) == 1]
        cat_dict[i] = cat[0]

    return cat_dict


### GBT HANDLER
## gbt model helper functions

def label_leaf_index(model, label):
    for tree in range(model._num_trees(label)):
        for leaf in model._leaves(label, tree):
            yield (tree, leaf)


def tree_index(model):
    for label in model._gbm_set:
        for tree in range(model._num_trees(label)):
            yield (label, tree)


tree_index.dimen = 2


def leaf_index(model):
    for label, tree in tree_index(model):
        for leaf in model._leaves(label, tree):
            yield (label, tree, leaf)


leaf_index.dimen = 3


def misic_interval_index(model):
    for var in model._breakpoint_index:
        for j in range(len(model._breakpoints(var))):
            yield (var, j)


misic_interval_index.dimen = 2


def misic_split_index(model):
    gbm_models = model._gbm_models
    for label, tree in tree_index(model):
        for encoding in gbm_models[label].get_branch_encodings(tree):
            yield (label, tree, encoding)


misic_split_index.dimen = 3


def alt_interval_index(model):
    for var in model.breakpoint_index:
        for j in range(1, len(model.breakpoints[var]) + 1):
            yield (var, j)


alt_interval_index.dimen = 2


def add_gbm_to_opt_model(space, gbm_model_dict, model, z_as_bin=False):
    add_gbm_parameters(space.cat_idx, gbm_model_dict, model)
    add_gbm_variables(model, z_as_bin)
    add_gbm_constraints(space.cat_idx, model)


def add_gbm_parameters(cat_idx, gbm_model_dict, model):
    model._gbm_models = gbm_model_dict

    model._gbm_set = set(gbm_model_dict.keys())
    model._num_trees = lambda label: \
        gbm_model_dict[label].n_trees

    model._leaves = lambda label, tree: \
        tuple(gbm_model_dict[label].get_leaf_encodings(tree))

    model._leaf_weight = lambda label, tree, leaf: \
        gbm_model_dict[label].get_leaf_weight(tree, leaf)

    vbs = [v.get_var_break_points() for v in gbm_model_dict.values()]

    all_breakpoints = {}
    for i in range(model._n_feat):
        if i in cat_idx:
            continue
        else:
            s = set()
            for vb in vbs:
                try:
                    s = s.union(set(vb[i]))
                except KeyError:
                    pass
            if s:
                all_breakpoints[i] = sorted(s)

    model._breakpoint_index = list(all_breakpoints.keys())

    model._breakpoints = lambda i: all_breakpoints[i]

    model._leaf_vars = lambda label, tree, leaf: \
        tuple(i
              for i in gbm_model_dict[label].get_participating_variables(
            tree, leaf))


def add_gbm_variables(model, z_as_bin=False):
    if not z_as_bin:
        model._z_l = model.addVars(
            leaf_index(model),
            lb=0,
            ub=GRB.INFINITY,
            name="z_l", vtype='C'
        )
    else:
        model._z_l = model.addVars(
            leaf_index(model),
            lb=0,
            ub=1,
            name="z_l", vtype=GRB.BINARY
        )

    model._y = model.addVars(
        misic_interval_index(model),
        name="y",
        vtype=GRB.BINARY
    )
    model.update()


def add_gbm_constraints(cat_idx, model):
    def single_leaf_rule(model_, label, tree):
        z_l, leaves = model_._z_l, model_._leaves
        return (quicksum(z_l[label, tree, leaf]
                         for leaf in leaves(label, tree))
                == 1)

    model.addConstrs(
        (single_leaf_rule(model, label, tree)
         for (label, tree) in tree_index(model)),
        name="single_leaf"
    )

    def left_split_r(model_, label, tree, split_enc):
        gbt = model_._gbm_models[label]
        split_var, split_val = gbt.get_branch_partition_pair(
            tree,
            split_enc
        )
        y_var = split_var

        if not isinstance(split_val, list):
            # for conti vars
            y_val = model_._breakpoints(y_var).index(split_val)
            return \
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_left_leaves(tree, split_enc)
                ) <= \
                model_._y[y_var, y_val]
        else:
            # for cat vars
            return \
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_left_leaves(tree, split_enc)
                ) <= \
                quicksum(
                    model_._cat_var_dict[split_var][cat]
                    for cat in split_val
                )

    def right_split_r(model_, label, tree, split_enc):
        gbt = model_._gbm_models[label]
        split_var, split_val = gbt.get_branch_partition_pair(
            tree,
            split_enc
        )
        y_var = split_var
        if not isinstance(split_val, list):
            # for conti vars
            y_val = model_._breakpoints(y_var).index(split_val)
            return \
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_right_leaves(tree, split_enc)
                ) <= \
                1 - model_._y[y_var, y_val]
        else:
            # for cat vars
            return \
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_right_leaves(tree, split_enc)
                ) <= 1 - \
                quicksum(
                    model_._cat_var_dict[split_var][cat]
                    for cat in split_val
                )

    def y_order_r(model_, i, j):
        if j == len(model_._breakpoints(i)):
            return Constraint.Skip
        return model_._y[i, j] <= model_._y[i, j + 1]

    def cat_sums(model_, i):
        return quicksum(
            model_._cat_var_dict[i][cat]
            for cat in model_._cat_var_dict[i].keys()
        ) == 1

    def var_lower_r(model_, i, j):
        lb = model_._cont_var_dict[i].lb
        j_bound = model_._breakpoints(i)[j]
        return model_._cont_var_dict[i] >= lb + (j_bound - lb) * (1 - model_._y[i, j])

    def var_upper_r(model_, i, j):
        ub = model_._cont_var_dict[i].ub
        j_bound = model_._breakpoints(i)[j]
        return model_._cont_var_dict[i] <= ub + (j_bound - ub) * (model_._y[i, j])

    model.addConstrs(
        (left_split_r(model, label, tree, encoding)
         for (label, tree, encoding) in misic_split_index(model)),
        name="left_split"
    )

    model.addConstrs(
        (right_split_r(model, label, tree, encoding)
         for (label, tree, encoding) in misic_split_index(model)),
        name="right_split"
    )

    # for conti vars
    model.addConstrs(
        (y_order_r(model, var, j)
         for (var, j) in misic_interval_index(model)
         if j != len(model._breakpoints(var)) - 1),
        name="y_order"
    )

    # for cat vars
    model.addConstrs(
        (cat_sums(model, var)
         for var in cat_idx),
        name="cat_sums"
    )

    model.addConstrs(
        (var_lower_r(model, var, j)
         for (var, j) in misic_interval_index(model)),
        name="var_lower"
    )

    model.addConstrs(
        (var_upper_r(model, var, j)
         for (var, j) in misic_interval_index(model)),
        name="var_upper"
    )
