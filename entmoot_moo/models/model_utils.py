from timeit import default_timer

def train_gbrt(X, y,
               cat_idx=None,
               random_state=101,
               n_trees=100):

    import lightgbm as lgb

    FIXED_PARAMS = {'objective': 'regression',
                    'metric': 'rmse',
                    'boosting': 'gbdt',
                    'num_boost_round': n_trees,
                    'max_depth': 3,
                    'min_data_in_leaf': 2,
                    'min_data_per_group': 2,
                    'random_state': random_state,
                    'verbose': -1}

    if cat_idx:
        train_data = lgb.Dataset(X, label=y,
                                 categorical_feature=cat_idx,
                                 free_raw_data=False,
                                 params={'verbose': -1})

        model = lgb.train(FIXED_PARAMS, train_data,
                          categorical_feature=cat_idx,
                          verbose_eval=False)
    else:
        train_data = lgb.Dataset(X, label=y,
                                 params={'verbose': -1})

        model = lgb.train(FIXED_PARAMS, train_data,
                          verbose_eval=False)

    return model, FIXED_PARAMS

class TreeEnsemble:
    def __init__(self,
                 X, y,
                 random_state=None,
                 cat_idx=None,
                 dump_model=False,
                 n_trees=200):

        self._X = X
        self._y = y

        # train tree model with rnd hyper tuning
        start_time = default_timer()
        self.tree, self.best_params = \
            train_gbrt(self._X, self._y.ravel(),
                       cat_idx=cat_idx,
                       random_state=random_state,
                       n_trees=n_trees)
        runtime = default_timer() - start_time
        print(f"* * * time tree training: {round(runtime)} s")

        # get gbm_model format and dump if necessary
        from entmoot_moo.models.lgbm_processing import order_tree_model_dict
        from entmoot_moo.models.gbm_model import GbmModel

        original_tree_model_dict = self.tree.dump_model()

        if dump_model:
            import json
            with open('tree_dict.json', 'w') as fp:
                json.dump(original_tree_model_dict, fp, indent=4)

        ordered_tree_model_dict = \
            order_tree_model_dict(
                original_tree_model_dict,
                cat_column=cat_idx
            )

        print(f"* * * n_trees = {len(ordered_tree_model_dict)}")
        self._gbm_model = GbmModel(ordered_tree_model_dict)