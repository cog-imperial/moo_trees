from entmoot_moo.models.enting import Enting

def check_model(model, space,
                random_state=None, model_unc="BDD",
                acq_func="LCB", kappa=1.96):

    if model == 'ENTING':
        return Enting(space,
                      acq_func=acq_func,
                      kappa=kappa,
                      model_unc=model_unc,
                      random_state=random_state)