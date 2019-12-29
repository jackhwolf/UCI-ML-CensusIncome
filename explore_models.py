import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier as DTClf
from util import preprocessor, splitfuncs, metrics, resultsmngr

'''
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
'''

# main methods to explore different models
def explore_models(*args):
    # setup our data
    pre = preprocessor()
    trx, try_ = pre.data
    del pre
    trx, try_, tex, tey = splitfuncs.splitBalanced(trx, try_)
    # setup our models
    models = {
        'SVC':       {'model': SVC,       'params': [None, {'C': 0.75}, {'C': 1.25}]},
        'LinearSVC': {'model': LinearSVC, 'params': [None, {'C': 0.75}, {'C': 1.25}]},
        'TreeClf':   {'model': DTClf,     'params': [None, {'max_depth': 5}, {'max_depth': 10}]}
    } if not args else args[0]
    # train all of the models
    for m in models:
        mdl = models[m]
        for i, p in enumerate(mdl['params']):
            # parse params and init model
            params = p if p is not None else {}
            print(m, params)
            model = mdl['model'](**params)
            # train and predict
            model.fit(trx, try_)
            trpreds = model.predict(trx)  # sanity check
            tepreds = model.predict(tex)
            # record metrics
            models[m]['params'][i] = [models[m]['params'][i], {
                'training-accuracy': metrics.acc(trpreds, try_),
                'testing-accuracy':  metrics.acc(tepreds, tey),
            }]
    return models


if __name__ == '__main__':
    fname = resultsmngr.save(explore_models())
    print(resultsmngr.load(fname))
    print(fname)
