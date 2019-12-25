import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier as DTClf
from util import preprocessor, splitfuncs, metrics


'''
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
'''

# main methods to test model performance
def testmodels(*args):
    # setup our data
    pre = preprocessor()
    trx, try_ = pre.data
    del pre
    trx, try_, tex, tey = splitfuncs.split(trx, try_)

    # setup our models
    models = {
        'svc':  {'model': SVC,       'params': [[None]]},
        'lsvc': {'model': LinearSVC, 'params': [[None]]},
        'tclf': {'model': DTClf,     'params': [[None]]}
    } if not args else args[0]

    # train all of the models
    for m in models:
        mdl = models[m]
        for i, p in enumerate(mdl['params']):
            # parse params and init model
            params = p[0] if p[0] is not None else {}
            model = mdl['model'](**params)
            # train and predict
            model.fit(trx, try_)
            trpreds = model.predict(trx)  # sanity check
            tepreds = model.predict(tex)
            # record metrics
            models[m]['params'][i].append({
                'training-accuracy': metrics.acc(trpreds, try_),
                'testing-accuracy':  metrics.acc(tepreds, tey),
            })
            
    return models


if __name__ == '__main__':
    print(testmodels())