import pickle
import numpy as np
import settings
import features as fts
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def load_data():
    """ Load data and split to features and classes.

    :returns:   features, classes
    """
    # load frame (signals)
    frame = np.load(f'{settings.DATA_PATH}/audio/actors.frame.npy',
                    allow_pickle=True)

    # extract features
    features = np.array([
        fts.extract_from(sample['signal'], sample['sample_rate'])
        for sample in frame
    ])
    # separate classes
    classes = frame['emotion']

    return (features, classes)


def svm(features, classes, test_train_ratio=0.2, **kwargs):
    """ Train Support Vector Classifier.

    :param features:            features set
    :param classes:             classes set
    :param test_train_ratio:    test/train split ratio

    :returns:                   classifier, accuracy score
    """
    # build pipeline
    pipe = Pipeline([('scale', StandardScaler()),
                     ('svc', SVC(gamma='scale', **kwargs))])

    # search thru parameters
    param_grid = {'svc__C': range(1, 20)}

    # use 5-fold cross-validation
    search = GridSearchCV(pipe, param_grid, iid=False, cv=5)

    # fit all estimators
    search.fit(features, classes)

    # pick best estimator
    est = search.best_estimator_

    # and it's score
    score = search.best_score_

    return est, score


def save(name, model):
    """ Save model to drive.

    :param name:    label for model
    :param model:   model
    """
    with open(f'{settings.MODELS_PATH}/{name}.pkl', 'ab') as f:
        pickle.dump(model, f)


def load(name):
    """ Load model from drive.

    :param name:    label of model

    :returns:       model
    """
    with open(f'{settings.MODELS_PATH}/{name}.pkl', 'rb') as f:
        return pickle.load(f)
