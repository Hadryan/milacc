import settings
import features as fts


def probs(clfs, signal, sample_rate=settings.SAMPLE_RATE):
    """ Use multiple classifiers for class estimation.
    ! Use only with set of binary classifiers.

    :param clfs:          classifiers
    :param signal:        signal (samples)
    :param sample_rate:   sample rate

    :returns:             guesses and their probabilities
    """
    # extract features from raw signal
    features = fts.extract_from(signal, sample_rate)

    candidates = []
    for emotion, clf in clfs.items():
        # predict probability of outcome classes
        prediction = clf.predict_proba([features])

        # pick probability of trained class
        prediction = prediction[0][1]
        candidates.append((emotion, prediction))

    # sort descending
    return sorted(candidates, key=lambda c: -c[1])
