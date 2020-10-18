import settings
import numpy as np
import librosa as lr
from utils import stopwatch


@stopwatch
def on_silence(y,
               merge=False,
               hop_length=settings.HOP_LENGTH,
               frame_length=settings.FRAME_LENGTH):
    """ Split signal into non-silent intervals.

    :param y:       samples (signal)
    :param merge:   merge splits into one signal

    :returns:       signal (or segments), splits
    """
    splits = lr.effects.split(y,
                              hop_length=hop_length,
                              frame_length=frame_length)

    # merge signal using provided indices
    if merge:
        return lr.effects.remix(y, intervals=splits)

    # return audio chunks based on indices
    return np.asarray([y[s:e] for s, e in splits]), splits


@stopwatch
def on_onsets(y,
              sr=settings.SAMPLE_RATE,
              backtrack=True,
              hop_length=settings.HOP_LENGTH):
    """ Split signal based on detected onsets.
    Optionally try to locate nearest silent spot.

    :param y:           samples (signal)
    :param sr:          sample rate
    :param backtrack:   locate silent spots

    :returns:           array, onsets
    """
    # extract backtracked onsets
    onsets = lr.onset.onset_detect(y=y,
                                   sr=sr,
                                   units='samples',
                                   backtrack=backtrack,
                                   hop_length=hop_length)

    # prepend first sample index
    onsets = np.insert(onsets, 0, 0)

    # append last sample index
    onsets = np.append(onsets, y.shape[0])

    # chunk to individual segments
    splits = [y[onsets[i]:onsets[i + 1]] for i, _ in enumerate(onsets[:-1])]
    return splits, onsets
