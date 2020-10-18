"""
    [Feature extraction functions]

 - energy:      calculates energy of signal
 - rmse:        calculates root-mean-square energy of signal
 - pitch:       estimates pitch (fundamental) frequency
 - formants:    estimates formant frequencies
 - mfccs:       calculate mel frequency cepstrum coefficients

"""
import math
import settings
import numpy as np
import pysptk as st
import librosa as lr
from utils import preemp, stopwatch, windowed


@stopwatch
def energy(y,
           hop_length=settings.HOP_LENGTH,
           frame_length=settings.FRAME_LENGTH):
    """ Calculate energy, the total magnitude of signal.
    Basically, how "loud" the signal is.

    :param y:   samples (signal)

    :returns:   energy
    """
    return np.array([(y[i:i + frame_length]**2).sum()
                     for i in np.arange(0, y.shape[0], hop_length)])


@stopwatch
def rmse(y, hop_length=settings.HOP_LENGTH,
         frame_length=settings.FRAME_LENGTH):
    """ Calculate root-mean-square energy of signal.
    It's defined as the area under the squared
    magnitude of the considered signal.

    :param y:   samples (signal)

    :returns:   RMS energy
    """
    return lr.feature.rmse(y, hop_length=256, frame_length=512,
                           center=True).flatten()


@stopwatch
def pitch(y, sr, low=50, high=2000):
    """ Estimate pitch using autocorrelation.
    Frequency bounds (low and high) must be defined.

    Algorithm comparison @ https://bit.ly/2Vw8y2A
    TODO | check out pysptk.rapt and pysptk.swipe

    :param y:       samples (signal)
    :param sr:      sample rate
    :param low:     low frequency bound
    :param high:    high frequency bound

    :returns:       estimated pitch
    """
    raise NotImplementedError
    r = lr.autocorrelate(y, max_size=5000)

    # Calc boundary indexes
    il = int(sr / high)
    im = int(sr / low)
    r[:il] = 0
    r[im:] = 0

    # Find peak index
    t_max = r.argmax()

    #  Divide for pitch estimation
    f0 = float(sr) / t_max
    return f0


@stopwatch
def formants(y, sr):
    """ Estimate formants of signal using
    Linear Predictive Coding (LPC).

    :param y:       samples (signal)
    :param sr:      sample rate
    :param preemp:  pre-emphasis filter coefficient

    :returns:       formant frequencies (Hz)
    """
    # Rule of thumb according to https://bit.ly/2JlE9Cs
    ncoeff = int(2 + sr / 1000)

    # Get LPC.
    A = st.lpc(y, order=ncoeff)

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) > 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.
    freqs = np.sort(angz * (sr / (2 * math.pi)))

    # TODO: Bandwidth calculation.

    return freqs


@stopwatch
def dsccs(mfccs):
    """ Calculate delta-spectral cepstral coefficients.

    TODO | Implementation according to paper @ https://bit.ly/2W8HIC0

    """
    raise NotImplementedError


@stopwatch
def extract_from(y, sr):
    """ Extract features from samples (signal).

    Feature collection inspired by paper @ https://bit.ly/2VjP0xX.

    :param y:       samples (signal)
    :param sr:      sample rate

    :returns:       features
    """
    # remove silence
    y, _ = lr.effects.trim(y)

    # window and pre-emphasis (filter)
    wy = windowed(y)
    pwy = preemp(wy)

    e0 = energy(y)
    # e1 = lr.feature.delta(e0, order=1)
    # e2 = lr.feature.delta(e0, order=2)

    # log normalize
    e0 = np.log1p(e0).mean()
    # e1 = np.log1p(e1).mean()
    # e2 = np.log1p(e2).mean()

    # calculate formants
    fx = formants(pwy, sr)

    # take only vocal formants
    fx = fx[:3]

    # calculate mfccs
    m0 = lr.feature.mfcc(y, sr=sr, n_mfcc=13, hop_length=settings.HOP_LENGTH)

    # drop 1st (constant) coefficient
    # TODO: discutable, might verify relevance
    # m0 = m0[1:]

    # calculate delta mfccs
    # m1 = lr.feature.delta(m0, order=1)

    # calculate delta-delta mfccs
    # m2 = lr.feature.delta(m0, order=2)

    # get mean rate per sample
    m0 = m0.mean(axis=1)
    # m1 = m1.mean(axis=1)
    # m2 = m2.mean(axis=1)

    features = np.array([e0, *fx, *m0])  # *m1])  #, *m2])
    return features
