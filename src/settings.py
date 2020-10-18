import os

# path to base folder
BASE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# absolute path to data / fixtrues / mocks
DATA_PATH = os.path.join(BASE_PATH, 'data')

# absolute path to models
MODELS_PATH = os.path.join(BASE_PATH, 'models')

# default sample rate
SAMPLE_RATE = 22050

# default hop length
HOP_LENGTH = 256

# default frame length
FRAME_LENGTH = 512

# version
VERSION = '0.0.1'
