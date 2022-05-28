import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from django.apps import AppConfig
from pathlib import Path
from django.conf import settings
import pickle
from  keras.models import model_from_json
import requests
from urllib3.exceptions import InsecureRequestWarning

class AnalyzerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analyzer'

    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    tokenizer_path = Path.joinpath(settings.MODELS_DIR, 'tokenizer.pickle')
    weights_path = Path.joinpath(settings.MODELS_DIR, 'model.hdf5')
    model_path = Path.joinpath(settings.MODELS_DIR, 'model.json')

    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
            
    with open(model_path, 'r') as f:
        loaded_model_json = f.read()
    loaded_model =model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path) 