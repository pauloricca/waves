import yaml
from models import SoundLibraryModel


sound_library: SoundLibraryModel = None

def load_sound_library(file_path: str) -> SoundLibraryModel:
    global sound_library
    with open(file_path) as file:
        raw_data = yaml.safe_load(file)
    sound_library = SoundLibraryModel.model_validate(raw_data)

def get_sound_model(sound_name: str):
    if sound_library is None:
        raise ValueError("Sound library not loaded. Please load the sound library first.")
    
    if sound_name not in sound_library.keys():
        raise ValueError(f"Sound '{sound_name}' not found in the sound library.")
    
    return sound_library[sound_name].model_copy(deep=True)
