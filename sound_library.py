import yaml

from models.sound_library_model import SoundLibraryModel


sound_library: SoundLibraryModel = None

def load_sound_library(file_path: str) -> SoundLibraryModel:
    global sound_library
    with open(file_path) as file:
        raw_data = yaml.safe_load(file)
    try:
        sound_library = SoundLibraryModel.model_validate(raw_data)
    except Exception as e:
        print(f"Error loading sound library: {e}")
    return sound_library

def get_sound_model(sound_name: str):
    if sound_library is None:
        raise ValueError("Sound library not loaded. Please load the sound library first.")
    
    if sound_name not in sound_library.keys():
        raise ValueError(f"Sound '{sound_name}' not found in the sound library.")
    
    return sound_library[sound_name]
