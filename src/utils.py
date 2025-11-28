import os
import re
import unicodedata

# Returns the absolute path to the project root directory.
def get_project_root():
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))
    return project_root

# Returns the absolute path to the raw audio data directory. Creates the directory if it does not exist.
def get_raw_audio_path():
    path = os.path.join(get_project_root(), "data", "raw_audio")
    os.makedirs(path, exist_ok=True)
    return path

# Returns the absolute path to the datasets directory. Creates the directory if it does not exist.
def get_dataset_path():
    path = os.path.join(get_project_root(), "data", "datasets")
    os.makedirs(path, exist_ok=True)
    return path

# Returns the absolute path to the models directory. Creates the directory if it does not exist.
def get_model_path():
    path = os.path.join(get_project_root(), "models")
    os.makedirs(path, exist_ok=True)
    return path

# Sanitizes the given string for usage in a path.
def sanitize_for_path(s):

    # Normalize full-width to ASCII when equivalent
    s = unicodedata.normalize("NFKC", s)

    # Replace anything that is not alphanumeric, dash, or underscore
    s = re.sub(r"[^\w\-]", "_", s)

    # Remove leading and trailing spaces
    s = s.strip()
    
    return s 