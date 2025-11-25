import os
import re
import unicodedata

# Returns the absolute path to the project root directory.
def get_project_root():
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))
    return project_root

# Returnst he absolute path to the raw audio data directory.
def get_raw_audio_path():
    return os.path.join(get_project_root(), "data", "raw_audio")

# Sanitizes the given string for usage in a path.
def sanitize_for_path(s):

    # Normalize full-width to ASCII when equivalent
    s = unicodedata.normalize("NFKC", s)

    # Replace anything that is not alphanumeric, dash, or underscore
    s = re.sub(r"[^\w\-]", "_", s)

    s = s.strip()
    
    return s 