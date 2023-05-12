import os

from typing import List

def iterate_files(directory: str) -> List:
    """Iterates over the files in the given directory and returns a list of 
    found files."""
    files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        fullpath = os.path.join(directory, filename)
        if os.path.isdir(fullpath):
            files += iterate_files(fullpath)
        else:
            files.append(fullpath)
    return files


def get_model_history_files(directory: str) -> List:
    """Returns a list of pkl model history files in the given directory. 
    Checks to make sure that a model file of the same name exists."""
    files = iterate_files(directory)
    # Sort files by name
    files.sort()
    
    model_ext = ".pth"
    history_ext = ".pkl"
    
    # Get the history files if a model file of the same name exists
    model_history_files = []
    for file in files:
        if file.endswith(history_ext):
            model_file = file.replace(history_ext, model_ext)
            if model_file in files:
                model_history_files.append(file)
    return model_history_files


if __name__ == "__main__":
    directory = "models/task1"
    model_history_files = get_model_history_files(directory)
    
    for file in model_history_files:
        print(file)
        
