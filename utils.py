import numpy as np

def load_object_by_ext(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".json":
        with open(filename, "r") as fl:
            return json.load(fl)
    elif ext in [".pkl", ".pckl", ".pickl", ".pickle"]:
        with open(filename, "rb") as fl:
            return pickle.load(fl)
    else:
        raise "format not supported" + ext


def save_object_by_ext(obj, filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".json":
        with open(filename, "w") as fl:
            return json.dump(obj, fl)
    elif ext in [".pkl", ".pckl", ".pickl", ".pickle"]:
        with open(filename, "wb") as fl:
            return pickle.dump(obj, fl)
    else:
        raise "format not supported" + ext

def get_lines_from_file(filename, lines, normalize=lambda x: x):
    with open(filename) as fl:
        text = np.array(fl.readlines())
        if lines is not None:
            lines = np.array(lines)
            text = text[lines]
        return (normalize(line.replace("\n", "")) for line in text)