from importlib import resources
import os
import functools
import random
import inflect
import re

IE = inflect.engine()
ASSETS_PATH = resources.files("assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def from_file(path, low=None, high=None, return_all=False):
    prompts = _load_lines(path)[low:high]
    if return_all:
        return prompts, {}
    else:
        return random.choice(prompts), {}

def simple_animals(return_all=False):
    return from_file("simple_animals.txt", return_all=return_all)

def hps_v2_all(return_all=False):
    return from_file("hps_v2_all.txt", return_all=return_all)

def eval_simple_animals(return_all=False):
    return from_file("eval_simple_animals.txt", return_all=return_all)

def eval_hps_v2_all(return_all=False):
    return from_file("hps_v2_all_eval.txt", return_all=return_all)

# def from_file(path, low=None, high=None):
#     prompts = _load_lines(path)[low:high]
#     return random.choice(prompts), {}

# def hps_v2_all():
#     return from_file("hps_v2_all.txt")

# def simple_animals():
#     return from_file("simple_animals.txt")

# def eval_simple_animals():
#     return from_file("eval_simple_animals.txt")

# def eval_hps_v2_all():
#     return from_file("hps_v2_all_eval.txt")


def sanitize_prompt(s: str, max_len: int = 30) -> str:
    """
    Preprocess prompt for saving img, preserving spaces,
    replacing underscores and special characters with spaces,
    and avoiding trailing spaces.
    """
    # Replace non-alphanumeric characters and underscores with spaces
    s = re.sub(r'[^0-9A-Za-z가-힣 ]', ' ', s)
    # Trim and collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # Apply max length limit
    if len(s) > max_len:
        s = s[:max_len].rstrip()

    return s