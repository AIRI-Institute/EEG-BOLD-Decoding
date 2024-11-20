import json
import math
import os
from typing import Optional

import numpy as np


def run_name_from_parameters(delay: int, sub: str, suffix: str):
    """
    Return a run name from its core parameters

    Parameters
    ----------
    delay : int
        Delay used in the run
    sub : str
        Subject in the run
    suffix : str
        A suffix of the run name (e.g. Subcort, SubcortTrend, ...)

    Returns
    -------
    run_name : str
        A name of the run
    """
    return f'Delay{delay}Sub{sub}{suffix}'


def get_run_name(run_id: str):
    """
    Return the name of the run from path to fMRI file

    Parameters
    ----------
    run_id : str
        Path to the fMRI file

    Returns
    -------
    run_name : str
        Name of the run associated with the file
    """
    return os.path.splitext(os.path.basename(run_id))[0]


def is_integer_with_custom_precision(num: float, eps: float = 1e-6, int_value: Optional[int] = None):
    """
    Checks if num is an integer equal to int_value with set precision

    Parameters
    ----------
    num : float
        A float which should be checked for being a whole number int_value
    eps : float
        A precision level. Default: 1e-6
    int_value : int or None
        A whole number which float num is checked for being. If None, int(num) is used. Default: None

    Returns
    -------
    check_result : bool
        True if num == int_value, False otherwise
    int_value : int
        An int whole number that num was checked against
    """
    if int_value is None:
        int_value = int(num)
    return abs(int_value - num) < eps, int_value


def normalize_time_series(series: np.ndarray, axis: Optional[int], keepdims: bool = True):
    """
    Normalizes time-series over desired axis

    Parameters
    ----------
    series : np.ndarray
        Time-series
    axis : int or None
        Axis for normalization
    keepdims : bool
        If True means and standard deviations will keep dimensions of the series. Default: True

    Returns
    -------
    normalized_series : np.ndarray
        Normalized time-series
    """
    return (series - np.mean(series, axis=axis, keepdims=keepdims)) / np.std(series, axis=axis, keepdims=keepdims)


def get_window_sizes_sample(size_sec: float, sampling_rate: int):
    """
    Get a size of the window in samples

    Parameters
    ----------
    size_sec : float
        A size of the window in seconds
    sampling_rate : int
        Sampling rate

    Returns
    -------
    window_size : int
        A size of the window in samples
    """
    return math.ceil(size_sec * sampling_rate)


def load_json(json_path: str, encoding: Optional[str] = None):
    """
    Loads a json file into a dict

    Parameters
    ----------
    json_path : str
        A path to the json file
    encoding : str or None
        A specific encoding to use while reading json. If None, uses default encoder. Default: None

    Returns
    -------
    jsf : dict
        json file as a dict
    """
    if encoding is None:
        with open(json_path) as f:
            jsf = json.load(f)
    else:
        with open(json_path, encoding=encoding) as f:
            jsf = json.load(f)
    return jsf


def save_json(save_path: str, data: dict):
    """
    Save a dict into a json file

    Parameters
    ----------
    save_path : str
        A path to save file
    data : dict
        A dict with data to be saved
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def print_wrapped_text(header_text, main_body, wrap_length=71, wrap_symbol='#'):
    """
    Prints a header wrapped around into a wrap_symbol and the main_body afterward

    Parameters
    ----------
    header_text : str
        A header of the text
    main_body : Any
        A main body of the text
    wrap_length : int
        A length of a line in characters. Default: 71
    wrap_symbol : str
        A symbol to use for wrapping. Default: '#'
    """
    assert len(header_text) < wrap_length, (len(header_text), wrap_length)
    if ((wrap_length - len(header_text) - 2) % 2) == 1:
        wrap_length += 1
    sides = (wrap_length - len(header_text) - 2) // 2
    new_text = ((wrap_length * wrap_symbol) + '\n' + (sides * wrap_symbol) + ' ' + header_text + ' '
                + (sides * wrap_symbol) + '\n' + (wrap_length * wrap_symbol))
    print(new_text)
    print(main_body)
    print(wrap_length * wrap_symbol)
