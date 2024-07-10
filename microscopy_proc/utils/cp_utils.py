import logging

import cupy as cp
import numpy as np


def clear_cuda_memory():
    # Also removing ALL references to the arguments
    logging.debug("Removing all cupy arrays in program (global and local)")
    all_vars = {**globals(), **locals()}
    var_keys = set(all_vars.keys())
    for k in var_keys:
        if isinstance(all_vars[k], cp.ndarray):
            logging.debug(f"REMOVING: {k}")
            exec("del k")
    logging.debug("Removed all cupy arrays")
    logging.debug("Clearing CUDA memory")
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    logging.debug("Cleared CUDA memory")


def clear_cuda_memory_decorator(func):
    def wrapper(*args, **kwargs):
        logging.debug("Clearing CUDA memory before func")
        clear_cuda_memory()
        logging.debug("Running func")
        res = func(*args, **kwargs)
        logging.debug("Clearing CUDA memory after func")
        clear_cuda_memory()
        logging.debug("Returning result")
        return res

    return wrapper


def np_2_cp_decorator(in_type=cp.float32, out_type=np.uint16):
    def decorator(func):
        def wrapper(arr, *args, **kwargs):
            logging.debug("Converting to cupy array (float32)")
            arr = cp.asarray(arr, dtype=in_type)
            logging.debug("Running func")
            res = func(arr, *args, **kwargs)
            logging.debug("Converting back to numpy (uint16)")
            res_np = res.get().astype(out_type)
            logging.debug("Returning result")
            return res_np

        return wrapper

    return decorator


def check_cuda_memory():
    logging.info(cp.get_default_memory_pool().used_bytes())
    logging.info(cp.get_default_memory_pool().n_free_blocks())
    logging.info(cp.get_default_pinned_memory_pool().n_free_blocks())
