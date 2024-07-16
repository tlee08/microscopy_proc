import logging

import cupy as cp


def clear_cuda_mem():
    # Also removing ALL references to the arguments
    logging.debug("Removing all cupy arrays in program (global and local)")
    all_vars = {**globals(), **locals()}
    var_keys = set(all_vars.keys())
    for k in var_keys:
        if isinstance(all_vars[k], cp.ndarray):
            logging.debug(f"REMOVING: {k}")
            exec("del k")
    logging.debug("Clearing CUDA memory")
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def clear_cuda_mem_dec(func):
    def wrapper(*args, **kwargs):
        clear_cuda_mem()
        res = func(*args, **kwargs)
        clear_cuda_mem()
        return res

    return wrapper


def check_cuda_mem():
    logging.info(cp.get_default_memory_pool().used_bytes())
    logging.info(cp.get_default_memory_pool().n_free_blocks())
    logging.info(cp.get_default_pinned_memory_pool().n_free_blocks())
