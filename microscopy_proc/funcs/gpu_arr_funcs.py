import logging

import cupy as cp
from cupyx.scipy import ndimage as cp_ndimage

from microscopy_proc.funcs.cpu_arr_funcs import CpuArrFuncs


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


class GpuArrFuncs(CpuArrFuncs):
    xp = cp
    xdimage = cp_ndimage

    @classmethod
    def tophat_filt(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().tophat_filt)(*args, **kwargs).get()

    @classmethod
    def dog_filt(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().dog_filt)(*args, **kwargs).get()

    @classmethod
    def gauss_subt_filt(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().gauss_subt_filt)(*args, **kwargs).get()

    @classmethod
    def intensity_cutoff(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().intensity_cutoff)(*args, **kwargs).get()

    @classmethod
    def otsu_thresh(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().otsu_thresh)(*args, **kwargs).get()

    @classmethod
    def mean_thresh(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().mean_thresh)(*args, **kwargs).get()

    @classmethod
    def manual_thresh(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().manual_thresh)(*args, **kwargs).get()

    @classmethod
    def label_objects_with_ids(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().label_objects_with_ids)(*args, **kwargs).get()

    @classmethod
    def label_objects_with_sizes(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().label_objects_with_sizes)(
            *args, **kwargs
        ).get()

    @classmethod
    def get_sizes(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().get_sizes)(*args, **kwargs)

    @classmethod
    def labels_map(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().labels_map)(*args, **kwargs).get()

    @classmethod
    def visualise_stats(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().visualise_stats)(*args, **kwargs)

    @classmethod
    def filter_by_size(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().filter_by_size)(*args, **kwargs).get()

    @classmethod
    def get_local_maxima(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().get_local_maxima)(*args, **kwargs).get()

    @classmethod
    def mask(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().mask)(*args, **kwargs).get()

    @classmethod
    def watershed_segm(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().watershed_segm)(*args, **kwargs).get()

    @classmethod
    def region_to_coords(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().region_to_coords)(*args, **kwargs)

    @classmethod
    def maxima_to_coords(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().maxima_to_coords)(*args, **kwargs)
