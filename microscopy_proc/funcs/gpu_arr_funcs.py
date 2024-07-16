import cupy as cp
from cupyx.scipy import ndimage as cp_ndimage

from microscopy_proc.funcs.cpu_arr_funcs import CpuArrFuncs
from microscopy_proc.utils.cp_utils import clear_cuda_mem_dec


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
    def label_with_ids(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().label_with_ids)(*args, **kwargs).get()

    @classmethod
    def label_with_sizes(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().label_with_sizes)(*args, **kwargs).get()

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
    def filt_by_size(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().filt_by_size)(*args, **kwargs).get()

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
