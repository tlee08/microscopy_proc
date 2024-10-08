import cupy as cp
from cupyx.scipy import ndimage as cp_ndimage

# from prefect import task
from microscopy_proc.funcs.cpu_cellc_funcs import CpuCellcFuncs
from microscopy_proc.utils.cp_utils import clear_cuda_mem_dec


class GpuCellcFuncs(CpuCellcFuncs):
    xp = cp
    xdimage = cp_ndimage

    @classmethod
    # @task
    def tophat_filt(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().tophat_filt)(*args, **kwargs).get()

    @classmethod
    # @task
    def dog_filt(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().dog_filt)(*args, **kwargs).get()

    @classmethod
    # @task
    def gauss_blur_filt(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().gauss_blur_filt)(*args, **kwargs).get()

    @classmethod
    # @task
    def gauss_subt_filt(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().gauss_subt_filt)(*args, **kwargs).get()

    @classmethod
    # @task
    def intensity_cutoff(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().intensity_cutoff)(*args, **kwargs).get()

    @classmethod
    # @task
    def otsu_thresh(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().otsu_thresh)(*args, **kwargs).get()

    @classmethod
    # @task
    def mean_thresh(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().mean_thresh)(*args, **kwargs).get()

    @classmethod
    # @task
    def manual_thresh(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().manual_thresh)(*args, **kwargs).get()

    @classmethod
    # @task
    def mask2ids(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().mask2ids)(*args, **kwargs).get()

    @classmethod
    # @task
    def label_with_volumes(cls, *args, **kwargs):
        # NOTE: Already returns a numpy array
        return clear_cuda_mem_dec(super().label_with_volumes)(*args, **kwargs)

    @classmethod
    # @task
    def ids2volumes(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().ids2volumes)(*args, **kwargs).get()

    @classmethod
    # @task
    def visualise_stats(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().visualise_stats)(*args, **kwargs)

    @classmethod
    # @task
    def volume_filter(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().volume_filter)(*args, **kwargs).get()

    @classmethod
    # @task
    def get_local_maxima(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().get_local_maxima)(*args, **kwargs).get()

    @classmethod
    # @task
    def mask(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().mask)(*args, **kwargs).get()

    @classmethod
    # @task
    def wshed_segm(cls, *args, **kwargs):
        # NOTE: This is a CPU function
        return clear_cuda_mem_dec(super().wshed_segm)(*args, **kwargs)

    @classmethod
    # @task
    def wshed_segm_volumes(cls, *args, **kwargs):
        # NOTE: This is a CPU function
        return clear_cuda_mem_dec(super().wshed_segm_volumes)(*args, **kwargs)

    @classmethod
    # @task
    def get_coords(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().get_coords)(*args, **kwargs)

    @classmethod
    # @task
    def get_cells(cls, *args, **kwargs):
        return clear_cuda_mem_dec(super().get_cells)(*args, **kwargs)
