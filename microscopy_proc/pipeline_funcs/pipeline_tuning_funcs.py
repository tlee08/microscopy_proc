import functools
from typing import Callable

import dask.array as da

from microscopy_proc.pipeline_funcs.pipeline_funcs import (
    PipelineFuncs,
    overwrite_check_decorator,
)
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import (
    disk_cache,
)
from microscopy_proc.utils.io_utils import read_json
from microscopy_proc.utils.proj_org_utils import (
    ProjFpModel,
)


class PipelineTuningFuncs(PipelineFuncs):
    ###################################################################################################
    # CROP RAW ZARR TO MAKE TUNING ZARR
    ###################################################################################################

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("overlap",))
    def make_tuning_img(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Crop raw zarr to make a smaller zarr for tuning the cell counting pipeline.
        """
        # Getting configs
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        # Reading
        raw_arr = da.from_zarr(pfm.raw)
        # Cropping
        raw_arr = raw_arr[
            slice(*configs.tuning_z_trim),
            slice(*configs.tuning_y_trim),
            slice(*configs.tuning_x_trim),
        ]
        # Converting to tuning filepaths
        pfm = pfm.copy()
        pfm.convert_to_tuning()
        # Saving
        raw_arr = disk_cache(raw_arr, pfm.raw)

    # @classmethod
    # def img_overlap(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
    #     # Converting to tuning filepaths
    #     pfm = pfm.copy()
    #     pfm.convert_to_tuning()
    #     # Running process
    #     super().img_overlap(pfm, overwrite)

    # @classmethod
    # def cellc1(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
    #     # Converting to tuning filepaths
    #     pfm = pfm.copy()
    #     pfm.convert_to_tuning()
    #     # Running process
    #     super().img_overlap(pfm, overwrite)


###################################################################################################
# DYNAMICALLY MAKING PROJECT PIPELINE METHODS (FROM EXPERIMENT METHODS) FOR PROJECT CLASS
###################################################################################################


def create_wrapped_method(func: Callable):
    @functools.wraps(func)
    def wrapper(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        # Converting pfm to tuning filepaths
        pfm = pfm.copy()
        pfm.convert_to_tuning()
        # Running process
        print(func)
        print(func.__doc__)
        func(pfm, overwrite)

    return wrapper


# Dynamically add methods to Project class
for func in [
    PipelineFuncs.img_overlap,
    PipelineFuncs.cellc1,
    PipelineFuncs.cellc2,
    PipelineFuncs.cellc3,
    PipelineFuncs.cellc4,
    PipelineFuncs.cellc5,
    PipelineFuncs.cellc6,
    PipelineFuncs.cellc7,
    PipelineFuncs.cellc8,
    PipelineFuncs.cellc9,
    PipelineFuncs.cellc10,
    PipelineFuncs.cellc11,
    PipelineFuncs.cellc_coords_only,
]:
    func_name = func.__name__
    setattr(PipelineTuningFuncs, func_name, create_wrapped_method(func))
