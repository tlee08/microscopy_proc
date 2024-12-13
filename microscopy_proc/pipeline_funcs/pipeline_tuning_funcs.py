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
    def make_tuning_arr(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
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

    ###################################################################################################
    # RUN CELL COUNTING PIPELINE WITH TUNING FILEPATHS
    ###################################################################################################

    @classmethod
    def img_overlap(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        pfm = pfm.copy().convert_to_tuning()
        print(pfm.raw)
        super().img_overlap(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc1(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc1(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc2(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc2(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc3(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc3(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc4(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc4(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc5(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc5(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc6(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc6(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc7(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc7(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc8(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc8(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc9(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc9(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc10(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc10(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc11(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc11(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc_coords_only(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc_coords_only(pfm.copy().convert_to_tuning(), overwrite=overwrite)
