import dask.array as da

from microscopy_proc.pipeline.pipeline import Pipeline
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import disk_cache
from microscopy_proc.utils.io_utils import read_json
from microscopy_proc.utils.logging_utils import init_logger, log_func_decorator
from microscopy_proc.utils.proj_org_utils import ProjFpModel


class PipelineTuning(Pipeline):
    logger = init_logger()

    ###################################################################################################
    # CROP RAW ZARR TO MAKE TUNING ZARR
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def make_tuning_arr(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Crop raw zarr to make a smaller zarr for tuning the cell counting pipeline.
        """
        cls.logger.debug("Reading config params")
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        cls.logger.debug("Reading raw zarr")
        raw_arr = da.from_zarr(pfm.raw)
        cls.logger.debug("Cropping raw zarr")
        raw_arr = raw_arr[
            slice(*configs.tuning_z_trim),
            slice(*configs.tuning_y_trim),
            slice(*configs.tuning_x_trim),
        ]
        cls.logger.debug("Converting pfm to tuning filepaths (copy)")
        pfm = pfm.copy().convert_to_tuning()
        if not overwrite and cls._check_file_exists(pfm, ("raw",)):
            cls.logger.debug("Don't overwrite specified and raw zarr exists. Skipping.")
            return
        cls.logger.debug("Saving cropped raw zarr")
        raw_arr = disk_cache(raw_arr, pfm.raw)

    ###################################################################################################
    # RUN CELL COUNTING PIPELINE WITH TUNING FILEPATHS
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def img_overlap(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().img_overlap(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc1(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc1(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc2(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc2(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc3(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc3(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc4(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc4(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc5(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc5(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc6(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc6(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc7(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc7(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc8(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc8(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc9(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc9(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc10(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc10(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc11(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc11(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    @log_func_decorator(logger)
    def cellc_coords_only(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        super().cellc_coords_only(pfm.copy().convert_to_tuning(), overwrite=overwrite)
