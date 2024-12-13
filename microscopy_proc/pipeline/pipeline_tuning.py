import dask.array as da

from microscopy_proc.pipeline.pipeline import Pipeline
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import (
    disk_cache,
)
from microscopy_proc.utils.io_utils import read_json
from microscopy_proc.utils.proj_org_utils import (
    ProjFpModel,
)


class PipelineTuning(Pipeline):
    ###################################################################################################
    # CROP RAW ZARR TO MAKE TUNING ZARR
    ###################################################################################################

    @classmethod
    def make_tuning_arr(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Crop raw zarr to make a smaller zarr for tuning the cell counting pipeline.
        """
        if not overwrite and cls._check_file_exists(pfm, ("overlap",)):
            return
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
        pfm = pfm.convert_to_tuning()
        # Saving
        raw_arr = disk_cache(raw_arr, pfm.raw)

    ###################################################################################################
    # RUN CELL COUNTING PIPELINE WITH TUNING FILEPATHS
    ###################################################################################################

    @classmethod
    def img_overlap(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        pfm = pfm.convert_to_tuning()
        print(pfm)
        cls.img_overlap(pfm, overwrite=overwrite)

    @classmethod
    def cellc1(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc1(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc2(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc2(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc3(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc3(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc4(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc4(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc5(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc5(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc6(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc6(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc7(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc7(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc8(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc8(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc9(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc9(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc10(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc10(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc11(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc11(pfm.copy().convert_to_tuning(), overwrite=overwrite)

    @classmethod
    def cellc_coords_only(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        cls.cellc_coords_only(pfm.copy().convert_to_tuning(), overwrite=overwrite)
