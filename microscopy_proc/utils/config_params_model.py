from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from microscopy_proc.constants import (
    DEPTH,
    PROC_CHUNKS,
    RESOURCES_DIR,
)
from microscopy_proc.utils.io_utils import read_json, write_json


class RefVersions(Enum):
    AVERAGE_TEMPLATE_25 = "average_template_25"
    ARA_NISSL_25 = "ara_nissl_25"


class AnnotVersions(Enum):
    CCF_2017_25 = "ccf_2017_25"
    CCF_2016_25 = "ccf_2016_25"
    CCF_2015_25 = "ccf_2015_25"


class MapVersions(Enum):
    ABA_ANNOTATIONS = "ABA_annotations"
    CM_ANNOTATIONS = "CM_annotations"


class ConfigParamsModel(BaseModel):
    """
    Pydantic model for registration parameters.
    """

    model_config = ConfigDict(
        extra="forbid",
        # arbitrary_types_allowed=True,
        validate_default=True,
        use_enum_values=True,
    )

    # REFERENCE
    atlas_dir: str = RESOURCES_DIR
    ref_version: RefVersions = RefVersions.AVERAGE_TEMPLATE_25
    annot_version: AnnotVersions = AnnotVersions.CCF_2016_25
    map_version: MapVersions = MapVersions.ABA_ANNOTATIONS
    # RAW
    zarr_chunksize: tuple[int, int, int] = PROC_CHUNKS
    # REGISTRATION
    ref_orient_ls: tuple[int, int, int] = (1, 2, 3)
    ref_z_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    ref_y_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    ref_x_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    z_rough: int = 3
    y_rough: int = 6
    x_rough: int = 6
    z_fine: float = 1.0
    y_fine: float = 0.6
    x_fine: float = 0.6
    z_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    y_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    x_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    # MASK
    mask_gaus_blur: int = 1
    mask_thresh: int = 300
    # CELL COUNT TUNING CROP
    tuning_z_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    tuning_y_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    tuning_x_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    # OVERLAP
    overlap_depth: int = DEPTH
    # CELL COUNTING
    tophat_sigma: int = 10
    dog_sigma1: int = 1
    dog_sigma2: int = 4
    large_gauss_sigma: int = 101
    threshd_value: int = 60
    min_threshd_size: int = 100
    max_threshd_size: int = 10000
    maxima_sigma: int = 10
    min_wshed_size: int = 1
    max_wshed_size: int = 1000
    # VISUAL CHECK
    heatmap_raw_radius: int = 5
    heatmap_trfm_radius: int = 3
    # COMBINE ARRAYS
    combine_cellc_z_trim: tuple[int | None, int | None, int | None] = (0, 10, None)
    combine_cellc_y_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    combine_cellc_x_trim: tuple[int | None, int | None, int | None] = (None, None, None)

    @model_validator(mode="after")
    def validate_trims(self):
        # Orient validation
        vect = np.array(self.ref_orient_ls)
        vect_abs = np.abs(vect)
        vect_abs_sorted = np.sort(vect_abs)
        assert np.all(vect_abs_sorted == np.array([1, 2, 3]))
        # TODO: Size validation
        # TODO: Trim validation
        return self

    def update(self, **kwargs):
        return self.model_validate(self.model_copy(update=kwargs))

    @classmethod
    def update_file(cls, fp: str, **kwargs):
        """
        Reads the json file in `fp`, updates the parameters with `kwargs`,
        writes the updated parameters back to `fp` (if there are any updates),
        and returns the model instance.
        """
        configs = cls.model_validate(read_json(fp))
        # Updating and saving if kwargs is not empty
        if kwargs != {}:
            configs = cls.model_validate(configs.model_copy(update=kwargs))
            write_json(fp, configs.model_dump())
        return configs
