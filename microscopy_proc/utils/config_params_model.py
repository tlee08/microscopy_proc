from pydantic import BaseModel, ConfigDict, model_validator

from microscopy_proc.constants import (
    DEPTH,
    PROC_CHUNKS,
    RESOURCES_DIR,
    AnnotVersions,
    MapVersions,
    RefVersions,
)
from microscopy_proc.utils.io_utils import read_json, write_json


class ConfigParamsModel(BaseModel):
    """
    Pydantic model for registration parameters.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # REFERENCE
    atlas_dir: str = RESOURCES_DIR
    ref_v: str = RefVersions.AVERAGE_TEMPLATE_25.value
    annot_v: str = AnnotVersions.CCF_2016_25.value
    map_v: str = MapVersions.ABA_ANNOTATIONS.value
    # RAW
    chunksize: tuple[int, int, int] = PROC_CHUNKS
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
    # OVERLAP
    depth: int = DEPTH
    # CELL COUNTING
    tophat_sigma: int = 10
    dog_sigma1: int = 1
    dog_sigma2: int = 4
    gauss_sigma: int = 101
    thresh_p: int = 60
    min_threshd: int = 100
    max_threshd: int = 10000
    maxima_sigma: int = 10
    min_wshed: int = 1
    max_wshed: int = 1000

    @model_validator(mode="after")
    def validate_trims(self):
        # Orient validation
        assert len(self.ref_orient_ls) == 3
        assert all(isinstance(i, int) for i in self.ref_orient_ls)
        ref_orient_ls_abs = [abs(i) for i in self.ref_orient_ls]
        assert max(ref_orient_ls_abs) == 3
        assert min(ref_orient_ls_abs) == 1
        assert sum(ref_orient_ls_abs) == 6
        # Size validation
        # Trim validation
        return self

    def update_params(self, **kwargs):
        return self.model_validate(self.model_copy(update=kwargs))

    @classmethod
    def update_params_file(cls, fp: str, **kwargs):
        """
        Reads the json file in `fp`, updates the parameters with `kwargs`,
        writes the updated parameters back to `fp` (if there are any updates),
        and returns the model instance.
        """
        rp = cls.model_validate(read_json(fp))
        # Updating and saving if kwargs is not empty
        if kwargs != {}:
            rp = cls.model_validate(rp.model_copy(update=kwargs))
            write_json(fp, rp.model_dump())
        return rp
