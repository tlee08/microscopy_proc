import os
from enum import Enum
from typing import Type

from microscopy_proc.utils.logging_utils import init_logger

# TODO: add 10_adaptv_f.zarr and move the xxx_f.zarr files to a new folder (e.g. "cellcount_final")
# NOTE: this allows "cellcount" to be removed to save space when pipeline is completed and output checked

logger = init_logger(__name__)


class RefFolders(Enum):
    REFERENCE = "reference"
    ANNOTATION = "annotation"
    MAPPING = "region_mapping"
    ELASTIX = "elastix_params"


class RefFpModel:
    """
    Pydantic model for reference file paths.
    """

    atlas_dir: str
    ref_version: str
    annot_version: str
    map_version: str

    def __init__(self, atlas_dir, ref_version, annot_version, map_version):
        """
        atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"

        Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
        """
        self.atlas_dir = atlas_dir
        self.ref_version = ref_version
        self.annot_version = annot_version
        self.map_version = map_version
        logger.debug(f"Getting Atlas RefFpModel for {atlas_dir}")

    @property
    def ref(self):
        reference_dir = RefFolders.REFERENCE.value
        return os.path.join(self.atlas_dir, reference_dir, f"{self.ref_version}.tif")

    @property
    def annot(self):
        annotation_dir = RefFolders.ANNOTATION.value
        return os.path.join(self.atlas_dir, annotation_dir, f"{self.annot_version}.tif")

    @property
    def map(self):
        mapping_dir = RefFolders.MAPPING.value
        return os.path.join(self.atlas_dir, mapping_dir, f"{self.map_version}.json")

    @property
    def affine(self):
        elastix_dir = RefFolders.ELASTIX.value
        return os.path.join(self.atlas_dir, elastix_dir, "align_affine.txt")

    @property
    def bspline(self):
        elastix_dir = RefFolders.ELASTIX.value
        return os.path.join(self.atlas_dir, elastix_dir, "align_bspline.txt")


class ProjFpModelBase:
    """
    Abstract model for project file paths.
    """

    root_dir: str
    subdirs: Type[Enum]

    _config_params: tuple[str | None, str] = (None, "config_params.json")
    _raw: tuple[str | None, str] = ("RAW", "raw.zarr")
    _ref: tuple[str | None, str] = ("REGISTRATION", "0a_reference.tif")
    _annot: tuple[str | None, str] = ("REGISTRATION", "0b_annotation.tif")
    _map: tuple[str | None, str] = ("REGISTRATION", "0c_mapping.json")
    _affine: tuple[str | None, str] = ("REGISTRATION", "0d_align_affine.txt")
    _bspline: tuple[str | None, str] = ("REGISTRATION", "0e_align_bspline.txt")
    _downsmpl1: tuple[str | None, str] = ("REGISTRATION", "1_downsmpl1.tif")
    _downsmpl2: tuple[str | None, str] = ("REGISTRATION", "2_downsmpl2.tif")
    _trimmed: tuple[str | None, str] = ("REGISTRATION", "3_trimmed.tif")
    _bounded: tuple[str | None, str] = ("REGISTRATION", "4_bounded.tif")
    _regresult: tuple[str | None, str] = ("REGISTRATION", "5_regresult.tif")
    _premask_blur: tuple[str | None, str] = ("MASK", "1_premask_blur.tif")
    _mask: tuple[str | None, str] = ("MASK", "2_mask_trimmed.tif")
    _outline: tuple[str | None, str] = ("MASK", "3_outline_reg.tif")
    _mask_reg: tuple[str | None, str] = ("MASK", "4_mask_reg.tif")
    _mask_df: tuple[str | None, str] = ("MASK", "5_mask.parquet")
    _overlap: tuple[str | None, str] = ("CELLCOUNT", "0_overlap.zarr")
    _bgrm: tuple[str | None, str] = ("CELLCOUNT", "1_bgrm.zarr")
    _dog: tuple[str | None, str] = ("CELLCOUNT", "2_dog.zarr")
    _adaptv: tuple[str | None, str] = ("CELLCOUNT", "3_adaptv.zarr")
    _threshd: tuple[str | None, str] = ("CELLCOUNT", "4_threshd.zarr")
    _threshd_volumes: tuple[str | None, str] = ("CELLCOUNT", "5_threshd_volumes.zarr")
    _threshd_filt: tuple[str | None, str] = ("CELLCOUNT", "6_threshd_filt.zarr")
    _maxima: tuple[str | None, str] = ("CELLCOUNT", "7_maxima.zarr")
    _wshed_volumes: tuple[str | None, str] = ("CELLCOUNT", "8_wshed_volumes.zarr")
    _wshed_filt: tuple[str | None, str] = ("CELLCOUNT", "9_wshed_filt.zarr")
    _threshd_final: tuple[str | None, str] = ("CELLCOUNT", "10_threshd_f.zarr")
    _maxima_final: tuple[str | None, str] = ("CELLCOUNT", "10_maxima_f.zarr")
    _wshed_final: tuple[str | None, str] = ("CELLCOUNT", "10_wshed_f.zarr")
    _maxima_df: tuple[str | None, str] = ("ANALYSIS", "1_maxima.parquet")
    _cells_raw_df: tuple[str | None, str] = ("ANALYSIS", "1_cells_raw.parquet")
    _cells_trfm_df: tuple[str | None, str] = ("ANALYSIS", "2_cells_trfm.parquet")
    _cells_df: tuple[str | None, str] = ("ANALYSIS", "3_cells.parquet")
    _cells_agg_df: tuple[str | None, str] = ("ANALYSIS", "4_cells_agg.parquet")
    _cells_agg_csv: tuple[str | None, str] = ("ANALYSIS", "5_cells_agg.csv")
    _points_raw: tuple[str | None, str] = ("VISUAL", "points_raw.zarr")
    _heatmap_raw: tuple[str | None, str] = ("VISUAL", "heatmap_raw.zarr")
    _points_trfm: tuple[str | None, str] = ("VISUAL", "points_trfm.tif")
    _heatmap_trfm: tuple[str | None, str] = ("VISUAL", "heatmap_trfm.tif")
    _comb_reg: tuple[str | None, str] = ("VISUAL_COMB", "comb_reg.tif")
    _comb_cellc: tuple[str | None, str] = ("VISUAL_COMB", "comb_cellc.tif")
    _comb_points: tuple[str | None, str] = ("VISUAL_COMB", "comb_points.tif")

    def __init__(self, root_dir: str, subdirs):
        self.root_dir = root_dir
        self.subdirs = subdirs
        self.assert_subdirs_exist()
        logger.debug(f'Getting ProjFpModel for "{root_dir}", with subdirs {subdirs}')

    @property
    def config_params(self) -> str:
        self._raise_not_set("config_params")

    @property
    def raw(self) -> str:
        self._raise_not_set("raw")

    @property
    def ref(self) -> str:
        self._raise_not_set("ref")

    @property
    def annot(self) -> str:
        self._raise_not_set("annot")

    @property
    def map(self) -> str:
        self._raise_not_set("map")

    @property
    def affine(self) -> str:
        self._raise_not_set("affine")

    @property
    def bspline(self) -> str:
        self._raise_not_set("bspline")

    @property
    def downsmpl1(self) -> str:
        self._raise_not_set("downsmpl1")

    @property
    def downsmpl2(self) -> str:
        self._raise_not_set("downsmpl2")

    @property
    def trimmed(self) -> str:
        self._raise_not_set("trimmed")

    @property
    def bounded(self) -> str:
        self._raise_not_set("bounded")

    @property
    def regresult(self) -> str:
        self._raise_not_set("regresult")
        return ""

    @property
    def premask_blur(self) -> str:
        self._raise_not_set("premask_blur")

    @property
    def mask(self) -> str:
        self._raise_not_set("mask")

    @property
    def outline(self) -> str:
        self._raise_not_set("outline")

    @property
    def mask_reg(self) -> str:
        self._raise_not_set("mask_reg")

    @property
    def mask_df(self) -> str:
        self._raise_not_set("mask_df")

    @property
    def overlap(self) -> str:
        self._raise_not_set("overlap")

    @property
    def bgrm(self) -> str:
        self._raise_not_set("bgrm")

    @property
    def dog(self) -> str:
        self._raise_not_set("dog")

    @property
    def adaptv(self) -> str:
        self._raise_not_set("adaptv")

    @property
    def threshd(self) -> str:
        self._raise_not_set("threshd")

    @property
    def threshd_volumes(self) -> str:
        self._raise_not_set("threshd_volumes")

    @property
    def threshd_filt(self) -> str:
        self._raise_not_set("threshd_filt")

    @property
    def maxima(self) -> str:
        self._raise_not_set("maxima")

    @property
    def wshed_volumes(self) -> str:
        self._raise_not_set("wshed_volumes")

    @property
    def wshed_filt(self) -> str:
        self._raise_not_set("wshed_filt")

    @property
    def threshd_final(self) -> str:
        self._raise_not_set("threshd_final")

    @property
    def maxima_final(self) -> str:
        self._raise_not_set("maxima_final")

    @property
    def wshed_final(self) -> str:
        self._raise_not_set("wshed_final")

    @property
    def maxima_df(self) -> str:
        self._raise_not_set("maxima_df")

    @property
    def cells_raw_df(self) -> str:
        self._raise_not_set("cells_raw_df")

    @property
    def cells_trfm_df(self) -> str:
        self._raise_not_set("cells_trfm_df")

    @property
    def cells_df(self) -> str:
        self._raise_not_set("cells_df")

    @property
    def cells_agg_df(self) -> str:
        self._raise_not_set("cells_agg_df")

    @property
    def cells_agg_csv(self) -> str:
        self._raise_not_set("cells_agg_csv")

    @property
    def points_raw(self) -> str:
        self._raise_not_set("points_raw")

    @property
    def heatmap_raw(self) -> str:
        self._raise_not_set("heatmap_raw")

    @property
    def points_trfm(self) -> str:
        self._raise_not_set("points_trfm")

    @property
    def heatmap_trfm(self) -> str:
        self._raise_not_set("heatmap_trfm")

    @property
    def comb_reg(self) -> str:
        self._raise_not_set("comb_reg")

    @property
    def comb_cellc(self) -> str:
        self._raise_not_set("comb_cellc")

    @property
    def comb_points(self) -> str:
        self._raise_not_set("comb_points")

    def assert_subdirs_exist(self):
        """
        Assert that all subdirectory values are set in the enum.
        """
        for attr in [
            "RAW",
            "REGISTRATION",
            "MASK",
            "CELLCOUNT",
            "ANALYSIS",
            "VISUAL",
            "VISUAL_COMB",
        ]:
            assert hasattr(
                self.subdirs, attr
            ), f"Subdirectory '{attr}' not set in subdirs enum attribute."
        logger.debug("Asserted that all relevant subdirs exist in subdirs attr.")

    def copy(self):
        return self.__init__(self.root_dir, self.subdirs)

    def make_subdirs(self):
        """
        Make project directories.
        """
        for folder in self.subdirs:
            os.makedirs(os.path.join(self.root_dir, folder.value), exist_ok=True)

    def _raise_not_set(self, attr):
        raise NotImplementedError(
            f"The filepath '{attr}' is not implemented in the model '{type(self)}'.\n"
            "Use a different model (recommended)"
            "or explicitly edit this model."
        )

    def _set_attribute(self, attr: str):
        """
        Refer to the underlying attribute (NOT the property. i.e. attribute has name "_<attr>")
        and build & return the file path.
        The attribute is in the format:

        ```
        _<attr> = (subdir, basename)

        subdir -> refers to attribute in self.subdir Enum
        basename -> basename string
        ```
        """
        subdir, basename = getattr(self, f"_{attr}")
        if subdir is None:
            return os.path.join(self.root_dir, basename)
        else:
            return os.path.join(self.root_dir, self.subdirs[subdir].value, basename)


class ProjSubdirs(Enum):
    RAW = "raw"
    REGISTRATION = "registration"
    MASK = "mask"
    CELLCOUNT = "cellcount"
    ANALYSIS = "analysis"
    VISUAL = "visual"
    VISUAL_COMB = "visual_comb"


class ProjFpModel(ProjFpModelBase):
    """
    Model for project file paths.
    """

    def __init__(self, root_dir: str):
        super().__init__(root_dir, ProjSubdirs)

        for attr in [
            "config_params",
            "raw",
            "ref",
            "annot",
            "map",
            "affine",
            "bspline",
            "downsmpl1",
            "downsmpl2",
            "trimmed",
            "bounded",
            "regresult",
            "premask_blur",
            "mask",
            "outline",
            "mask_reg",
            "mask_df",
            "overlap",
            "bgrm",
            "dog",
            "adaptv",
            "threshd",
            "threshd_volumes",
            "threshd_filt",
            "maxima",
            "wshed_volumes",
            "wshed_filt",
            "threshd_final",
            "maxima_final",
            "wshed_final",
            "maxima_df",
            "cells_raw_df",
            "cells_trfm_df",
            "cells_df",
            "cells_agg_df",
            "cells_agg_csv",
            "points_raw",
            "heatmap_raw",
            "points_trfm",
            "heatmap_trfm",
            "comb_reg",
            "comb_cellc",
            "comb_points",
        ]:
            setattr(
                self.__class__,
                attr,
                property(lambda self, attr=attr: self._set_attribute(attr)),
            )


class ProjSubdirsTuning(Enum):
    RAW = "raw_tuning"
    REGISTRATION = "registration"
    MASK = "mask"
    CELLCOUNT = "cellcount_tuning"
    ANALYSIS = "analysis_tuning"
    VISUAL = "visual"
    VISUAL_COMB = "visual_comb"


class ProjFpModelTuning(ProjFpModelBase):
    """
    Model for tuning-mode project file paths.
    """

    def __init__(self, root_dir: str):
        super().__init__(root_dir, ProjSubdirsTuning)

        for attr in [
            "config_params",
            "raw",
            "overlap",
            "bgrm",
            "dog",
            "adaptv",
            "threshd",
            "threshd_volumes",
            "threshd_filt",
            "maxima",
            "wshed_volumes",
            "wshed_filt",
            "threshd_final",
            "maxima_final",
            "wshed_final",
            "maxima_df",
            "cells_raw_df",
        ]:
            setattr(
                self.__class__,
                attr,
                property(lambda self, attr=attr: self._set_attribute(attr)),
            )
