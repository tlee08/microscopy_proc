import os
from enum import Enum
from typing import Type

from microscopy_proc.utils.logging_utils import init_logger
from microscopy_proc.utils.misc_utils import enum2list, get_current_function_name

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

    def __init__(self, root_dir: str, subdirs):
        # Storing base attributes
        self.root_dir: str = root_dir
        self.subdirs: Type[Enum] = subdirs
        # Checking that all subdirs are set in the enum
        self.assert_subdirs_exist()
        # Shortcut variables of subdirs
        raw = self.subdirs.RAW
        registration = self.subdirs.REGISTRATION
        mask = self.subdirs.MASK
        cellcount = self.subdirs.CELLCOUNT
        analysis = self.subdirs.ANALYSIS
        visual = self.subdirs.VISUAL
        visual_comb = self.subdirs.VISUAL_COMB
        # Filepath attributes are in the format: _<attr> = (implemented, [subdirs, ..., basename])
        self._config_params: tuple[bool, str | None, str] = (False, ["config_params.json"])
        self._raw: tuple[bool, str | None, str] = (False, [raw, "raw.zarr"])
        self._ref: tuple[bool, str | None, str] = (False, [registration, "0a_reference.tif"])
        self._annot: tuple[bool, str | None, str] = (False, [registration, "0b_annotation.tif"])
        self._map: tuple[bool, str | None, str] = (False, [registration, "0c_mapping.json"])
        self._affine: tuple[bool, str | None, str] = (False, [registration, "0d_align_affine.txt"])
        self._bspline: tuple[bool, str | None, str] = (False, [registration, "0e_align_bspline.txt"])
        self._downsmpl1: tuple[bool, str | None, str] = (False, [registration, "1_downsmpl1.tif"])
        self._downsmpl2: tuple[bool, str | None, str] = (False, [registration, "2_downsmpl2.tif"])
        self._trimmed: tuple[bool, str | None, str] = (False, [registration, "3_trimmed.tif"])
        self._bounded: tuple[bool, str | None, str] = (False, [registration, "4_bounded.tif"])
        self._regresult: tuple[bool, str | None, str] = (False, [registration, "5_regresult.tif"])
        self._premask_blur: tuple[bool, str | None, str] = (False, [mask, "1_premask_blur.tif"])
        self._mask: tuple[bool, str | None, str] = (False, [mask, "2_mask_trimmed.tif"])
        self._outline: tuple[bool, str | None, str] = (False, [mask, "3_outline_reg.tif"])
        self._mask_reg: tuple[bool, str | None, str] = (False, [mask, "4_mask_reg.tif"])
        self._mask_df: tuple[bool, str | None, str] = (False, [mask, "5_mask.parquet"])
        self._overlap: tuple[bool, str | None, str] = (False, [cellcount, "0_overlap.zarr"])
        self._bgrm: tuple[bool, str | None, str] = (False, [cellcount, "1_bgrm.zarr"])
        self._dog: tuple[bool, str | None, str] = (False, [cellcount, "2_dog.zarr"])
        self._adaptv: tuple[bool, str | None, str] = (False, [cellcount, "3_adaptv.zarr"])
        self._threshd: tuple[bool, str | None, str] = (False, [cellcount, "4_threshd.zarr"])
        self._threshd_volumes: tuple[bool, str | None, str] = (False, [cellcount, "5_threshd_volumes.zarr"])
        self._threshd_filt: tuple[bool, str | None, str] = (False, [cellcount, "6_threshd_filt.zarr"])
        self._maxima: tuple[bool, str | None, str] = (False, [cellcount, "7_maxima.zarr"])
        self._wshed_volumes: tuple[bool, str | None, str] = (False, [cellcount, "8_wshed_volumes.zarr"])
        self._wshed_filt: tuple[bool, str | None, str] = (False, [cellcount, "9_wshed_filt.zarr"])
        self._threshd_final: tuple[bool, str | None, str] = (False, [cellcount, "10_threshd_f.zarr"])
        self._maxima_final: tuple[bool, str | None, str] = (False, [cellcount, "10_maxima_f.zarr"])
        self._wshed_final: tuple[bool, str | None, str] = (False, [cellcount, "10_wshed_f.zarr"])
        self._maxima_df: tuple[bool, str | None, str] = (False, [analysis, "1_maxima.parquet"])
        self._cells_raw_df: tuple[bool, str | None, str] = (False, [analysis, "1_cells_raw.parquet"])
        self._cells_trfm_df: tuple[bool, str | None, str] = (False, [analysis, "2_cells_trfm.parquet"])
        self._cells_df: tuple[bool, str | None, str] = (False, [analysis, "3_cells.parquet"])
        self._cells_agg_df: tuple[bool, str | None, str] = (False, [analysis, "4_cells_agg.parquet"])
        self._cells_agg_csv: tuple[bool, str | None, str] = (False, [analysis, "5_cells_agg.csv"])
        self._points_raw: tuple[bool, str | None, str] = (False, [visual, "points_raw.zarr"])
        self._heatmap_raw: tuple[bool, str | None, str] = (False, [visual, "heatmap_raw.zarr"])
        self._points_trfm: tuple[bool, str | None, str] = (False, [visual, "points_trfm.tif"])
        self._heatmap_trfm: tuple[bool, str | None, str] = (False, [visual, "heatmap_trfm.tif"])
        self._comb_reg: tuple[bool, str | None, str] = (False, [visual_comb, "comb_reg.tif"])
        self._comb_cellc: tuple[bool, str | None, str] = (False, [visual_comb, "comb_cellc.tif"])
        self._comb_points: tuple[bool, str | None, str] = (False, [visual_comb, "comb_points.tif"])
        # Setting properties corresponding to each attribute
        self.config_params = property(lambda: self._get_attribute("config_params"))
        self.raw = property(lambda: self._get_attribute("raw"))
        self.ref = property(lambda: self._get_attribute("ref"))
        self.annot = property(lambda: self._get_attribute("annot"))
        self.map = property(lambda: self._get_attribute("map"))
        self.affine = property(lambda: self._get_attribute("affine"))
        self.bspline = property(lambda: self._get_attribute("bspline"))
        self.downsmpl1 = property(lambda: self._get_attribute("downsmpl1"))
        self.downsmpl2 = property(lambda: self._get_attribute("downsmpl2"))
        self.trimmed = property(lambda: self._get_attribute("trimmed"))
        self.bounded = property(lambda: self._get_attribute("bounded"))
        self.regresult = property(lambda: self._get_attribute("regresult"))
        self.premask_blur = property(lambda: self._get_attribute("premask_blur"))
        self.mask = property(lambda: self._get_attribute("mask"))
        self.outline = property(lambda: self._get_attribute("outline"))
        self.mask_reg = property(lambda: self._get_attribute("mask_reg"))
        self.mask_df = property(lambda: self._get_attribute("mask_df"))
        self.overlap = property(lambda: self._get_attribute("overlap"))
        self.bgrm = property(lambda: self._get_attribute("bgrm"))
        self.dog = property(lambda: self._get_attribute("dog"))
        self.adaptv = property(lambda: self._get_attribute("adaptv"))
        self.threshd = property(lambda: self._get_attribute("threshd"))
        self.threshd_volumes = property(lambda: self._get_attribute("threshd_volumes"))
        self.threshd_filt = property(lambda: self._get_attribute("threshd_filt"))
        self.maxima = property(lambda: self._get_attribute("maxima"))
        self.wshed_volumes = property(lambda: self._get_attribute("wshed_volumes"))
        self.wshed_filt = property(lambda: self._get_attribute("wshed_filt"))
        self.threshd_final = property(lambda: self._get_attribute("threshd_final"))
        self.maxima_final = property(lambda: self._get_attribute("maxima_final"))
        self.wshed_final = property(lambda: self._get_attribute("wshed_final"))
        self.maxima_df = property(lambda: self._get_attribute("maxima_df"))
        self.cells_raw_df = property(lambda: self._get_attribute("cells_raw_df"))
        self.cells_trfm_df = property(lambda: self._get_attribute("cells_trfm_df"))
        self.cells_df = property(lambda: self._get_attribute("cells_df"))
        self.cells_agg_df = property(lambda: self._get_attribute("cells_agg_df"))
        self.cells_agg_csv = property(lambda: self._get_attribute("cells_agg_csv"))
        self.points_raw = property(lambda: self._get_attribute("points_raw"))
        self.heatmap_raw = property(lambda: self._get_attribute("heatmap_raw"))
        self.points_trfm = property(lambda: self._get_attribute("points_trfm"))
        self.heatmap_trfm = property(lambda: self._get_attribute("heatmap_trfm"))
        self.comb_reg = property(lambda: self._get_attribute("comb_reg"))
        self.comb_cellc = property(lambda: self._get_attribute("comb_cellc"))
        self.comb_points = property(lambda: self._get_attribute("comb_points"))
        # Logging
        logger.debug(f'Getting ProjFpModel for "{root_dir}", with subdirs, {enum2list(subdirs)}')

    @property
    def config_params(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def raw(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def ref(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def annot(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def map(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def affine(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def bspline(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def downsmpl1(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def downsmpl2(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def trimmed(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def bounded(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def regresult(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def premask_blur(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def mask(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def outline(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def mask_reg(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def mask_df(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def overlap(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def bgrm(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def dog(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def adaptv(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def threshd(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def threshd_volumes(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def threshd_filt(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def maxima(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def wshed_volumes(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def wshed_filt(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def threshd_final(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def maxima_final(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def wshed_final(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def maxima_df(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def cells_raw_df(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def cells_trfm_df(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def cells_df(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def cells_agg_df(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def cells_agg_csv(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def points_raw(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def heatmap_raw(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def points_trfm(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def heatmap_trfm(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def comb_reg(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def comb_cellc(self) -> str:
        self._get_attribute(get_current_function_name())

    @property
    def comb_points(self) -> str:
        self._get_attribute(get_current_function_name())

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
            assert hasattr(self.subdirs, attr), f"Subdirectory '{attr}' not set in subdirs enum attribute."
        logger.debug("Asserted that all relevant subdirs exist in subdirs attr.")

    def copy(self):
        return self.__init__(self.root_dir, self.subdirs)

    def make_subdirs(self):
        """
        Make project directories.
        """
        for folder in self.subdirs:
            os.makedirs(os.path.join(self.root_dir, folder.value), exist_ok=True)

    def _get_attribute(self, attr: str) -> str:
        """
        This method is used to get a given attribute's file path.
        It is called within the property getters.

        Refer to the underlying attribute (NOT the property. i.e. attribute has name "_<attr>")
        and build & return the file path.
        The attribute is in the format:

        ```
        _<attr> = (implemented, subdir, basename)

        implemented -> whether the attribute is implemented in the model
        subdir -> refers to attribute in self.subdir Enum
        basename -> basename string
        ```
        """
        # Get attribute values
        implemented, subdir, basename = getattr(self, f"_{attr}")
        # If attribute is not implemented, raise error
        if not implemented:
            raise NotImplementedError(
                f"The filepath '{attr}' is not implemented in the model '{type(self)}'.\n"
                "Use a different model (recommended)"
                "or explicitly edit this model."
            )
        # If subdir is None, return filepath is "root_dir/basenane"
        if subdir is None:
            return os.path.join(self.root_dir, basename)
        # Otherwise, return filepath is "root_dir/subdirname/basename"
        else:
            subdirname = self.subdirs[subdir].value
            return os.path.join(self.root_dir, subdirname, basename)


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

        # Setting attributes as implemented
        self._config_params[0] = True
        self._raw[0] = True
        self._ref[0] = True
        self._annot[0] = True
        self._map[0] = True
        self._affine[0] = True
        self._bspline[0] = True
        self._downsmpl1[0] = True
        self._downsmpl2[0] = True
        self._trimmed[0] = True
        self._bounded[0] = True
        self._regresult[0] = True
        self._premask_blur[0] = True
        self._mask[0] = True
        self._outline[0] = True
        self._mask_reg[0] = True
        self._mask_df[0] = True
        self._overlap[0] = True
        self._bgrm[0] = True
        self._dog[0] = True
        self._adaptv[0] = True
        self._threshd[0] = True
        self._threshd_volumes[0] = True
        self._threshd_filt[0] = True
        self._maxima[0] = True
        self._wshed_volumes[0] = True
        self._wshed_filt[0] = True
        self._threshd_final[0] = True
        self._maxima_final[0] = True
        self._wshed_final[0] = True
        self._maxima_df[0] = True
        self._cells_raw_df[0] = True
        self._cells_trfm_df[0] = True
        self._cells_df[0] = True
        self._cells_agg_df[0] = True
        self._cells_agg_csv[0] = True
        self._points_raw[0] = True
        self._heatmap_raw[0] = True
        self._points_trfm[0] = True
        self._heatmap_trfm[0] = True
        self._comb_reg[0] = True
        self._comb_cellc[0] = True
        self._comb_points[0] = True


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

        # Setting attributes as implemented
        self._config_params[0] = True
        self._raw[0] = True
        self._overlap[0] = True
        self._bgrm[0] = True
        self._dog[0] = True
        self._adaptv[0] = True
        self._threshd[0] = True
        self._threshd_volumes[0] = True
        self._threshd_filt[0] = True
        self._maxima[0] = True
        self._wshed_volumes[0] = True
        self._wshed_filt[0] = True
        self._threshd_final[0] = True
        self._maxima_final[0] = True
        self._wshed_final[0] = True
        self._maxima_df[0] = True
        self._cells_raw_df[0] = True
