import os
from enum import Enum

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


class ProjFpAttr:
    """ """

    def __init__(self, val, get_callable=None, set_callable=None):
        self.val = val
        self.get_callable = get_callable
        self.set_callable = set_callable

    def get(self):
        if self.get_callable:
            return self.get_callable()
        return self.val

    def set(self, val):
        if self.set_callable:
            self.val = self.set_callable()
        self.val = val


class ProjFpPath:
    """ """

    def __init__(self, root_dir: str, path_dirs_ls: list[str], implemented: bool = False):
        self.root_dir = root_dir
        self.path_dirs_ls = path_dirs_ls
        self.implemented = implemented

    def set_implement(self, value: bool = True):
        """
        Set whether an attribute is implemented in the model.
        Defaults to True.
        """
        self.implemented = value

    @property
    def val(self):
        """
        Used to get a given attribute's file path.
        It is called within the property getters.

        Refer to the underlying attribute (NOT the property. i.e. attribute has name "_<attr>")
        and build & return the file path.
        The attribute is in the format:

        ```
        _<attr> = (implemented, [subdirs, ..., basename])

        implemented -> whether the attribute is implemented in the model
        [subdirs, ..., basename] -> Effectively the split path that is to be joined
        ```
        """
        # Asserting that the path_dirs_ls has at least 1 element (the basename)
        assert len(self.path_dirs_ls) > 0, "Path directories list is empty."
        # Assertignt that the last item in path_dirs_ls has an extension
        assert os.path.splitext(self.path_dirs_ls[-1])[1], "Last item in path_dirs_ls must have an extension."
        # If attribute is not implemented, raise error
        if not self.implemented:
            raise NotImplementedError(
                "This filepath is not implemented.\n"
                "Activate this by calling 'set_implement'."
                "or explicitly edit this model."
            )
        # Returning filepath as "root_dir/subdir1/subdir2/.../basename"
        return os.path.join(self.root_dir, *self.path_dirs_ls)


class ProjFpModelBase:
    """
    Abstract model for project file paths.
    """

    def __init__(
        self,
        root_dir: str,
        raw: str,
        registration: str,
        mask: str,
        cellcount: str,
        analysis: str,
        visual: str,
        visual_comb: str,
    ):
        # Storing base attributes
        self.root_dir: str = ProjFpAttr(root_dir, set_callable=self.init_filepath_attributes)
        self.raw: str = ProjFpAttr(raw, set_callable=self.init_filepath_attributes)
        self.registration: str = ProjFpAttr(registration, set_callable=self.init_filepath_attributes)
        self.mask: str = ProjFpAttr(mask, set_callable=self.init_filepath_attributes)
        self.cellcount: str = ProjFpAttr(cellcount, set_callable=self.init_filepath_attributes)
        self.analysis: str = ProjFpAttr(analysis, set_callable=self.init_filepath_attributes)
        self.visual: str = ProjFpAttr(visual, set_callable=self.init_filepath_attributes)
        self.visual_comb: str = ProjFpAttr(visual_comb, set_callable=self.init_filepath_attributes)
        # Setting filepath attributes
        self.init_filepath_attributes()

    def init_filepath_attributes(self):
        # Making subdirs list
        self.subdirs_ls = [
            self.raw,
            self.registration,
            self.mask,
            self.cellcount,
            self.analysis,
            self.visual,
            self.visual_comb,
        ]
        # Filepath attributes are in the format: _<attr> = (implemented, [subdirs, ..., basename])
        self.config_params: ProjFpPath = ProjFpPath(self.root_dir, ["config_params.json"])
        self.raw: ProjFpPath = ProjFpPath(self.root_dir, [self.raw, "raw.zarr"])
        self.ref: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "0a_reference.tif"])
        self.annot: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "0b_annotation.tif"])
        self.map: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "0c_mapping.json"])
        self.affine: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "0d_align_affine.txt"])
        self.bspline: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "0e_align_bspline.txt"])
        self.downsmpl1: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "1_downsmpl1.tif"])
        self.downsmpl2: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "2_downsmpl2.tif"])
        self.trimmed: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "3_trimmed.tif"])
        self.bounded: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "4_bounded.tif"])
        self.regresult: ProjFpPath = ProjFpPath(self.root_dir, [self.registration, "5_regresult.tif"])
        self.premask_blur: ProjFpPath = ProjFpPath(self.root_dir, [self.mask, "1_premask_blur.tif"])
        self.mask: ProjFpPath = ProjFpPath(self.root_dir, [self.mask, "2_mask_trimmed.tif"])
        self.outline: ProjFpPath = ProjFpPath(self.root_dir, [self.mask, "3_outline_reg.tif"])
        self.mask_reg: ProjFpPath = ProjFpPath(self.root_dir, [self.mask, "4_mask_reg.tif"])
        self.mask_df: ProjFpPath = ProjFpPath(self.root_dir, [self.mask, "5_mask.parquet"])
        self.overlap: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "0_overlap.zarr"])
        self.bgrm: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "1_bgrm.zarr"])
        self.dog: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "2_dog.zarr"])
        self.adaptv: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "3_adaptv.zarr"])
        self.threshd: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "4_threshd.zarr"])
        self.threshd_volumes: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "5_threshd_volumes.zarr"])
        self.threshd_filt: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "6_threshd_filt.zarr"])
        self.maxima: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "7_maxima.zarr"])
        self.wshed_volumes: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "8_wshed_volumes.zarr"])
        self.wshed_filt: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "9_wshed_filt.zarr"])
        self.threshd_final: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "10_threshd_f.zarr"])
        self.maxima_final: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "10_maxima_f.zarr"])
        self.wshed_final: ProjFpPath = ProjFpPath(self.root_dir, [self.cellcount, "10_wshed_f.zarr"])
        self.maxima_df: ProjFpPath = ProjFpPath(self.root_dir, [self.analysis, "1_maxima.parquet"])
        self.cells_raw_df: ProjFpPath = ProjFpPath(self.root_dir, [self.analysis, "1_cells_raw.parquet"])
        self.cells_trfm_df: ProjFpPath = ProjFpPath(self.root_dir, [self.analysis, "2_cells_trfm.parquet"])
        self.cells_df: ProjFpPath = ProjFpPath(self.root_dir, [self.analysis, "3_cells.parquet"])
        self.cells_agg_df: ProjFpPath = ProjFpPath(self.root_dir, [self.analysis, "4_cells_agg.parquet"])
        self.cells_agg_csv: ProjFpPath = ProjFpPath(self.root_dir, [self.analysis, "5_cells_agg.csv"])
        self.points_raw: ProjFpPath = ProjFpPath(self.root_dir, [self.visual, "points_raw.zarr"])
        self.heatmap_raw: ProjFpPath = ProjFpPath(self.root_dir, [self.visual, "heatmap_raw.zarr"])
        self.points_trfm: ProjFpPath = ProjFpPath(self.root_dir, [self.visual, "points_trfm.tif"])
        self.heatmap_trfm: ProjFpPath = ProjFpPath(self.root_dir, [self.visual, "heatmap_trfm.tif"])
        self.comb_reg: ProjFpPath = ProjFpPath(self.root_dir, [self.visual_comb, "comb_reg.tif"])
        self.comb_cellc: ProjFpPath = ProjFpPath(self.root_dir, [self.visual_comb, "comb_cellc.tif"])
        self.comb_points: ProjFpPath = ProjFpPath(self.root_dir, [self.visual_comb, "comb_points.tif"])

    def copy(self):
        return self.__init__(
            root_dir=self.root_dir,
            raw=self.raw,
            registration=self.registration,
            mask=self.mask,
            cellcount=self.cellcount,
            analysis=self.analysis,
            visual=self.visual,
            visual_comb=self.visual_comb,
        )

    def make_subdirs(self):
        """
        Make project directories.
        """
        for subdir in self.subdirs_ls:
            os.makedirs(os.path.join(self.root_dir, subdir), exist_ok=True)

    def export2dict(self) -> dict:
        return {
            "config_params": self.config_params.value,
            "raw": self.raw.value,
            "ref": self.ref.value,
            "annot": self.annot.value,
            "map": self.map.value,
            "affine": self.affine.value,
            "bspline": self.bspline.value,
            "downsmpl1": self.downsmpl1.value,
            "downsmpl2": self.downsmpl2.value,
            "trimmed": self.trimmed.value,
            "bounded": self.bounded.value,
            "regresult": self.regresult.value,
            "premask_blur": self.premask_blur.value,
            "mask": self.mask.value,
            "outline": self.outline.value,
            "mask_reg": self.mask_reg.value,
            "mask_df": self.mask_df.value,
            "overlap": self.overlap.value,
            "bgrm": self.bgrm.value,
            "dog": self.dog.value,
            "adaptv": self.adaptv.value,
            "threshd": self.threshd.value,
            "threshd_volumes": self.threshd_volumes.value,
            "threshd_filt": self.threshd_filt.value,
            "maxima": self.maxima.value,
            "wshed_volumes": self.wshed_volumes.value,
            "wshed_filt": self.wshed_filt.value,
            "threshd_final": self.threshd_final.value,
            "maxima_final": self.maxima_final.value,
            "wshed_final": self.wshed_final.value,
            "maxima_df": self.maxima_df.value,
            "cells_raw_df": self.cells_raw_df.value,
            "cells_trfm_df": self.cells_trfm_df.value,
            "cells_df": self.cells_df.value,
            "cells_agg_df": self.cells_agg_df.value,
            "cells_agg_csv": self.cells_agg_csv.value,
            "points_raw": self.points_raw.value,
            "heatmap_raw": self.heatmap_raw.value,
            "points_trfm": self.points_trfm.value,
            "heatmap_trfm": self.heatmap_trfm.value,
            "comb_reg": self.comb_reg.value,
            "comb_cellc": self.comb_cellc.value,
            "comb_points": self.comb_points.value,
        }


class ProjFpModel(ProjFpModelBase):
    """
    Model for project file paths.
    """

    def __init__(self, root_dir: str):
        super().__init__(
            root_dir=root_dir,
            raw="raw",
            registration="registration",
            mask="mask",
            cellcount="cellcount",
            analysis="analysis",
            visual="visual",
            visual_comb="visual_comb",
        )
        # Setting attributes as implemented
        self.config_params.set_implement()
        self.raw.set_implement()
        self.ref.set_implement()
        self.annot.set_implement()
        self.map.set_implement()
        self.affine.set_implement()
        self.bspline.set_implement()
        self.downsmpl1.set_implement()
        self.downsmpl2.set_implement()
        self.trimmed.set_implement()
        self.bounded.set_implement()
        self.regresult.set_implement()
        self.premask_blur.set_implement()
        self.mask.set_implement()
        self.outline.set_implement()
        self.mask_reg.set_implement()
        self.mask_df.set_implement()
        self.overlap.set_implement()
        self.bgrm.set_implement()
        self.dog.set_implement()
        self.adaptv.set_implement()
        self.threshd.set_implement()
        self.threshd_volumes.set_implement()
        self.threshd_filt.set_implement()
        self.maxima.set_implement()
        self.wshed_volumes.set_implement()
        self.wshed_filt.set_implement()
        self.threshd_final.set_implement()
        self.maxima_final.set_implement()
        self.wshed_final.set_implement()
        self.maxima_df.set_implement()
        self.cells_raw_df.set_implement()
        self.cells_trfm_df.set_implement()
        self.cells_df.set_implement()
        self.cells_agg_df.set_implement()
        self.cells_agg_csv.set_implement()
        self.points_raw.set_implement()
        self.heatmap_raw.set_implement()
        self.points_trfm.set_implement()
        self.heatmap_trfm.set_implement()
        self.comb_reg.set_implement()
        self.comb_cellc.set_implement()
        self.comb_points.set_implement()


class ProjFpModelTuning(ProjFpModelBase):
    """
    Model for tuning-mode project file paths.
    """

    def __init__(self, root_dir: str):
        super().__init__(
            root_dir=root_dir,
            raw="raw_tuning",
            registration="registration",
            mask="mask",
            cellcount="cellcount_tuning",
            analysis="analysis_tuning",
            visual="visual",
            visual_comb="visual_comb",
        )
        # Setting attributes as implemented
        self.config_params.set_implement()
        self.raw.set_implement()
        self.overlap.set_implement()
        self.bgrm.set_implement()
        self.dog.set_implement()
        self.adaptv.set_implement()
        self.threshd.set_implement()
        self.threshd_volumes.set_implement()
        self.threshd_filt.set_implement()
        self.maxima.set_implement()
        self.wshed_volumes.set_implement()
        self.wshed_filt.set_implement()
        self.threshd_final.set_implement()
        self.maxima_final.set_implement()
        self.wshed_final.set_implement()
        self.maxima_df.set_implement()
        self.cells_raw_df.set_implement()
