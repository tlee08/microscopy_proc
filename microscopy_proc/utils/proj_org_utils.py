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
            self.get_callable()
        return self.val

    def set(self, val):
        if self.set_callable:
            self.set_callable()
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
        self.root_dir = ProjFpAttr(root_dir, set_callable=self.init_filepath_attributes)
        self.raw = ProjFpAttr(raw, set_callable=self.init_filepath_attributes)
        self.registration = ProjFpAttr(registration, set_callable=self.init_filepath_attributes)
        self.mask = ProjFpAttr(mask, set_callable=self.init_filepath_attributes)
        self.cellcount = ProjFpAttr(cellcount, set_callable=self.init_filepath_attributes)
        self.analysis = ProjFpAttr(analysis, set_callable=self.init_filepath_attributes)
        self.visual = ProjFpAttr(visual, set_callable=self.init_filepath_attributes)
        self.visual_comb = ProjFpAttr(visual_comb, set_callable=self.init_filepath_attributes)
        # Setting filepath attributes
        self.init_filepath_attributes()

    def init_filepath_attributes(self):
        # Making subdirs list
        self.subdirs_ls: list[ProjFpAttr] = [
            self.raw,
            self.registration,
            self.mask,
            self.cellcount,
            self.analysis,
            self.visual,
            self.visual_comb,
        ]
        # Filepath attributes are in the format: _<attr> = (implemented, [subdirs, ..., basename])
        self.config_params = ProjFpPath(self.root_dir.get(), ["config_params.json"])
        self.raw = ProjFpPath(self.root_dir.get(), [self.raw.get(), "raw.zarr"])
        self.ref = ProjFpPath(self.root_dir.get(), [self.registration.get(), "0a_reference.tif"])
        self.annot = ProjFpPath(self.root_dir.get(), [self.registration.get(), "0b_annotation.tif"])
        self.map = ProjFpPath(self.root_dir.get(), [self.registration.get(), "0c_mapping.json"])
        self.affine = ProjFpPath(self.root_dir.get(), [self.registration.get(), "0d_align_affine.txt"])
        self.bspline = ProjFpPath(self.root_dir.get(), [self.registration.get(), "0e_align_bspline.txt"])
        self.downsmpl1 = ProjFpPath(self.root_dir.get(), [self.registration.get(), "1_downsmpl1.tif"])
        self.downsmpl2 = ProjFpPath(self.root_dir.get(), [self.registration.get(), "2_downsmpl2.tif"])
        self.trimmed = ProjFpPath(self.root_dir.get(), [self.registration.get(), "3_trimmed.tif"])
        self.bounded = ProjFpPath(self.root_dir.get(), [self.registration.get(), "4_bounded.tif"])
        self.regresult = ProjFpPath(self.root_dir.get(), [self.registration.get(), "5_regresult.tif"])
        self.premask_blur = ProjFpPath(self.root_dir.get(), [self.mask.get(), "1_premask_blur.tif"])
        self.mask = ProjFpPath(self.root_dir.get(), [self.mask.get(), "2_mask_trimmed.tif"])
        self.outline = ProjFpPath(self.root_dir.get(), [self.mask.get(), "3_outline_reg.tif"])
        self.mask_reg = ProjFpPath(self.root_dir.get(), [self.mask.get(), "4_mask_reg.tif"])
        self.mask_df = ProjFpPath(self.root_dir.get(), [self.mask.get(), "5_mask.parquet"])
        self.overlap = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "0_overlap.zarr"])
        self.bgrm = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "1_bgrm.zarr"])
        self.dog = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "2_dog.zarr"])
        self.adaptv = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "3_adaptv.zarr"])
        self.threshd = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "4_threshd.zarr"])
        self.threshd_volumes = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "5_threshd_volumes.zarr"])
        self.threshd_filt = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "6_threshd_filt.zarr"])
        self.maxima = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "7_maxima.zarr"])
        self.wshed_volumes = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "8_wshed_volumes.zarr"])
        self.wshed_filt = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "9_wshed_filt.zarr"])
        self.threshd_final = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "10_threshd_f.zarr"])
        self.maxima_final = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "10_maxima_f.zarr"])
        self.wshed_final = ProjFpPath(self.root_dir.get(), [self.cellcount.get(), "10_wshed_f.zarr"])
        self.maxima_df = ProjFpPath(self.root_dir.get(), [self.analysis.get(), "1_maxima.parquet"])
        self.cells_raw_df = ProjFpPath(self.root_dir.get(), [self.analysis.get(), "1_cells_raw.parquet"])
        self.cells_trfm_df = ProjFpPath(self.root_dir.get(), [self.analysis.get(), "2_cells_trfm.parquet"])
        self.cells_df = ProjFpPath(self.root_dir.get(), [self.analysis.get(), "3_cells.parquet"])
        self.cells_agg_df = ProjFpPath(self.root_dir.get(), [self.analysis.get(), "4_cells_agg.parquet"])
        self.cells_agg_csv = ProjFpPath(self.root_dir.get(), [self.analysis.get(), "5_cells_agg.csv"])
        self.points_raw = ProjFpPath(self.root_dir.get(), [self.visual.get(), "points_raw.zarr"])
        self.heatmap_raw = ProjFpPath(self.root_dir.get(), [self.visual.get(), "heatmap_raw.zarr"])
        self.points_trfm = ProjFpPath(self.root_dir.get(), [self.visual.get(), "points_trfm.tif"])
        self.heatmap_trfm = ProjFpPath(self.root_dir.get(), [self.visual.get(), "heatmap_trfm.tif"])
        self.comb_reg = ProjFpPath(self.root_dir.get(), [self.visual_comb.get(), "comb_reg.tif"])
        self.comb_cellc = ProjFpPath(self.root_dir.get(), [self.visual_comb.get(), "comb_cellc.tif"])
        self.comb_points = ProjFpPath(self.root_dir.get(), [self.visual_comb.get(), "comb_points.tif"])

    def copy(self):
        return self.__init__(
            root_dir=self.root_dir.get(),
            raw=self.raw.get(),
            registration=self.registration.get(),
            mask=self.mask.get(),
            cellcount=self.cellcount.get(),
            analysis=self.analysis.get(),
            visual=self.visual.get(),
            visual_comb=self.visual_comb.get(),
        )

    def make_subdirs(self):
        """
        Make project directories.
        """
        for subdir in self.subdirs_ls:
            os.makedirs(os.path.join(self.root_dir.get(), subdir.get()), exist_ok=True)

    def export2dict(self) -> dict:
        return {
            "config_params": self.config_params.val,
            "raw": self.raw.val,
            "ref": self.ref.val,
            "annot": self.annot.val,
            "map": self.map.val,
            "affine": self.affine.val,
            "bspline": self.bspline.val,
            "downsmpl1": self.downsmpl1.val,
            "downsmpl2": self.downsmpl2.val,
            "trimmed": self.trimmed.val,
            "bounded": self.bounded.val,
            "regresult": self.regresult.val,
            "premask_blur": self.premask_blur.val,
            "mask": self.mask.val,
            "outline": self.outline.val,
            "mask_reg": self.mask_reg.val,
            "mask_df": self.mask_df.val,
            "overlap": self.overlap.val,
            "bgrm": self.bgrm.val,
            "dog": self.dog.val,
            "adaptv": self.adaptv.val,
            "threshd": self.threshd.val,
            "threshd_volumes": self.threshd_volumes.val,
            "threshd_filt": self.threshd_filt.val,
            "maxima": self.maxima.val,
            "wshed_volumes": self.wshed_volumes.val,
            "wshed_filt": self.wshed_filt.val,
            "threshd_final": self.threshd_final.val,
            "maxima_final": self.maxima_final.val,
            "wshed_final": self.wshed_final.val,
            "maxima_df": self.maxima_df.val,
            "cells_raw_df": self.cells_raw_df.val,
            "cells_trfm_df": self.cells_trfm_df.val,
            "cells_df": self.cells_df.val,
            "cells_agg_df": self.cells_agg_df.val,
            "cells_agg_csv": self.cells_agg_csv.val,
            "points_raw": self.points_raw.val,
            "heatmap_raw": self.heatmap_raw.val,
            "points_trfm": self.points_trfm.val,
            "heatmap_trfm": self.heatmap_trfm.val,
            "comb_reg": self.comb_reg.val,
            "comb_cellc": self.comb_cellc.val,
            "comb_points": self.comb_points.val,
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
