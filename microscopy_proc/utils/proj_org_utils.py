import inspect
import os
from abc import ABC, abstractmethod


class ObservedAttr:
    """ """

    def __init__(self, val, get_callable=None, set_callable=None):
        self._val = val
        self.get_callable = get_callable
        self.set_callable = set_callable

    @property
    def val(self):
        # First run callable
        if self.get_callable:
            self.get_callable()
        # Then return val
        return self._val

    @val.setter
    def val(self, val):
        # First update val
        self._val = val
        # Then run callable
        if self.set_callable:
            self.set_callable()


class FpAttr:
    """ """

    def __init__(self, path_dirs_ls: list[str], implemented: bool = False):
        self.path_dirs_ls = path_dirs_ls
        self.implemented = implemented

    def set_implement(self):
        self.implemented = True

    def unset_implement(self):
        self.implemented = False

    @property
    def val(self):
        """
        Return the joined path_dirs_ls as a filepath.

        If path_dirs_ls is empty, raise an AssertionError.
        If the last item in path_dirs_ls does not have an extension, raise an AssertionError.
        If the attribute is not implemented, raise a NotImplementedError.
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
                "or explicitly edit this model.\n"
                f"When implemented, the path will be: {os.path.join(*self.path_dirs_ls)}"
            )
        # Returning joined filepath
        return os.path.join(*self.path_dirs_ls)


class FpModel(ABC):
    """
    Abstract model for file paths.
    """

    root_dir: ObservedAttr
    subdirs_ls: list[ObservedAttr]

    @abstractmethod
    def set_filepaths(self):
        pass

    def copy(self):
        # Getting the class constructor parameters
        params_ls = list(dict(inspect.signature(self.__init__).parameters).keys())
        # Constructing an identical model with the corresponding parameter attributes
        return self.__init__(**{param: getattr(self, param).val for param in params_ls})

    def make_subdirs(self):
        """
        Make project directories from all subdirs in
        """
        for subdir in self.subdirs_ls:
            os.makedirs(os.path.join(self.root_dir.val, subdir.val), exist_ok=True)

    def export2dict(self) -> dict:
        """
        Returns a dict of all the FpModel attributes
        """
        export_dict = {}
        # For each attribute in the model
        for attr in dir(self):
            # Skipping private attributes
            if attr.startswith("_"):
                continue
            # If the attribute is a FpPath, add it to the export dict
            if isinstance(getattr(self, attr), FpAttr):
                export_dict[attr] = getattr(self, attr).val
        # Returning
        return export_dict


class RefFpModel(FpModel):
    """
    Model for reference file paths.
    """

    def __init__(self, root_dir: str, ref_version: str, annot_version: str, map_version: str):
        """
        atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"

        Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
        """
        # Storing base attributes
        # and wrapping in a FpAttr to reset filepath attributes whenever they are changed
        self.root_dir = ObservedAttr(root_dir, set_callable=self.set_filepaths)
        self.ref_version = ObservedAttr(ref_version, set_callable=self.set_filepaths)
        self.annot_version = ObservedAttr(annot_version, set_callable=self.set_filepaths)
        self.map_version = ObservedAttr(map_version, set_callable=self.set_filepaths)
        # Constant subdirectory names
        self.reference_sdir = ObservedAttr("reference", set_callable=self.set_filepaths)
        self.annotation_sdor = ObservedAttr("annotation", set_callable=self.set_filepaths)
        self.mapping_sdir = ObservedAttr("region_mapping", set_callable=self.set_filepaths)
        self.elastix_sdir = ObservedAttr("elastix_params", set_callable=self.set_filepaths)
        # Setting filepath attributes
        self.subdirs_ls = []
        self.set_filepaths()

    def set_filepaths(self):
        # Making subdirs list
        self.subdirs_ls = [
            self.reference_sdir,
            self.annotation_sdor,
            self.mapping_sdir,
            self.elastix_sdir,
        ]
        # Setting filepath attributes
        self.ref = FpAttr(
            [self.root_dir.val, self.reference_sdir.val, f"{self.ref_version.val}.tif"],
            True,
        )
        self.annot = FpAttr(
            [
                self.root_dir.val,
                self.annotation_sdor.val,
                f"{self.annot_version.val}.tif",
            ],
            True,
        )
        self.map = FpAttr(
            [self.root_dir.val, self.mapping_sdir.val, f"{self.map_version.val}.json"],
            True,
        )
        self.affine = FpAttr([self.root_dir.val, self.elastix_sdir.val, "align_affine.txt"], True)
        self.bspline = FpAttr([self.root_dir.val, self.elastix_sdir.val, "align_bspline.txt"], True)


class ProjFpModelBase(FpModel):
    """
    Abstract model for project file paths.
    """

    def __init__(
        self,
        root_dir: str,
        raw_sdir: str,
        registration_sdir: str,
        mask_sdir: str,
        cellcount_sdir: str,
        analysis_sdir: str,
        visual_sdir: str,
    ):
        # Storing base attributes
        # and wrapping in a FpAttr to reset filepath attributes whenever they are changed
        self.root_dir = ObservedAttr(root_dir, set_callable=self.set_filepaths)
        self.raw_sdir = ObservedAttr(raw_sdir, set_callable=self.set_filepaths)
        self.registration_sdir = ObservedAttr(registration_sdir, set_callable=self.set_filepaths)
        self.mask_sdir = ObservedAttr(mask_sdir, set_callable=self.set_filepaths)
        self.cellcount_sdir = ObservedAttr(cellcount_sdir, set_callable=self.set_filepaths)
        self.analysis_sdir = ObservedAttr(analysis_sdir, set_callable=self.set_filepaths)
        self.visual_sdir = ObservedAttr(visual_sdir, set_callable=self.set_filepaths)
        # Setting filepath attributes
        self.subdirs_ls = []
        self.set_filepaths()

    def set_filepaths(self):
        # Making subdirs list
        self.subdirs_ls = [
            self.raw_sdir,
            self.registration_sdir,
            self.mask_sdir,
            self.cellcount_sdir,
            self.analysis_sdir,
            self.visual_sdir,
        ]
        # Setting filepath attributes
        self.config_params = FpAttr([self.root_dir.val, "config_params.json"])
        self.raw = FpAttr([self.root_dir.val, self.raw_sdir.val, "raw.zarr"])
        self.ref = FpAttr([self.root_dir.val, self.registration_sdir.val, "0a_reference.tif"])
        self.annot = FpAttr([self.root_dir.val, self.registration_sdir.val, "0b_annotation.tif"])
        self.map = FpAttr([self.root_dir.val, self.registration_sdir.val, "0c_mapping.json"])
        self.affine = FpAttr([self.root_dir.val, self.registration_sdir.val, "0d_align_affine.txt"])
        self.bspline = FpAttr([self.root_dir.val, self.registration_sdir.val, "0e_align_bspline.txt"])
        self.downsmpl1 = FpAttr([self.root_dir.val, self.registration_sdir.val, "1_downsmpl1.tif"])
        self.downsmpl2 = FpAttr([self.root_dir.val, self.registration_sdir.val, "2_downsmpl2.tif"])
        self.trimmed = FpAttr([self.root_dir.val, self.registration_sdir.val, "3_trimmed.tif"])
        self.bounded = FpAttr([self.root_dir.val, self.registration_sdir.val, "4_bounded.tif"])
        self.regresult = FpAttr([self.root_dir.val, self.registration_sdir.val, "5_regresult.tif"])
        self.premask_blur = FpAttr([self.root_dir.val, self.mask_sdir.val, "1_premask_blur.tif"])
        self.mask_fill = FpAttr([self.root_dir.val, self.mask_sdir.val, "2_mask_trimmed.tif"])
        self.mask_outline = FpAttr([self.root_dir.val, self.mask_sdir.val, "3_outline_reg.tif"])
        self.mask_reg = FpAttr([self.root_dir.val, self.mask_sdir.val, "4_mask_reg.tif"])
        self.mask_df = FpAttr([self.root_dir.val, self.mask_sdir.val, "5_mask.parquet"])
        self.overlap = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "0_overlap.zarr"])
        self.bgrm = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "1_bgrm.zarr"])
        self.dog = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "2_dog.zarr"])
        self.adaptv = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "3_adaptv.zarr"])
        self.threshd = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "4_threshd.zarr"])
        self.threshd_volumes = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "5_threshd_volumes.zarr"])
        self.threshd_filt = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "6_threshd_filt.zarr"])
        self.maxima = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "7_maxima.zarr"])
        self.wshed_volumes = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "8_wshed_volumes.zarr"])
        self.wshed_filt = FpAttr([self.root_dir.val, self.cellcount_sdir.val, "9_wshed_filt.zarr"])
        self.maxima_df = FpAttr([self.root_dir.val, self.analysis_sdir.val, "1_maxima.parquet"])
        self.cells_raw_df = FpAttr([self.root_dir.val, self.analysis_sdir.val, "1_cells_raw.parquet"])
        self.cells_trfm_df = FpAttr([self.root_dir.val, self.analysis_sdir.val, "2_cells_trfm.parquet"])
        self.cells_df = FpAttr([self.root_dir.val, self.analysis_sdir.val, "3_cells.parquet"])
        self.cells_agg_df = FpAttr([self.root_dir.val, self.analysis_sdir.val, "4_cells_agg.parquet"])
        self.cells_agg_csv = FpAttr([self.root_dir.val, self.analysis_sdir.val, "5_cells_agg.csv"])
        self.threshd_final = FpAttr([self.root_dir.val, self.visual_sdir.val, "threshd.zarr"])
        self.maxima_final = FpAttr([self.root_dir.val, self.visual_sdir.val, "maxima.zarr"])
        self.wshed_final = FpAttr([self.root_dir.val, self.visual_sdir.val, "wshed.zarr"])
        self.points_raw = FpAttr([self.root_dir.val, self.visual_sdir.val, "points_raw.zarr"])
        self.heatmap_raw = FpAttr([self.root_dir.val, self.visual_sdir.val, "heatmap_raw.zarr"])
        self.points_trfm = FpAttr([self.root_dir.val, self.visual_sdir.val, "points_trfm.tif"])
        self.heatmap_trfm = FpAttr([self.root_dir.val, self.visual_sdir.val, "heatmap_trfm.tif"])
        self.comb_reg = FpAttr([self.root_dir.val, self.visual_sdir.val, "comb_reg.tif"])
        self.comb_cellc = FpAttr([self.root_dir.val, self.visual_sdir.val, "comb_cellc.tif"])
        self.comb_points = FpAttr([self.root_dir.val, self.visual_sdir.val, "comb_points.tif"])


class ProjFpModel(ProjFpModelBase):
    """
    Model for project file paths.
    """

    def __init__(self, root_dir: str):
        super().__init__(
            root_dir=root_dir,
            raw_sdir="raw",
            registration_sdir="registration",
            mask_sdir="mask",
            cellcount_sdir="cellcount",
            analysis_sdir="analysis",
            visual_sdir="visual",
        )

    def set_filepaths(self):
        super().set_filepaths()
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
        self.mask_fill.set_implement()
        self.mask_outline.set_implement()
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
            raw_sdir="raw_tuning",
            registration_sdir="registration",
            mask_sdir="mask",
            cellcount_sdir="cellcount_tuning",
            analysis_sdir="analysis_tuning",
            visual_sdir="visual",
        )

    def set_filepaths(self):
        super().set_filepaths()
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
        self.maxima_df.set_implement()
        self.cells_raw_df.set_implement()
