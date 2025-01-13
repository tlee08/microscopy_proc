import os
import re
import shutil

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster
from natsort import natsorted
from scipy import ndimage

from microscopy_proc import ELASTIX_ENABLED, GPU_ENABLED
from microscopy_proc.constants import (
    ANNOT_COLUMNS_FINAL,
    CELL_AGG_MAPPINGS,
    TRFM,
    AnnotColumns,
    CellColumns,
    Coords,
    MaskColumns,
)
from microscopy_proc.funcs.cpu_cellc_funcs import CpuCellcFuncs as Cf
from microscopy_proc.funcs.map_funcs import MapFuncs
from microscopy_proc.funcs.mask_funcs import MaskFuncs
from microscopy_proc.funcs.reg_funcs import RegFuncs
from microscopy_proc.funcs.tiff2zarr_funcs import Tiff2ZarrFuncs
from microscopy_proc.funcs.viewer_funcs import ViewerFuncs
from microscopy_proc.funcs.visual_check_funcs_dask import VisualCheckFuncsDask
from microscopy_proc.funcs.visual_check_funcs_tiff import VisualCheckFuncsTiff
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import (
    block2coords,
    cluster_proc_contxt,
    da_overlap,
    da_trim,
    disk_cache,
)
from microscopy_proc.utils.io_utils import read_json, sanitise_smb_df, write_json
from microscopy_proc.utils.logging_utils import init_logger, log_func_decorator
from microscopy_proc.utils.misc_utils import enum2list, import_extra_error_func
from microscopy_proc.utils.proj_org_utils import (
    ProjFpModel,
    ProjFpModelBase,
    ProjFpModelTuning,
    RefFpModel,
)

# Optional dependency: gpu
if GPU_ENABLED:
    from dask_cuda import LocalCUDACluster

    from microscopy_proc.funcs.gpu_cellc_funcs import GpuCellcFuncs as Gf
else:
    LocalCUDACluster = LocalCluster
    Gf = Cf
    logger = init_logger(__name__)
    logger.info("Warning GPU functionality not installed.")
    logger.info("Using CPU functionality instead (much slower).")
    logger.info('Can install with `pip install "microscopy_proc[gpu]"`')
# Optional dependency: elastix
if ELASTIX_ENABLED:
    from microscopy_proc.funcs.elastix_funcs import ElastixFuncs
else:
    ElastixFuncs = import_extra_error_func("elastix")


class Pipeline:
    logger = init_logger(__name__)

    ###################################################################################################
    # CHECK PFM FILE EXISTS
    ###################################################################################################
    @classmethod
    def _check_files_exist(cls, pfm: ProjFpModel, pfm_fp_ls: tuple[str, ...] = tuple()):
        """
        Returns whether the fpm attribute in
        `pfm_fp_ls` is a filepath that already exsits.
        """
        cls.logger.debug(f"Iterating through filepaths in `pfm_fp_ls`: {pfm_fp_ls}")
        for pfm_fp in pfm_fp_ls:
            if os.path.exists(getattr(pfm, pfm_fp)):
                cls.logger.debug(f"{pfm_fp} already exists.")
                cls.logger.debug("Returning True.")
                return True
        cls.logger.debug("None of the filepaths in `pfm_fp_ls` exist.")
        cls.logger.debug("Returning False.")
        return False

    ###################################################################################################
    # GETTING PROJECT FILEPATH MODEL
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def get_pfm(cls, proj_dir: str) -> ProjFpModelBase:
        """
        Returns a ProjFpModel object created from the project directory.
        """
        pfm = ProjFpModel(proj_dir)
        pfm.make_subdirs()
        return pfm

    @classmethod
    @log_func_decorator(logger)
    def get_pfm_tuning(cls, proj_dir: str) -> ProjFpModelBase:
        """
        Returns a ProjFpModel object created from the project directory.
        """
        pfm_tuning = ProjFpModelTuning(proj_dir)
        pfm_tuning.make_subdirs()
        return pfm_tuning

    ###################################################################################################
    # UPDATE CONFIGS
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def update_configs(cls, pfm: ProjFpModel, **kwargs) -> ConfigParamsModel:
        """
        If config_params file does not exist, makes a new one.

        Then updates the config_params file with the kwargs.
        If there are no kwargs, will not update the file
        (other than making it if it did not exist).

        Also creates all the project sub-directories too.

        Finally, returns the ConfigParamsModel object.
        """
        cls.logger.debug("Making all the project sub-directories")
        cls.logger.debug("Reading/creating params json")
        try:
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            cls.logger.debug("The configs file exists")
        except FileNotFoundError:
            cls.logger.debug("The configs file does NOT exists")
            cls.logger.debug("Creating new configs file")
            configs = ConfigParamsModel()
            cls.logger.debug("Saving newly created configs file")
            write_json(pfm.config_params, configs.model_dump())
        cls.logger.debug("Updating and saving configs if kwargs is not empty")
        if kwargs != {}:
            cls.logger.debug(f"kwargs are given: {kwargs}")
            configs = configs.model_validate(configs.model_copy(update=kwargs))
            cls.logger.debug("Updating the configs file")
            write_json(pfm.config_params, configs.model_dump())
        cls.logger.debug("Returning the configs file")
        return configs

    ###################################################################################################
    # CONVERT TIFF TO ZARR FUNCS
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def tiff2zarr(cls, pfm: ProjFpModel, in_fp: str, overwrite: bool = False) -> None:
        """
        _summary_

        Parameters
        ----------
        pfm : ProjFpModel
            _description_
        in_fp : str
            _description_
        overwrite : bool, optional
            _description_, by default False

        Raises
        ------
        ValueError
            _description_
        """
        if not overwrite and cls._check_files_exist(pfm, ("raw",)):
            return
        cls.logger.debug("Reading config params")
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        cls.logger.debug("Making zarr from tiff file(s)")
        with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=6)):
            if os.path.isdir(in_fp):
                cls.logger.debug(f"in_fp ({in_fp}) is a directory")
                cls.logger.debug("Making zarr from tiff file stack in directory")
                Tiff2ZarrFuncs.tiffs2zarr(
                    in_fp_ls=tuple(
                        natsorted(
                            (
                                os.path.join(in_fp, i)
                                for i in os.listdir(in_fp)
                                if re.search(r".tif$", i)
                            )
                        )
                    ),
                    out_fp=pfm.raw,
                    chunks=configs.zarr_chunksize,
                )
            elif os.path.isfile(in_fp):
                cls.logger.debug(f"in_fp ({in_fp}) is a file")
                cls.logger.debug("Making zarr from big-tiff file")
                Tiff2ZarrFuncs.btiff2zarr(
                    in_fp=in_fp,
                    out_fp=pfm.raw,
                    chunks=configs.zarr_chunksize,
                )
            else:
                raise ValueError(f'Input file path, "{in_fp}" does not exist.')

    ###################################################################################################
    # REGISTRATION PIPELINE FUNCS
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def ref_prepare(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(
            pfm, ("ref", "annot", "map", "affine", "bspline")
        ):
            return
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        # Making ref_fp_model of original atlas images filepaths
        rfm = RefFpModel(
            configs.atlas_dir,
            configs.ref_version,
            configs.annot_version,
            configs.map_version,
        )
        # Making atlas images
        for fp_i, fp_o in [
            (rfm.ref, pfm.ref),
            (rfm.annot, pfm.annot),
        ]:
            # Reading
            arr = tifffile.imread(fp_i)
            # Reorienting
            arr = RegFuncs.reorient(arr, configs.ref_orient_ls)
            # Slicing
            arr = arr[
                slice(*configs.ref_z_trim),
                slice(*configs.ref_y_trim),
                slice(*configs.ref_x_trim),
            ]
            # Saving
            tifffile.imwrite(fp_o, arr)
        # Copying region mapping json to project folder
        shutil.copyfile(rfm.map, pfm.map)
        # Copying transformation files
        shutil.copyfile(rfm.affine, pfm.affine)
        shutil.copyfile(rfm.bspline, pfm.bspline)

    @classmethod
    @log_func_decorator(logger)
    def img_rough(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("downsmpl1",)):
            return
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        with cluster_proc_contxt(LocalCluster()):
            # Reading
            raw_arr = da.from_zarr(pfm.raw)
            # Rough downsample
            downsmpl1_arr = RegFuncs.downsmpl_rough(
                raw_arr, configs.z_rough, configs.y_rough, configs.x_rough
            )
            # Computing (from dask array)
            downsmpl1_arr = downsmpl1_arr.compute()
            # Saving
            tifffile.imwrite(pfm.downsmpl1, downsmpl1_arr)

    @classmethod
    @log_func_decorator(logger)
    def img_fine(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("downsmpl2",)):
            return
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        # Reading
        downsmpl1_arr = tifffile.imread(pfm.downsmpl1)
        # Fine downsample
        downsmpl2_arr = RegFuncs.downsmpl_fine(
            downsmpl1_arr, configs.z_fine, configs.y_fine, configs.x_fine
        )
        # Saving
        tifffile.imwrite(pfm.downsmpl2, downsmpl2_arr)

    @classmethod
    @log_func_decorator(logger)
    def img_trim(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("trimmed",)):
            return
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        # Reading
        downsmpl2_arr = tifffile.imread(pfm.downsmpl2)
        # Trim
        trimmed_arr = downsmpl2_arr[
            slice(*configs.z_trim),
            slice(*configs.y_trim),
            slice(*configs.x_trim),
        ]
        # Saving
        tifffile.imwrite(pfm.trimmed, trimmed_arr)

    @classmethod
    @log_func_decorator(logger)
    def elastix_registration(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("regresult",)):
            return
        # Running Elastix registration
        ElastixFuncs.registration(
            fixed_img_fp=pfm.trimmed,
            moving_img_fp=pfm.ref,
            output_img_fp=pfm.regresult,
            affine_fp=pfm.affine,
            bspline_fp=pfm.bspline,
        )

    ###################################################################################################
    # MASK PIPELINE FUNCS
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def make_mask(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Makes mask of actual image in reference space.
        Also stores # and proportion of existent voxels
        for each region.
        """
        if not overwrite and cls._check_files_exist(
            pfm, ("mask", "outline", "mask_reg", "mask_df")
        ):
            return
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        # Reading annot (proj oriented and trimmed) and trimmed imgs
        annot_arr = tifffile.imread(pfm.annot)
        trimmed_arr = tifffile.imread(pfm.trimmed)
        # Storing annot_arr shape
        s = annot_arr.shape
        # Making mask
        blur_arr = Gf.gauss_blur_filt(trimmed_arr, configs.mask_gaus_blur)
        tifffile.imwrite(pfm.premask_blur, blur_arr)
        mask_arr = Gf.manual_thresh(blur_arr, configs.mask_thresh)
        tifffile.imwrite(pfm.mask, mask_arr)

        # Make outline
        outline_df = MaskFuncs.make_outline(mask_arr)
        # Transformix on coords
        outline_df[[Coords.Z.value, Coords.Y.value, Coords.X.value]] = (
            ElastixFuncs.transformation_coords(
                outline_df,
                pfm.ref,
                pfm.regresult,
            )[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
            .round(0)
            .astype(np.int32)
        )
        # Filtering out of bounds coords
        outline_df = outline_df.query(
            f"({Coords.Z.value} >= 0) & ({Coords.Z.value} < {s[0]}) & "
            f"({Coords.Y.value} >= 0) & ({Coords.Y.value} < {s[1]}) & "
            f"({Coords.X.value} >= 0) & ({Coords.X.value} < {s[2]})"
        )

        # Make outline img (1 for in, 2 for out)
        # TODO: convert to return np.array and save out-of-function
        VisualCheckFuncsTiff.coords2points(
            outline_df[outline_df.is_in == 1], s, pfm.outline
        )
        in_arr = tifffile.imread(pfm.outline)
        VisualCheckFuncsTiff.coords2points(
            outline_df[outline_df.is_in == 0], s, pfm.outline
        )
        out_arr = tifffile.imread(pfm.outline)
        tifffile.imwrite(pfm.outline, in_arr + out_arr * 2)

        # Fill in outline to recreate mask (not perfect)
        mask_reg_arr = MaskFuncs.fill_outline(outline_df, s)
        # Opening (removes FP) and closing (fills FN)
        mask_reg_arr = ndimage.binary_closing(mask_reg_arr, iterations=2).astype(
            np.uint8
        )
        mask_reg_arr = ndimage.binary_opening(mask_reg_arr, iterations=2).astype(
            np.uint8
        )
        # Saving
        tifffile.imwrite(pfm.mask_reg, mask_reg_arr)

        # Counting mask voxels in each region
        # Getting original annot fp by making ref_fp_model
        rfm = RefFpModel(
            configs.atlas_dir,
            configs.ref_version,
            configs.annot_version,
            configs.map_version,
        )
        # Reading original annot
        annot_orig_arr = tifffile.imread(rfm.annot)
        # Getting the annotation name for every cell (zyx coord)
        mask_df = pd.merge(
            left=MaskFuncs.mask2region_counts(
                np.full(annot_orig_arr.shape, 1), annot_orig_arr
            ),
            right=MaskFuncs.mask2region_counts(mask_reg_arr, annot_arr),
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("_annot", "_mask"),
        ).fillna(0)
        # Reading annotation mappings json
        annot_df = MapFuncs.annot_dict2df(read_json(pfm.map))
        # Combining (summing) the mask_df volumes for parent regions using the annot_df
        mask_df = MapFuncs.combine_nested_regions(mask_df, annot_df)
        # Calculating proportion of mask volume in each region
        mask_df[MaskColumns.VOLUME_PROP.value] = (
            mask_df[MaskColumns.VOLUME_MASK.value]
            / mask_df[MaskColumns.VOLUME_ANNOT.value]
        )
        # Selecting and ordering relevant columns
        mask_df = mask_df[[*ANNOT_COLUMNS_FINAL, *enum2list(MaskColumns)]]
        # Saving
        mask_df.to_parquet(pfm.mask_df)

    ###################################################################################################
    # CROP RAW ZARR TO MAKE TUNING ZARR
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def make_tuning_arr(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Crop raw zarr to make a smaller zarr for tuning the cell counting pipeline.
        """
        cls.logger.debug("Converting/ensuring pfm is production filepaths (copy)")
        pfm = ProjFpModel(pfm.root_dir)
        pfm_tuning = ProjFpModelTuning(pfm.root_dir)
        cls.logger.debug("Reading config params")
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        cls.logger.debug("Reading raw zarr")
        raw_arr = da.from_zarr(pfm.raw)
        cls.logger.debug("Cropping raw zarr")
        raw_arr = raw_arr[
            slice(*configs.tuning_z_trim),
            slice(*configs.tuning_y_trim),
            slice(*configs.tuning_x_trim),
        ]
        if not overwrite and cls._check_files_exist(pfm_tuning, ("raw",)):
            cls.logger.debug("Don't overwrite specified and raw zarr exists. Skipping.")
            return
        cls.logger.debug("Saving cropped raw zarr")
        raw_arr = disk_cache(raw_arr, pfm_tuning.raw)

    ###################################################################################################
    # CELL COUNTING PIPELINE FUNCS
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def img_overlap(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("overlap",)):
            return
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        # Making overlap image
        with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=4)):
            raw_arr = da.from_zarr(pfm.raw, chunks=configs.zarr_chunksize)
            overlap_arr = da_overlap(raw_arr, d=configs.overlap_depth)
            overlap_arr = disk_cache(overlap_arr, pfm.overlap)

    @classmethod
    @log_func_decorator(logger)
    def cellc1(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 1

        Top-hat filter (background subtraction)
        """
        if not overwrite and cls._check_files_exist(pfm, ("bgrm",)):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCUDACluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # Reading input images
            overlap_arr = da.from_zarr(pfm.overlap)
            # Declaring processing instructions
            bgrm_arr = da.map_blocks(
                Gf.tophat_filt,
                overlap_arr,
                configs.tophat_sigma,
            )
            # Computing and saving
            bgrm_arr = disk_cache(bgrm_arr, pfm.bgrm)

    @classmethod
    @log_func_decorator(logger)
    def cellc2(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 2

        Difference of Gaussians (edge detection)
        """
        if not overwrite and cls._check_files_exist(pfm, ("dog",)):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCUDACluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # Reading input images
            bgrm_arr = da.from_zarr(pfm.bgrm)
            # Declaring processing instructions
            dog_arr = da.map_blocks(
                Gf.dog_filt,
                bgrm_arr,
                configs.dog_sigma1,
                configs.dog_sigma2,
            )
            # Computing and saving
            dog_arr = disk_cache(dog_arr, pfm.dog)

    @classmethod
    @log_func_decorator(logger)
    def cellc3(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 3

        Gaussian subtraction with large sigma for adaptive thresholding
        """
        if not overwrite and cls._check_files_exist(pfm, ("adaptv",)):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCUDACluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # Reading input images
            dog_arr = da.from_zarr(pfm.dog)
            # Declaring processing instructions
            adaptv_arr = da.map_blocks(
                Gf.gauss_subt_filt,
                dog_arr,
                configs.large_gauss_sigma,
            )
            # Computing and saving
            adaptv_arr = disk_cache(adaptv_arr, pfm.adaptv)

    @classmethod
    @log_func_decorator(logger)
    def cellc4(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 4

        Currently, manual thresholding.
        Ideally, mean thresholding with standard deviation offset
        """
        if not overwrite and cls._check_files_exist(pfm, ("threshd",)):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # # Visually inspect sd offset
            # t_p =adaptv_arr.sum() / (np.prod(adaptv_arr.shape) - (adaptv_arr == 0).sum())
            # t_p = t_p.compute()
            # Reading input images
            adaptv_arr = da.from_zarr(pfm.adaptv)
            # Declaring processing instructions
            threshd_arr = da.map_blocks(
                Cf.manual_thresh,
                adaptv_arr,
                configs.threshd_value,
            )
            # Computing and saving
            threshd_arr = disk_cache(threshd_arr, pfm.threshd)

    @classmethod
    @log_func_decorator(logger)
    def cellc5(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 5

        Getting object sizes
        """
        if not overwrite and cls._check_files_exist(pfm, ("threshd_volumes",)):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster(n_workers=6, threads_per_worker=1)):
            # Reading input images
            threshd_arr = da.from_zarr(pfm.threshd)
            # Declaring processing instructions
            threshd_volumes_arr = da.map_blocks(
                Cf.label_with_volumes,
                threshd_arr,
            )
            # Computing and saving
            threshd_volumes_arr = disk_cache(threshd_volumes_arr, pfm.threshd_volumes)

    @classmethod
    @log_func_decorator(logger)
    def cellc6(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 6

        Filter out large objects (likely outlines, not cells)
        """
        if not overwrite and cls._check_files_exist(pfm, ("threshd_filt",)):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # Reading input images
            threshd_volumes_arr = da.from_zarr(pfm.threshd_volumes)
            # Declaring processing instructions
            threshd_filt_arr = da.map_blocks(
                Cf.volume_filter,
                threshd_volumes_arr,
                configs.min_threshd_size,
                configs.max_threshd_size,
            )
            # Computing and saving
            threshd_filt_arr = disk_cache(threshd_filt_arr, pfm.threshd_filt)

    @classmethod
    @log_func_decorator(logger)
    def cellc7(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 7

        Get maxima of image masked by labels.
        """
        if not overwrite and cls._check_files_exist(pfm, ("maxima",)):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCUDACluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # Reading input images
            overlap_arr = da.from_zarr(pfm.overlap)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt)
            # Declaring processing instructions
            maxima_arr = da.map_blocks(
                Gf.get_local_maxima,
                overlap_arr,
                configs.maxima_sigma,
                threshd_filt_arr,
            )
            # Computing and saving
            maxima_arr = disk_cache(maxima_arr, pfm.maxima)

    @classmethod
    @log_func_decorator(logger)
    def cellc8(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 8

        Watershed segmentation volumes.
        """
        if not overwrite and cls._check_files_exist(pfm, ("wshed_volumes",)):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster(n_workers=3, threads_per_worker=1)):
            # n_workers=2
            # Reading input images
            overlap_arr = da.from_zarr(pfm.overlap)
            maxima_arr = da.from_zarr(pfm.maxima)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt)
            # Declaring processing instructions
            wshed_volumes_arr = da.map_blocks(
                Cf.wshed_segm_volumes,
                overlap_arr,
                maxima_arr,
                threshd_filt_arr,
            )
            # Computing and saving
            wshed_volumes_arr = disk_cache(wshed_volumes_arr, pfm.wshed_volumes)

    @classmethod
    @log_func_decorator(logger)
    def cellc9(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 9

        Filter out large watershed objects (again cell areas, not cells).
        """
        if not overwrite and cls._check_files_exist(pfm, ("wshed_filt",)):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # Reading input images
            wshed_volumes_arr = da.from_zarr(pfm.wshed_volumes)
            # Declaring processing instructions
            wshed_filt_arr = da.map_blocks(
                Cf.volume_filter,
                wshed_volumes_arr,
                configs.min_wshed_size,
                configs.max_wshed_size,
            )
            # Computing and saving
            wshed_filt_arr = disk_cache(wshed_filt_arr, pfm.wshed_filt)

    @classmethod
    @log_func_decorator(logger)
    def cellc10(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 10

        Trimming filtered regions overlaps to make:
        - Trimmed maxima image
        - Trimmed threshold image
        - Trimmed watershed image
        """
        if not overwrite and cls._check_files_exist(
            pfm, ("maxima_final", "threshd_final", "wshed_final")
        ):
            return
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # Reading input images
            maxima_arr = da.from_zarr(pfm.maxima)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt)
            wshed_volumes_arr = da.from_zarr(pfm.wshed_volumes)
            # Declaring processing instructions
            maxima_final_arr = da_trim(maxima_arr, d=configs.overlap_depth)
            threshd_final_arr = da_trim(threshd_filt_arr, d=configs.overlap_depth)
            wshed_final_arr = da_trim(wshed_volumes_arr, d=configs.overlap_depth)
            # Computing and saving
            maxima_final_arr = disk_cache(maxima_final_arr, pfm.maxima_final)
            threshd_final_arr = disk_cache(threshd_final_arr, pfm.threshd_final)
            wshed_final_arr = disk_cache(wshed_final_arr, pfm.wshed_final)

    @classmethod
    @log_func_decorator(logger)
    def cellc11(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 11

        Calculate the maxima and watershed, save the cells.

        Basically a repeat of cellc8 and cellc9 but needs to be done to
        get the cell volumes in a table. Hence, don't run cellc8 and cellc9 if
        you don't want to view the cells visually (good for pipeline, not for tuning).
        """
        if not overwrite and cls._check_files_exist(pfm, ("cells_raw_df",)):
            return
        with cluster_proc_contxt(LocalCluster(n_workers=2, threads_per_worker=1)):
            # n_workers=2
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # Reading input images
            raw_arr = da.from_zarr(pfm.raw)
            overlap_arr = da.from_zarr(pfm.overlap)
            maxima_arr = da.from_zarr(pfm.maxima)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt)
            # Declaring processing instructions
            # Getting maxima coords and cell measures in table
            cells_df = block2coords(
                Cf.get_cells,
                raw_arr,
                overlap_arr,
                maxima_arr,
                threshd_filt_arr,
                configs.overlap_depth,
            )
            # Converting from dask to pandas
            cells_df = cells_df.compute()
            # Filtering out by volume (same filter cellc9_pipeline volume_filter)
            cells_df = cells_df.query(
                f"({CellColumns.VOLUME.value} >= {configs.min_wshed_size}) & "
                f"({CellColumns.VOLUME.value} <= {configs.max_wshed_size})"
            )
            # Computing and saving as parquet
            cells_df.to_parquet(pfm.cells_raw_df)

    @classmethod
    @log_func_decorator(logger)
    def cellc_coords_only(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Get maxima coords.
        Very basic but faster version of cellc11_pipeline get_cells.
        """
        if not overwrite and cls._check_files_exist(pfm, ("maxima_df",)):
            return
        # Reading filtered and maxima images (trimmed - orig space)
        with cluster_proc_contxt(LocalCluster(n_workers=6, threads_per_worker=1)):
            # Read filtered and maxima images (trimmed - orig space)
            maxima_final_arr = da.from_zarr(pfm.maxima_final)
            # Declaring processing instructions
            # Storing coords of each maxima in df
            coords_df = block2coords(
                Gf.get_coords,
                maxima_final_arr,
            )
            # Converting from dask to pandas
            coords_df = coords_df.compute()
            # Computing and saving as parquet
            coords_df.to_parquet(pfm.maxima_df)

    ###################################################################################################
    # CELL COUNT REALIGNMENT TO REFERENCE AND AGGREGATION PIPELINE FUNCS
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def transform_coords(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        `in_id` and `out_id` are either maxima or region

        NOTE: saves the cells_trfm dataframe as pandas parquet.
        """
        if not overwrite and cls._check_files_exist(pfm, ("cells_trfm_df",)):
            return
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        with cluster_proc_contxt(LocalCluster(n_workers=4, threads_per_worker=1)):
            # Setting output key (in the form "<maxima/region>_trfm_df")
            # Getting cell coords
            cells_df = pd.read_parquet(pfm.cells_raw_df)
            # Sanitising (removing smb columns)
            cells_df = sanitise_smb_df(cells_df)
            # Taking only Coords.Z.value, Coords.Y.value, Coords.X.value coord columns
            cells_df = cells_df[enum2list(Coords)]
            # Scaling to resampled rough space
            # NOTE: this downsampling uses slicing so must be computed differently
            cells_df = cells_df / np.array(
                (configs.z_rough, configs.y_rough, configs.x_rough)
            )
            # Scaling to resampled space
            cells_df = cells_df * np.array(
                (configs.z_fine, configs.y_fine, configs.x_fine)
            )
            # Trimming/offsetting to sliced space
            cells_df = cells_df - np.array(
                [s[0] or 0 for s in (configs.z_trim, configs.y_trim, configs.x_trim)]
            )
            # Converting back to DataFrame
            cells_df = pd.DataFrame(cells_df, columns=enum2list(Coords))

            cells_trfm_df = ElastixFuncs.transformation_coords(
                cells_df, pfm.ref, pfm.regresult
            )
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_df = dd.from_pandas(cells_df, npartitions=1)
            # Fitting resampled space to atlas image with Transformix (from Elastix registration step)
            # cells_df = cells_df.repartition(
            #     npartitions=int(np.ceil(cells_df.shape[0].compute() / ROWSPPART))
            # )
            # cells_df = cells_df.map_partitions(
            #     ElastixFuncs.transformation_coords, pfm.ref, pfm.regresult
            # )
            cells_trfm_df.to_parquet(pfm.cells_trfm_df)

    @classmethod
    @log_func_decorator(logger)
    def cell_mapping(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Using the transformed cell coordinates, get the region ID and name for each cell
        corresponding to the reference atlas.

        NOTE: saves the cells dataframe as pandas parquet.
        """
        if not overwrite and cls._check_files_exist(pfm, ("cells_df",)):
            return
        # Getting region for each detected cell (i.e. row) in cells_df
        with cluster_proc_contxt(LocalCluster()):
            # Reading cells_raw and cells_trfm dataframes
            cells_df = pd.read_parquet(pfm.cells_raw_df)
            coords_trfm = pd.read_parquet(pfm.cells_trfm_df)
            # Sanitising (removing smb columns)
            cells_df = sanitise_smb_df(cells_df)
            coords_trfm = sanitise_smb_df(coords_trfm)
            # Making unique incrementing index
            cells_df = cells_df.reset_index(drop=True)
            # Setting the transformed coords
            cells_df[f"{Coords.Z.value}_{TRFM}"] = coords_trfm[Coords.Z.value].values
            cells_df[f"{Coords.Y.value}_{TRFM}"] = coords_trfm[Coords.Y.value].values
            cells_df[f"{Coords.X.value}_{TRFM}"] = coords_trfm[Coords.X.value].values

            # Reading annotation image
            annot_arr = tifffile.imread(pfm.annot)
            # Getting the annotation ID for every cell (zyx coord)
            # Getting transformed coords (that are within tbe bounds_arr, and their corresponding idx)
            s = annot_arr.shape
            trfm_loc = (
                cells_df[
                    [
                        f"{Coords.Z.value}_{TRFM}",
                        f"{Coords.Y.value}_{TRFM}",
                        f"{Coords.X.value}_{TRFM}",
                    ]
                ]
                .round(0)
                .astype(np.int32)
                .query(
                    f"({Coords.Z.value}_{TRFM} >= 0) & ({Coords.Z.value}_{TRFM} < {s[0]}) & "
                    f"({Coords.Y.value}_{TRFM} >= 0) & ({Coords.Y.value}_{TRFM} < {s[1]}) & "
                    f"({Coords.X.value}_{TRFM} >= 0) & ({Coords.X.value}_{TRFM} < {s[2]})"
                )
            )
            # Getting the pixel values of each valid transformed coord (hence the specified index)
            # By complex array indexing on ar_annot's (z, y, x) dimensions.
            # nulls are imputed with -1
            cells_df[AnnotColumns.ID.value] = pd.Series(
                annot_arr[*trfm_loc.values.T].astype(np.uint32),
                index=trfm_loc.index,
            ).fillna(-1)

            # Reading annotation mappings dataframe
            annot_df = MapFuncs.annot_dict2df(read_json(pfm.map))
            # Getting the annotation name for every cell (zyx coord)
            cells_df = MapFuncs.df_map_ids(cells_df, annot_df)
            # Saving to disk
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_df = dd.from_pandas(cells_df)
            cells_df.to_parquet(pfm.cells_df)

    @classmethod
    @log_func_decorator(logger)
    def group_cells(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Grouping cells by region name and aggregating total cell volume
        and cell count for each region.

        NOTE: saves the cells_agg dataframe as pandas parquet.
        """
        if not overwrite and cls._check_files_exist(pfm, ("cells_agg_df",)):
            return
        # Making cells_agg_df
        with cluster_proc_contxt(LocalCluster()):
            # Reading cells dataframe
            cells_df = pd.read_parquet(pfm.cells_df)
            # Sanitising (removing smb columns)
            cells_df = sanitise_smb_df(cells_df)
            # Grouping cells by region name and aggregating on given mappings
            cells_agg_df = cells_df.groupby(AnnotColumns.ID.value).agg(
                CELL_AGG_MAPPINGS
            )
            cells_agg_df.columns = list(CELL_AGG_MAPPINGS.keys())
            # Reading annotation mappings dataframe
            # Making df of region names and their parent region names
            annot_df = MapFuncs.annot_dict2df(read_json(pfm.map))
            # Combining (summing) the cells_agg_df values for parent regions using the annot_df
            cells_agg_df = MapFuncs.combine_nested_regions(cells_agg_df, annot_df)
            # Calculating integrated average intensity (sum_intensity / volume)
            cells_agg_df[CellColumns.IOV.value] = (
                cells_agg_df[CellColumns.SUM_INTENSITY.value]
                / cells_agg_df[CellColumns.VOLUME.value]
            )
            # Selecting and ordering relevant columns
            cells_agg_df = cells_agg_df[[*ANNOT_COLUMNS_FINAL, *enum2list(CellColumns)]]
            # Saving to disk
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_agg = dd.from_pandas(cells_agg)
            cells_agg_df.to_parquet(pfm.cells_agg_df)

    @classmethod
    @log_func_decorator(logger)
    def cells2csv(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("cells_agg_csv",)):
            return
        # Reading cells dataframe
        cells_agg_df = pd.read_parquet(pfm.cells_agg_df)
        # Sanitising (removing smb columns)
        cells_agg_df = sanitise_smb_df(cells_agg_df)
        # Saving to csv
        cells_agg_df.to_csv(pfm.cells_agg_csv)

    ###################################################################################################
    # VISUAL CHECK
    ###################################################################################################
    @classmethod
    @log_func_decorator(logger)
    def coords2points_raw(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("points_raw",)):
            return
        with cluster_proc_contxt(LocalCluster()):
            VisualCheckFuncsDask.coords2points(
                coords=pd.read_parquet(pfm.cells_raw_df),
                shape=da.from_zarr(pfm.raw).shape,
                out_fp=pfm.points_raw,
            )

    @classmethod
    @log_func_decorator(logger)
    def coords2heatmap_raw(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("heatmap_raw",)):
            return
        with cluster_proc_contxt(LocalCluster()):
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            VisualCheckFuncsDask.coords2heatmap(
                coords=pd.read_parquet(pfm.cells_raw_df),
                shape=da.from_zarr(pfm.raw).shape,
                out_fp=pfm.heatmap_raw,
                radius=configs.heatmap_raw_radius,
            )

    @classmethod
    @log_func_decorator(logger)
    def coords2points_trfm(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("points_trfm",)):
            return
        with cluster_proc_contxt(LocalCluster()):
            VisualCheckFuncsTiff.coords2points(
                coords=pd.read_parquet(pfm.cells_trfm_df),
                shape=tifffile.imread(pfm.ref).shape,
                out_fp=pfm.points_trfm,
            )

    @classmethod
    @log_func_decorator(logger)
    def coords2heatmap_trfm(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("heatmap_trfm",)):
            return
        with cluster_proc_contxt(LocalCluster()):
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            VisualCheckFuncsTiff.coords2heatmap(
                coords=pd.read_parquet(pfm.cells_trfm_df),
                shape=tifffile.imread(pfm.ref).shape,
                out_fp=pfm.heatmap_trfm,
                radius=configs.heatmap_trfm_radius,
            )

    ###################################################################################################
    # COMBINING/MERGING ARRAYS IN RGB LAYERS
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def combine_reg(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("comb_reg",)):
            return
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.trimmed, pfm.regresult, pfm.regresult),
            # 2nd regresult means the combining works in ImageJ
            fp_out=pfm.comb_reg,
        )

    @classmethod
    @log_func_decorator(logger)
    def combine_cellc(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("comb_cellc",)):
            return
        configs = ConfigParamsModel.read_fp(pfm.config_params)
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.raw, pfm.threshd_final, pfm.wshed_final),
            fp_out=pfm.comb_cellc,
            trimmer=(
                slice(*configs.combine_cellc_z_trim),
                slice(*configs.combine_cellc_y_trim),
                slice(*configs.combine_cellc_x_trim),
            ),
        )

    @classmethod
    @log_func_decorator(logger)
    def combine_points(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        if not overwrite and cls._check_files_exist(pfm, ("comb_points",)):
            return
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.ref, pfm.annot, pfm.heatmap_trfm),
            # 2nd regresult means the combining works in ImageJ
            fp_out=pfm.comb_points,
        )

    ###################################################################################################
    # ALL PIPELINE FUNCTION
    ###################################################################################################

    @classmethod
    @log_func_decorator(logger)
    def run_all(cls, in_fp: str, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Running all pipelines in order.
        """
        # Running all pipelines in order
        Pipeline.tiff2zarr(pfm, in_fp, overwrite=overwrite)
        Pipeline.ref_prepare(pfm, overwrite=overwrite)
        Pipeline.img_rough(pfm, overwrite=overwrite)
        Pipeline.img_fine(pfm, overwrite=overwrite)
        Pipeline.img_trim(pfm, overwrite=overwrite)
        Pipeline.elastix_registration(pfm, overwrite=overwrite)
        Pipeline.make_mask(pfm, overwrite=overwrite)
        Pipeline.img_overlap(pfm, overwrite=overwrite)
        Pipeline.cellc1(pfm, overwrite=overwrite)
        Pipeline.cellc2(pfm, overwrite=overwrite)
        Pipeline.cellc3(pfm, overwrite=overwrite)
        Pipeline.cellc4(pfm, overwrite=overwrite)
        Pipeline.cellc5(pfm, overwrite=overwrite)
        Pipeline.cellc6(pfm, overwrite=overwrite)
        Pipeline.cellc7(pfm, overwrite=overwrite)
        Pipeline.cellc8(pfm, overwrite=overwrite)
        Pipeline.cellc9(pfm, overwrite=overwrite)
        Pipeline.cellc10(pfm, overwrite=overwrite)
        Pipeline.cellc11(pfm, overwrite=overwrite)
        Pipeline.cellc_coords_only(pfm, overwrite=overwrite)
        Pipeline.transform_coords(pfm, overwrite=overwrite)
        Pipeline.cell_mapping(pfm, overwrite=overwrite)
        Pipeline.group_cells(pfm, overwrite=overwrite)
        Pipeline.cells2csv(pfm, overwrite=overwrite)
        Pipeline.coords2points_raw(pfm, overwrite=overwrite)
        Pipeline.coords2heatmap_raw(pfm, overwrite=overwrite)
        Pipeline.coords2points_trfm(pfm, overwrite=overwrite)
        Pipeline.coords2heatmap_trfm(pfm, overwrite=overwrite)
        Pipeline.combine_reg(pfm, overwrite=overwrite)
        Pipeline.combine_cellc(pfm, overwrite=overwrite)
        Pipeline.combine_points(pfm, overwrite=overwrite)
