import os
import re
import shutil
from typing import Type

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster
from natsort import natsorted
from scipy import ndimage

from microscopy_proc import DASK_CUDA_ENABLED, ELASTIX_ENABLED, package_is_importable
from microscopy_proc.constants import (
    ANNOT_COLUMNS_FINAL,
    CELL_AGG_MAPPINGS,
    TRFM,
    AnnotColumns,
    CellColumns,
    Coords,
    MaskColumns,
)
from microscopy_proc.funcs.cpu_cellc_funcs import CpuCellcFuncs
from microscopy_proc.funcs.map_funcs import MapFuncs
from microscopy_proc.funcs.mask_funcs import MaskFuncs
from microscopy_proc.funcs.reg_funcs import RegFuncs
from microscopy_proc.funcs.tiff2zarr_funcs import Tiff2ZarrFuncs
from microscopy_proc.funcs.visual_check_funcs_tiff import VisualCheckFuncsTiff
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import (
    block2coords,
    cluster_process,
    da_overlap,
    disk_cache,
)
from microscopy_proc.utils.diagnostics_utils import file_exists_msg
from microscopy_proc.utils.io_utils import (
    read_json,
    sanitise_smb_df,
    write_json,
    write_parquet,
)
from microscopy_proc.utils.logging_utils import init_logger_file
from microscopy_proc.utils.misc_utils import enum2list, import_extra_error_func
from microscopy_proc.utils.proj_org_utils import (
    ProjFpModel,
    ProjFpModelBase,
    ProjFpModelTuning,
    RefFpModel,
)

logger = init_logger_file(__name__)


# Optional dependency: gpu (with dask-cuda)
if DASK_CUDA_ENABLED:
    from dask_cuda import LocalCUDACluster
else:
    LocalCUDACluster = lambda: LocalCluster(n_workers=1, threads_per_worker=1)
    logger.warning(
        "Warning Dask-Cuda functionality not installed.\n"
        "Using single GPU functionality instead (1 worker)\n"
        "Dask-Cuda currently only available on Linux"
    )
# Optional dependency: gpu
if package_is_importable("cupy"):
    from microscopy_proc.funcs.gpu_cellc_funcs import GpuCellcFuncs
else:
    # TODO: allow more flexibility in number of workers here
    LocalCUDACluster = lambda: LocalCluster(n_workers=2, threads_per_worker=1)
    GpuCellcFuncs = CpuCellcFuncs
    logger.warning(
        "Warning GPU functionality not installed.\n"
        "Using CPU functionality instead (much slower).\n"
        'Can install with `pip install "microscopy_proc[gpu]"`'
    )
# Optional dependency: elastix
if ELASTIX_ENABLED:
    from microscopy_proc.funcs.elastix_funcs import ElastixFuncs
else:
    ElastixFuncs = import_extra_error_func("elastix")
    logger.warning(
        "Warning Elastix functionality not installed and unavailable.\n"
        'Can install with `pip install "microscopy_proc[elastix]"`'
    )


class Pipeline:
    # Clusters
    # heavy (few workers - carrying high RAM computations)
    heavy_n_workers = 2
    heavy_threads_per_worker = 1
    # busy (many workers - carrying low RAM computations)
    busy_n_workers = 6
    busy_threads_per_worker = 2
    # gpu
    _gpu_cluster = LocalCUDACluster
    # GPU enabled cell funcs
    cellc_funcs: Type[CpuCellcFuncs] = GpuCellcFuncs

    ###################################################################################################
    # SETTING PROCESSING CONFIGS (NUMBER OF WORKERS, GPU ENABLED, ETC.)
    ###################################################################################################

    @classmethod
    def heavy_cluster(cls):
        return LocalCluster(n_workers=cls.heavy_n_workers, threads_per_worker=cls.heavy_threads_per_worker)

    @classmethod
    def busy_cluster(cls):
        return LocalCluster(n_workers=cls.busy_n_workers, threads_per_worker=cls.busy_threads_per_worker)

    @classmethod
    def gpu_cluster(cls):
        return cls._gpu_cluster()

    @classmethod
    def set_gpu(cls, enabled: bool = True):
        if enabled:
            cls._gpu_cluster = LocalCUDACluster
            cls.cellc_funcs = GpuCellcFuncs
        else:
            cls._gpu_cluster = lambda: LocalCluster(
                n_workers=cls.heavy_n_workers, threads_per_worker=cls.heavy_threads_per_worker
            )
            cls.cellc_funcs = CpuCellcFuncs

    ###################################################################################################
    # GETTING PROJECT FILEPATH MODEL
    ###################################################################################################

    @classmethod
    def get_pfm(cls, proj_dir: str) -> ProjFpModelBase:
        """
        Returns a ProjFpModel object created from the project directory.
        """
        pfm = ProjFpModel(proj_dir)
        return pfm

    @classmethod
    def get_pfm_tuning(cls, proj_dir: str) -> ProjFpModelBase:
        """
        Returns a ProjFpModel object created from the project directory.
        """
        pfm_tuning = ProjFpModelTuning(proj_dir)
        return pfm_tuning

    ###################################################################################################
    # UPDATE CONFIGS
    ###################################################################################################

    @classmethod
    def update_configs(cls, pfm: ProjFpModelBase, **kwargs) -> ConfigParamsModel:
        """
        If config_params file does not exist, makes a new one.

        Then updates the config_params file with the kwargs.
        If there are no kwargs, will not update the file
        (other than making it if it did not exist).

        Also creates all the project sub-directories too.

        Finally, returns the ConfigParamsModel object.
        """
        logger = init_logger_file()
        logger.debug("Making all the project sub-directories")
        logger.debug("Reading/creating params json")
        try:
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            logger.debug("The configs file exists so using this file.")
        except FileNotFoundError:
            logger.debug("The configs file does NOT exists.")
            configs = ConfigParamsModel()
            logger.debug("Saving newly created configs file.")
            write_json(pfm.config_params.val, configs.model_dump())
        if kwargs != {}:
            logger.debug(f"kwargs is not empty. They are: {kwargs}")
            configs_new = configs.model_validate(configs.model_copy(update=kwargs))
            if configs_new != configs:
                logger.debug("New configs are different from old configs. Overwriting to file.")
                write_json(pfm.config_params.val, configs_new.model_dump())
        logger.debug("Returning the configs file")
        return configs

    ###################################################################################################
    # CONVERT TIFF TO ZARR FUNCS
    ###################################################################################################

    @classmethod
    def tiff2zarr(cls, pfm: ProjFpModelBase, in_fp: str, overwrite: bool = False) -> None:
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
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.raw.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        logger.debug("Reading config params")
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        logger.debug("Making zarr from tiff file(s)")
        with cluster_process(LocalCluster(n_workers=1, threads_per_worker=6)):  # TODO: is this faster without cluster?
            if os.path.isdir(in_fp):
                logger.debug(f"in_fp ({in_fp}) is a directory")
                logger.debug("Making zarr from tiff file stack in directory")
                Tiff2ZarrFuncs.tiffs2zarr(
                    in_fp_ls=tuple(
                        natsorted((os.path.join(in_fp, i) for i in os.listdir(in_fp) if re.search(r".tif$", i)))
                    ),
                    out_fp=pfm.raw.val,
                    chunks=configs.zarr_chunksize,
                )
            elif os.path.isfile(in_fp):
                logger.debug(f"in_fp ({in_fp}) is a file")
                logger.debug("Making zarr from big-tiff file")
                Tiff2ZarrFuncs.btiff2zarr(
                    in_fp=in_fp,
                    out_fp=pfm.raw.val,
                    chunks=configs.zarr_chunksize,
                )
            else:
                raise ValueError(f'Input file path, "{in_fp}" does not exist.')

    ###################################################################################################
    # REGISTRATION PIPELINE FUNCS
    ###################################################################################################

    @classmethod
    def reg_ref_prepare(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.ref.val, pfm.annot.val, pfm.map.val, pfm.affine.val, pfm.bspline.val):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        # Making ref_fp_model of original atlas images filepaths
        rfm = RefFpModel(
            configs.atlas_dir,
            configs.ref_version,
            configs.annot_version,
            configs.map_version,
        )
        # Making atlas images
        for fp_i, fp_o in [
            (rfm.ref.val, pfm.ref.val),
            (rfm.annot.val, pfm.annot.val),
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
        shutil.copyfile(rfm.map.val, pfm.map.val)
        # Copying transformation files
        shutil.copyfile(rfm.affine.val, pfm.affine.val)
        shutil.copyfile(rfm.bspline.val, pfm.bspline.val)

    @classmethod
    def reg_img_rough(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite and os.path.exist(pfm.downsmpl1.val):
            return logger.warning(file_exists_msg(pfm.downsmpl1.val))
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        with cluster_process(cls.busy_cluster()):  # TODO:  is this faster without cluster?
            # Reading
            raw_arr = da.from_zarr(pfm.raw.val)
            # Rough downsample
            downsmpl1_arr = RegFuncs.downsmpl_rough(raw_arr, configs.z_rough, configs.y_rough, configs.x_rough)
            # Computing (from dask array)
            downsmpl1_arr = downsmpl1_arr.compute()
            # Saving
            tifffile.imwrite(pfm.downsmpl1.val, downsmpl1_arr)

    @classmethod
    def reg_img_fine(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.downsmpl2.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        # Reading
        downsmpl1_arr = tifffile.imread(pfm.downsmpl1.val)
        # Fine downsample
        downsmpl2_arr = RegFuncs.downsmpl_fine(downsmpl1_arr, configs.z_fine, configs.y_fine, configs.x_fine)
        # Saving
        tifffile.imwrite(pfm.downsmpl2.val, downsmpl2_arr)

    @classmethod
    def reg_img_trim(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.trimmed.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        # Reading
        downsmpl2_arr = tifffile.imread(pfm.downsmpl2.val)
        # Trim
        trimmed_arr = downsmpl2_arr[
            slice(*configs.z_trim),
            slice(*configs.y_trim),
            slice(*configs.x_trim),
        ]
        # Saving
        tifffile.imwrite(pfm.trimmed.val, trimmed_arr)

    @classmethod
    def reg_img_bound(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.bounded.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        # Asserting that lower bound is less than upper bound
        assert configs.lower_bound[0] < configs.upper_bound[0], (
            "Error in config parameters: " "lower bound condition must be less than upper bound condition."
        )
        assert configs.lower_bound[1] <= configs.lower_bound[0], (
            "Error in config parameters: "
            "lower bound final value must be less than or equal to lower bound condition."
        )
        assert configs.upper_bound[1] >= configs.upper_bound[0], (
            "Error in config parameters: "
            "upper bound final value must be greater than or equal to upper bound condition."
        )
        # Reading
        trimmed_arr = tifffile.imread(pfm.trimmed.val)
        bounded_arr = trimmed_arr
        # Bounding lower
        bounded_arr[bounded_arr < configs.lower_bound[0]] = configs.lower_bound[1]
        # Bounding upper
        bounded_arr[bounded_arr > configs.upper_bound[0]] = configs.upper_bound[1]
        # Saving
        tifffile.imwrite(pfm.bounded.val, bounded_arr)

    @classmethod
    def reg_elastix(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.regresult.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Running Elastix registration
        ElastixFuncs.registration(
            fixed_img_fp=pfm.bounded.val,
            moving_img_fp=pfm.ref.val,
            output_img_fp=pfm.regresult.val,
            affine_fp=pfm.affine.val,
            bspline_fp=pfm.bspline.val,
        )

    ###################################################################################################
    # MASK PIPELINE FUNCS
    ###################################################################################################

    @classmethod
    def make_mask(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Makes mask of actual image in reference space.
        Also stores # and proportion of existent voxels
        for each region.
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.mask_fill.val, pfm.mask_outline.val, pfm.mask_reg.val, pfm.mask_df.val):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        # Reading annot img (proj oriented and trimmed) and bounded img
        annot_arr = tifffile.imread(pfm.annot.val)
        bounded_arr = tifffile.imread(pfm.bounded.val)
        # Storing annot_arr shape
        s = annot_arr.shape
        # Making mask
        blur_arr = cls.cellc_funcs.gauss_blur_filt(bounded_arr, configs.mask_gaus_blur)
        tifffile.imwrite(pfm.premask_blur.val, blur_arr)
        mask_arr = cls.cellc_funcs.manual_thresh(blur_arr, configs.mask_thresh)
        tifffile.imwrite(pfm.mask_fill.val, mask_arr)

        # Make outline
        outline_df = MaskFuncs.make_outline(mask_arr)
        # Transformix on coords
        outline_df[[Coords.Z.value, Coords.Y.value, Coords.X.value]] = (
            ElastixFuncs.transformation_coords(
                outline_df,
                pfm.ref.val,
                pfm.regresult.val,
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
        VisualCheckFuncsTiff.coords2points(outline_df[outline_df.is_in == 1], s, pfm.mask_outline.val)
        in_arr = tifffile.imread(pfm.mask_outline.val)
        VisualCheckFuncsTiff.coords2points(outline_df[outline_df.is_in == 0], s, pfm.mask_outline.val)
        out_arr = tifffile.imread(pfm.mask_outline.val)
        tifffile.imwrite(pfm.mask_outline.val, in_arr + out_arr * 2)

        # Fill in outline to recreate mask (not perfect)
        mask_reg_arr = MaskFuncs.fill_outline(outline_df, s)
        # Opening (removes FP) and closing (fills FN)
        mask_reg_arr = ndimage.binary_closing(mask_reg_arr, iterations=2).astype(np.uint8)
        mask_reg_arr = ndimage.binary_opening(mask_reg_arr, iterations=2).astype(np.uint8)
        # Saving
        tifffile.imwrite(pfm.mask_reg.val, mask_reg_arr)

        # Counting mask voxels in each region
        # Getting original annot fp by making ref_fp_model
        rfm = RefFpModel(
            configs.atlas_dir,
            configs.ref_version,
            configs.annot_version,
            configs.map_version,
        )
        # Reading original annot
        annot_orig_arr = tifffile.imread(rfm.annot.val)
        # Getting the annotation name for every cell (zyx coord)
        mask_df = pd.merge(
            left=MaskFuncs.mask2region_counts(np.full(annot_orig_arr.shape, 1), annot_orig_arr),
            right=MaskFuncs.mask2region_counts(mask_reg_arr, annot_arr),
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("_annot", "_mask"),
        ).fillna(0)
        # Reading annotation mappings json
        annot_df = MapFuncs.annot_dict2df(read_json(pfm.map.val))
        # Combining (summing) the mask_df volumes for parent regions using the annot_df
        mask_df = MapFuncs.combine_nested_regions(mask_df, annot_df)
        # Calculating proportion of mask volume in each region
        mask_df[MaskColumns.VOLUME_PROP.value] = (
            mask_df[MaskColumns.VOLUME_MASK.value] / mask_df[MaskColumns.VOLUME_ANNOT.value]
        )
        # Selecting and ordering relevant columns
        mask_df = mask_df[[*ANNOT_COLUMNS_FINAL, *enum2list(MaskColumns)]]
        # Saving
        write_parquet(mask_df, pfm.mask_df.val)

    ###################################################################################################
    # CROP RAW ZARR TO MAKE TUNING ZARR
    ###################################################################################################

    @classmethod
    def make_tuning_arr(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Crop raw zarr to make a smaller zarr for tuning the cell counting pipeline.
        """
        logger = init_logger_file()
        logger.debug("Converting/ensuring pfm is production filepaths (copy)")
        root_dir = pfm.root_dir.val
        pfm = ProjFpModel(root_dir)
        pfm_tuning = ProjFpModelTuning(root_dir)
        logger.debug("Reading config params")
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        with cluster_process(cls.busy_cluster()):  # TODO:  is this faster without cluster?
            logger.debug("Reading raw zarr")
            raw_arr = da.from_zarr(pfm.raw.val)
            logger.debug("Cropping raw zarr")
            raw_arr = raw_arr[
                slice(*configs.tuning_z_trim),
                slice(*configs.tuning_y_trim),
                slice(*configs.tuning_x_trim),
            ]
            if not overwrite:
                for fp in (pfm_tuning.raw.val,):
                    if os.path.exists(fp):
                        return logger.warning(file_exists_msg(fp))
            logger.debug("Saving cropped raw zarr")
            raw_arr = disk_cache(raw_arr, pfm_tuning.raw.val)

    ###################################################################################################
    # CELL COUNTING PIPELINE FUNCS
    ###################################################################################################

    @classmethod
    def img_overlap(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.overlap.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        # Making overlap image
        with cluster_process(cls.heavy_cluster()):  # TODO:  is this faster without cluster?
            raw_arr = da.from_zarr(pfm.raw.val, chunks=configs.zarr_chunksize)
            overlap_arr = da_overlap(raw_arr, d=configs.overlap_depth)
            overlap_arr = disk_cache(overlap_arr, pfm.overlap.val)

    @classmethod
    def cellc1(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 1

        Top-hat filter (background subtraction)
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.bgrm.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # Reading input images
            overlap_arr = da.from_zarr(pfm.overlap.val)
            # Declaring processing instructions
            bgrm_arr = da.map_blocks(
                cls.cellc_funcs.tophat_filt,
                overlap_arr,
                configs.tophat_sigma,
            )
            # Computing and saving
            bgrm_arr = disk_cache(bgrm_arr, pfm.bgrm.val)

    @classmethod
    def cellc2(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 2

        Difference of Gaussians (edge detection)
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.dog.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # Reading input images
            bgrm_arr = da.from_zarr(pfm.bgrm.val)
            # Declaring processing instructions
            dog_arr = da.map_blocks(
                cls.cellc_funcs.dog_filt,
                bgrm_arr,
                configs.dog_sigma1,
                configs.dog_sigma2,
            )
            # Computing and saving
            dog_arr = disk_cache(dog_arr, pfm.dog.val)

    @classmethod
    def cellc3(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 3

        Gaussian subtraction with large sigma for adaptive thresholding
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.adaptv.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # Reading input images
            dog_arr = da.from_zarr(pfm.dog.val)
            # Declaring processing instructions
            adaptv_arr = da.map_blocks(
                cls.cellc_funcs.gauss_subt_filt,
                dog_arr,
                configs.large_gauss_sigma,
            )
            # Computing and saving
            adaptv_arr = disk_cache(adaptv_arr, pfm.adaptv.val)

    @classmethod
    def cellc4(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 4

        Currently, manual thresholding.
        Ideally, mean thresholding with standard deviation offset
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.threshd.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # # Visually inspect sd offset
            # t_p =adaptv_arr.sum() / (np.prod(adaptv_arr.shape) - (adaptv_arr == 0).sum())
            # t_p = t_p.compute()
            # Reading input images
            adaptv_arr = da.from_zarr(pfm.adaptv.val)
            # Declaring processing instructions
            threshd_arr = da.map_blocks(
                cls.cellc_funcs.manual_thresh,  # NOTE: previously CPU
                adaptv_arr,
                configs.threshd_value,
            )
            # Computing and saving
            threshd_arr = disk_cache(threshd_arr, pfm.threshd.val)

    @classmethod
    def cellc5(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 5

        Getting object sizes
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.threshd_volumes.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Reading input images
            threshd_arr = da.from_zarr(pfm.threshd.val)
            # Declaring processing instructions
            threshd_volumes_arr = da.map_blocks(
                cls.cellc_funcs.mask2volume,  # NOTE: previously CPU
                threshd_arr,
            )
            # Computing and saving
            threshd_volumes_arr = disk_cache(threshd_volumes_arr, pfm.threshd_volumes.val)

    @classmethod
    def cellc6(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 6

        Filter out large objects (likely outlines, not cells)
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.threshd_filt.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # Reading input images
            threshd_volumes_arr = da.from_zarr(pfm.threshd_volumes.val)
            # Declaring processing instructions
            threshd_filt_arr = da.map_blocks(
                cls.cellc_funcs.volume_filter,  # NOTE: previously CPU
                threshd_volumes_arr,
                configs.min_threshd_size,
                configs.max_threshd_size,
            )
            # Computing and saving
            threshd_filt_arr = disk_cache(threshd_filt_arr, pfm.threshd_filt.val)

    @classmethod
    def cellc7(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 7

        Get maxima of image masked by labels.
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.maxima.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # Reading input images
            overlap_arr = da.from_zarr(pfm.overlap.val)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt.val)
            # Declaring processing instructions
            maxima_arr = da.map_blocks(
                cls.cellc_funcs.get_local_maxima,
                overlap_arr,
                configs.maxima_sigma,
                threshd_filt_arr,
            )
            # Computing and saving
            maxima_arr = disk_cache(maxima_arr, pfm.maxima.val)

    @classmethod
    def cellc7b(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 7

        Convert maxima mask to uniquely labelled points.
        """
        # TODO: Check that the results of cellc10 and cellc7b, cellc8a, cellc10a are the same (df)
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.maxima_labels.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.gpu_cluster()):
            # Reading input images
            maxima_arr = da.from_zarr(pfm.maxima.val)
            # Declaring processing instructions
            maxima_labels_arr = da.map_blocks(
                cls.cellc_funcs.mask2label,
                maxima_arr,
            )
            maxima_labels_arr = disk_cache(maxima_labels_arr, pfm.maxima_labels.val)

    # NOTE: NOT NEEDED TO GET WSHED_LABELS AS ALL WE NEED IS WSBEED_VOLUMES FOR CELLC10b
    # @classmethod
    # def cellc8a(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
    #     """
    #     Cell counting pipeline - Step 8

    #     Watershed segmentation labels.
    #     """
    #     logger = init_logger_file()
    #     if not overwrite:
    #         for fp in (pfm.maxima_labels.val,):
    #             if os.path.exists(fp):
    #                 return logger.warning(file_exists_msg(fp))
    #     with cluster_process(cls.heavy_cluster()):
    #         # Reading input images
    #         overlap_arr = da.from_zarr(pfm.overlap.val)
    #         maxima_labels_arr = da.from_zarr(pfm.maxima_labels.val)
    #         threshd_filt_arr = da.from_zarr(pfm.threshd_filt.val)
    #         # Declaring processing instructions
    #         wshed_labels_arr = da.map_blocks(
    #             cls.cellc_funcs.wshed_segm,
    #             overlap_arr,
    #             maxima_labels_arr,
    #             threshd_filt_arr,
    #         )
    #         wshed_labels_arr = disk_cache(wshed_labels_arr, pfm.wshed_labels.val)

    # @classmethod
    # def cellc8b(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
    #     """
    #     Cell counting pipeline - Step 8

    #     Watershed segmentation volumes.
    #     """
    #     logger = init_logger_file()
    #     if not overwrite:
    #         for fp in (pfm.maxima_labels.val,):
    #             if os.path.exists(fp):
    #                 return logger.warning(file_exists_msg(fp))
    #     with cluster_process(cls.heavy_cluster()):
    #         # Reading input images
    #         wshed_labels_arr = da.from_zarr(pfm.wshed_labels.val)
    #         # Declaring processing instructions
    #         wshed_volumes_arr = da.map_blocks(
    #             cls.cellc_funcs.label2volume,
    #             wshed_labels_arr,
    #         )
    #         wshed_volumes_arr = disk_cache(wshed_volumes_arr, pfm.wshed_volumes.val)

    @classmethod
    def cellc8(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 8

        Watershed segmentation volumes.
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.wshed_volumes.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.heavy_cluster()):
            # n_workers=2
            # Reading input images
            overlap_arr = da.from_zarr(pfm.overlap.val)
            maxima_arr = da.from_zarr(pfm.maxima.val)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt.val)
            # Declaring processing instructions
            wshed_volumes_arr = da.map_blocks(
                CpuCellcFuncs.wshed_segm_volumes,
                overlap_arr,
                maxima_arr,
                threshd_filt_arr,
            )
            wshed_volumes_arr = disk_cache(wshed_volumes_arr, pfm.wshed_volumes.val)

    @classmethod
    def cellc9(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 9

        Filter out large watershed objects (again cell areas, not cells).
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.wshed_filt.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # Reading input images
            wshed_volumes_arr = da.from_zarr(pfm.wshed_volumes.val)
            # Declaring processing instructions
            wshed_filt_arr = da.map_blocks(
                cls.cellc_funcs.volume_filter,  # NOTE: previously CPU
                wshed_volumes_arr,
                configs.min_wshed_size,
                configs.max_wshed_size,
            )
            # Computing and saving
            wshed_filt_arr = disk_cache(wshed_filt_arr, pfm.wshed_filt.val)

    @classmethod
    def cellc10(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 11

        Calculate the maxima and watershed, save the cells.

        Basically a repeat of cellc8 and cellc9 but needs to be done to
        get the cell volumes in a table. Hence, don't run cellc8 and cellc9 if
        you don't want to view the cells visually (good for pipeline, not for tuning).
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.cells_raw_df.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.heavy_cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # Reading input images
            raw_arr = da.from_zarr(pfm.raw.val)
            overlap_arr = da.from_zarr(pfm.overlap.val)
            maxima_arr = da.from_zarr(pfm.maxima.val)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt.val)
            # Declaring processing instructions
            # Getting maxima coords and cell measures in table
            cells_df = block2coords(
                CpuCellcFuncs.get_cells,
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
            write_parquet(cells_df, pfm.cells_raw_df.val)

    @classmethod
    def cellc10b(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Alternative to cellc10.

        Uses raw, overlap, maxima_labels, wshed_filt (so wshed computation not run again).
        Also allows GPU processing.
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.cells_raw_df.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.heavy_cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # Reading input images
            raw_arr = da.from_zarr(pfm.raw.val)
            overlap_arr = da.from_zarr(pfm.overlap.val)
            maxima_labels_arr = da.from_zarr(pfm.maxima_labels.val)
            wshed_labels_arr = da.from_zarr(pfm.wshed_labels.val)
            wshed_filt_arr = da.from_zarr(pfm.wshed_filt.val)
            # Declaring processing instructions
            # Getting maxima coords and cell measures in table
            cells_df = block2coords(
                CpuCellcFuncs.get_cells_b,
                raw_arr,
                overlap_arr,
                maxima_labels_arr,
                wshed_labels_arr,
                wshed_filt_arr,
                configs.overlap_depth,
            )
            # Converting from dask to pandas
            cells_df = cells_df.compute()
            # Computing and saving as parquet
            write_parquet(cells_df, pfm.cells_raw_df.val)

    @classmethod
    def cellc_coords_only(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Get maxima coords.
        Very basic but faster version of cellc11_pipeline get_cells.
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.maxima_df.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.gpu_cluster()):
            # Read filtered and maxima images (trimmed to orig space)
            maxima_final_arr = da.from_zarr(pfm.maxima_final.val)
            # Declaring processing instructions
            # Storing coords of each maxima in df
            coords_df = block2coords(
                cls.cellc_funcs.get_coords,
                maxima_final_arr,
            )
            # Converting from dask to pandas
            coords_df = coords_df.compute()
            # Computing and saving as parquet
            write_parquet(coords_df, pfm.maxima_df.val)

    ###################################################################################################
    # CELL COUNT REALIGNMENT TO REFERENCE AND AGGREGATION PIPELINE FUNCS
    ###################################################################################################

    @classmethod
    def transform_coords(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        `in_id` and `out_id` are either maxima or region

        NOTE: saves the cells_trfm dataframe as pandas parquet.
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.cells_trfm_df.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Getting configs
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        with cluster_process(cls.busy_cluster()):
            # Setting output key (in the form "<maxima/region>_trfm_df")
            # Getting cell coords
            cells_df = pd.read_parquet(pfm.cells_raw_df.val)
            # Sanitising (removing smb columns)
            cells_df = sanitise_smb_df(cells_df)
            # Taking only Coords.Z.value, Coords.Y.value, Coords.X.value coord columns
            cells_df = cells_df[enum2list(Coords)]
            # Scaling to resampled rough space
            # NOTE: this downsampling uses slicing so must be computed differently
            cells_df = cells_df / np.array((configs.z_rough, configs.y_rough, configs.x_rough))
            # Scaling to resampled space
            cells_df = cells_df * np.array((configs.z_fine, configs.y_fine, configs.x_fine))
            # Trimming/offsetting to sliced space
            cells_df = cells_df - np.array([s[0] or 0 for s in (configs.z_trim, configs.y_trim, configs.x_trim)])
            # Converting back to DataFrame
            cells_df = pd.DataFrame(cells_df, columns=enum2list(Coords))

            cells_trfm_df = ElastixFuncs.transformation_coords(cells_df, pfm.ref.val, pfm.regresult.val)
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_df = dd.from_pandas(cells_df, npartitions=1)
            # Fitting resampled space to atlas image with Transformix (from Elastix registration step)
            # cells_df = cells_df.repartition(
            #     npartitions=int(np.ceil(cells_df.shape[0].compute() / ROWSPPART))
            # )
            # cells_df = cells_df.map_partitions(
            #     ElastixFuncs.transformation_coords, pfm.ref.val, pfm.regresult.val
            # )
            write_parquet(cells_trfm_df, pfm.cells_trfm_df.val)

    @classmethod
    def cell_mapping(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Using the transformed cell coordinates, get the region ID and name for each cell
        corresponding to the reference atlas.

        NOTE: saves the cells dataframe as pandas parquet.
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.cells_df.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Getting region for each detected cell (i.e. row) in cells_df
        with cluster_process(cls.busy_cluster()):
            # Reading cells_raw and cells_trfm dataframes
            cells_df = pd.read_parquet(pfm.cells_raw_df.val)
            coords_trfm = pd.read_parquet(pfm.cells_trfm_df.val)
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
            annot_arr = tifffile.imread(pfm.annot.val)
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
            annot_df = MapFuncs.annot_dict2df(read_json(pfm.map.val))
            # Getting the annotation name for every cell (zyx coord)
            cells_df = MapFuncs.df_map_ids(cells_df, annot_df)
            # Saving to disk
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_df = dd.from_pandas(cells_df)
            write_parquet(cells_df, pfm.cells_df.val)

    @classmethod
    def group_cells(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Grouping cells by region name and aggregating total cell volume
        and cell count for each region.

        NOTE: saves the cells_agg dataframe as pandas parquet.
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.cells_agg_df.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Making cells_agg_df
        with cluster_process(cls.busy_cluster()):
            # Reading cells dataframe
            cells_df = pd.read_parquet(pfm.cells_df.val)
            # Sanitising (removing smb columns)
            cells_df = sanitise_smb_df(cells_df)
            # Grouping cells by region name and aggregating on given mappings
            cells_agg_df = cells_df.groupby(AnnotColumns.ID.value).agg(CELL_AGG_MAPPINGS)
            cells_agg_df.columns = list(CELL_AGG_MAPPINGS.keys())
            # Reading annotation mappings dataframe
            # Making df of region names and their parent region names
            annot_df = MapFuncs.annot_dict2df(read_json(pfm.map.val))
            # Combining (summing) the cells_agg_df values for parent regions using the annot_df
            cells_agg_df = MapFuncs.combine_nested_regions(cells_agg_df, annot_df)
            # Calculating integrated average intensity (sum_intensity / volume)
            cells_agg_df[CellColumns.IOV.value] = (
                cells_agg_df[CellColumns.SUM_INTENSITY.value] / cells_agg_df[CellColumns.VOLUME.value]
            )
            # Selecting and ordering relevant columns
            cells_agg_df = cells_agg_df[[*ANNOT_COLUMNS_FINAL, *enum2list(CellColumns)]]
            # Saving to disk
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_agg = dd.from_pandas(cells_agg)
            write_parquet(cells_agg_df, pfm.cells_agg_df.val)

    @classmethod
    def cells2csv(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.cells_agg_csv.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        # Reading cells dataframe
        cells_agg_df = pd.read_parquet(pfm.cells_agg_df.val)
        # Sanitising (removing smb columns)
        cells_agg_df = sanitise_smb_df(cells_agg_df)
        # Saving to csv
        cells_agg_df.to_csv(pfm.cells_agg_csv.val)

    ###################################################################################################
    # ALL PIPELINE FUNCTION
    ###################################################################################################

    @classmethod
    def run_pipeline(cls, in_fp: str, proj_dir: str, overwrite: bool = False, **kwargs) -> None:
        """
        Running all pipelines in order.
        """
        # Getting PFMs
        pfm = cls.get_pfm(proj_dir)
        pfm_tuning = cls.get_pfm_tuning(proj_dir)
        # Updating project configs
        cls.update_configs(pfm, **kwargs)
        # Running all pipelines in order
        # tiff to zarr
        cls.tiff2zarr(pfm, in_fp, overwrite=overwrite)
        # Registration
        cls.reg_ref_prepare(pfm, overwrite=overwrite)
        cls.reg_img_rough(pfm, overwrite=overwrite)
        cls.reg_img_fine(pfm, overwrite=overwrite)
        cls.reg_img_trim(pfm, overwrite=overwrite)
        cls.reg_img_bound(pfm, overwrite=overwrite)
        cls.reg_elastix(pfm, overwrite=overwrite)
        # Coverage mask
        cls.make_mask(pfm, overwrite=overwrite)
        # Cell counting
        cls.make_tuning_arr(pfm, overwrite=overwrite)
        for pfm_i in [
            pfm_tuning,
            pfm,
        ]:
            cls.img_overlap(pfm_i, overwrite=overwrite)
            cls.cellc1(pfm_i, overwrite=overwrite)
            cls.cellc2(pfm_i, overwrite=overwrite)
            cls.cellc3(pfm_i, overwrite=overwrite)
            cls.cellc4(pfm_i, overwrite=overwrite)
            cls.cellc5(pfm_i, overwrite=overwrite)
            cls.cellc6(pfm_i, overwrite=overwrite)
            cls.cellc7(pfm_i, overwrite=overwrite)
            cls.cellc8(pfm_i, overwrite=overwrite)
            cls.cellc9(pfm_i, overwrite=overwrite)
            cls.cellc10(pfm_i, overwrite=overwrite)
        # Cell mapping
        cls.transform_coords(pfm, overwrite=overwrite)
        cls.cell_mapping(pfm, overwrite=overwrite)
        cls.group_cells(pfm, overwrite=overwrite)
        cls.cells2csv(pfm, overwrite=overwrite)
