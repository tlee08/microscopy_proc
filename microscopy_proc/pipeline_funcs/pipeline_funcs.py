import functools
import logging
import os
import re
import shutil
from typing import Callable

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
from microscopy_proc.funcs.map_funcs import (
    annot_dict2df,
    combine_nested_regions,
    df_map_ids,
)
from microscopy_proc.funcs.mask_funcs import (
    fill_outline,
    make_outline,
    mask2region_counts,
)
from microscopy_proc.funcs.reg_funcs import (
    downsmpl_fine,
    downsmpl_rough,
    reorient,
)
from microscopy_proc.funcs.tiff2zarr_funcs import btiff2zarr, tiffs2zarr
from microscopy_proc.funcs.viewer_funcs import combine_arrs
from microscopy_proc.funcs.visual_check_funcs_dask import (
    coords2heatmap as coords2heatmap_dask,
)
from microscopy_proc.funcs.visual_check_funcs_dask import (
    coords2points as coords2points_dask,
)
from microscopy_proc.funcs.visual_check_funcs_tiff import (
    coords2heatmap as coords2heatmap_tiff,
)
from microscopy_proc.funcs.visual_check_funcs_tiff import (
    coords2points as coords2points_tiff,
)
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import (
    block2coords,
    cluster_proc_contxt,
    da_overlap,
    da_trim,
    disk_cache,
)
from microscopy_proc.utils.io_utils import read_json, sanitise_smb_df
from microscopy_proc.utils.misc_utils import enum2list, import_extra_error_func
from microscopy_proc.utils.proj_org_utils import (
    ProjFpModel,
    RefFpModel,
)

# Optional dependency: gpu
if GPU_ENABLED:
    from dask_cuda import LocalCUDACluster

    from microscopy_proc.funcs.gpu_cellc_funcs import GpuCellcFuncs as Gf
else:
    LocalCUDACluster = LocalCluster
    Gf = Cf
    print(
        "Warning GPU functionality not installed.\n"
        "Using CPU functionality instead (much slower).\n"
        'Can install with `pip install "microscopy_proc[gpu]"`'
    )
# Optional dependency: elastix
if ELASTIX_ENABLED:
    from microscopy_proc.funcs.elastix_funcs import registration, transformation_coords
else:
    registration = import_extra_error_func("elastix")
    transformation_coords = import_extra_error_func("elastix")


###################################################################################################
# OVERWRITE DECORATOR
###################################################################################################


def overwrite_check_decorator(pfm_fp_ls: tuple[str, ...] = tuple()):
    """
    Decorator to check overwrite and will
    not run the function if the fpm attribute in
    `pfm_fp_ls` is a filepath that already exsits.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Getting pfm arg
            pfm: ProjFpModel = kwargs.get("pfm", args[0])
            assert isinstance(pfm, ProjFpModel)
            # Getting overwrite arg
            overwrite = kwargs.get("overwrite", False)
            assert isinstance(overwrite, bool)
            # If overwrite is False, check if output file exists
            if not overwrite:
                # Iterating through attributes in pfm_fp_ls
                for pfm_fp in pfm_fp_ls:
                    # If attribute exists and is a filepath, then don't run func
                    if os.path.exists(getattr(pfm, pfm_fp)):
                        logging.info(
                            f"Skipping {func.__name__} as {pfm_fp} already exists."
                        )
                        return
            # Running func
            return func(*args, **kwargs)

        return wrapper

    return decorator


class PipelineFuncs:
    ###################################################################################################
    # CONVERT TIFF TO ZARR FUNCS
    ###################################################################################################

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("raw",))
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
        # Getting configs
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        # Making zarr from tiff file(s)
        with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=6)):
            if os.path.isdir(in_fp):
                # If in_fp is a directory, make zarr from the tiff file stack in directory
                tiffs2zarr(
                    in_fp_ls=natsorted(
                        [
                            os.path.join(in_fp, i)
                            for i in os.listdir(in_fp)
                            if re.search(r".tif$", i)
                        ]
                    ),
                    out_fp=pfm.raw,
                    chunks=configs.zarr_chunksize,
                )
            elif os.path.isfile(in_fp):
                # If in_fp is a file, make zarr from the btiff file
                btiff2zarr(
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
    @overwrite_check_decorator(pfm_fp_ls=("ref", "annot", "map", "affine", "bspline"))
    def ref_prepare(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        # Not overwriting if specified and output file exists
        # TODO
        # Getting configs
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        # Making ref_fp_model of original atlas images filepaths
        rfm = RefFpModel.get_ref_fp_model(
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
            arr = reorient(arr, configs.ref_orient_ls)
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
    @overwrite_check_decorator(pfm_fp_ls=("downsmpl1",))
    def img_rough(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        # Getting configs
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        with cluster_proc_contxt(LocalCluster()):
            # Reading
            raw_arr = da.from_zarr(pfm.raw)
            # Rough downsample
            downsmpl1_arr = downsmpl_rough(
                raw_arr, configs.z_rough, configs.y_rough, configs.x_rough
            )
            # Computing (from dask array)
            downsmpl1_arr = downsmpl1_arr.compute()
            # Saving
            tifffile.imwrite(pfm.downsmpl1, downsmpl1_arr)

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("downsmpl2",))
    def img_fine(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        # Getting configs
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        # Reading
        downsmpl1_arr = tifffile.imread(pfm.downsmpl1)
        # Fine downsample
        downsmpl2_arr = downsmpl_fine(
            downsmpl1_arr, configs.z_fine, configs.y_fine, configs.x_fine
        )
        # Saving
        tifffile.imwrite(pfm.downsmpl2, downsmpl2_arr)

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("trimmed",))
    def img_trim(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        # Getting configs
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
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
    @overwrite_check_decorator(pfm_fp_ls=("regresult",))
    def elastix_registration(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        # Running Elastix registration
        registration(
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
    @overwrite_check_decorator(pfm_fp_ls=("mask", "outline", "mask_reg", "mask_df"))
    def make_mask(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Makes mask of actual image in reference space.
        Also stores # and proportion of existent voxels
        for each region.
        """
        # Getting configs
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
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
        outline_df = make_outline(mask_arr)
        # Transformix on coords
        outline_df[[Coords.Z.value, Coords.Y.value, Coords.X.value]] = (
            transformation_coords(
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
        coords2points_tiff(outline_df[outline_df.is_in == 1], s, pfm.outline)
        in_arr = tifffile.imread(pfm.outline)
        coords2points_tiff(outline_df[outline_df.is_in == 0], s, pfm.outline)
        out_arr = tifffile.imread(pfm.outline)
        tifffile.imwrite(pfm.outline, in_arr + out_arr * 2)

        # Fill in outline to recreate mask (not perfect)
        mask_reg_arr = fill_outline(outline_df, s)
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
        rfm = RefFpModel.get_ref_fp_model(
            configs.atlas_dir,
            configs.ref_version,
            configs.annot_version,
            configs.map_version,
        )
        # Reading original annot
        annot_orig_arr = tifffile.imread(rfm.annot)
        # Getting the annotation name for every cell (zyx coord)
        mask_df = pd.merge(
            left=mask2region_counts(np.full(annot_orig_arr.shape, 1), annot_orig_arr),
            right=mask2region_counts(mask_reg_arr, annot_arr),
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("_annot", "_mask"),
        ).fillna(0)
        # Reading annotation mappings json
        annot_df = annot_dict2df(read_json(pfm.map))
        # Combining (summing) the mask_df volumes for parent regions using the annot_df
        mask_df = combine_nested_regions(mask_df, annot_df)
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
    # CELL COUNTING PIPELINE FUNCS
    ###################################################################################################

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("overlap",))
    def img_overlap(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        # Getting configs
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        # Making overlap image
        with cluster_proc_contxt(LocalCluster(n_workers=1, threads_per_worker=4)):
            raw_arr = da.from_zarr(pfm.raw, chunks=configs.zarr_chunksize)
            overlap_arr = da_overlap(raw_arr, d=configs.overlap_depth)
            overlap_arr = disk_cache(overlap_arr, pfm.overlap)

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("bgrm",))
    def cellc1(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 1

        Top-hat filter (background subtraction)
        """
        # Making Dask cluster
        with cluster_proc_contxt(LocalCUDACluster()):
            # Getting configs
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
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
    @overwrite_check_decorator(pfm_fp_ls=("dog",))
    def cellc2(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 2

        Difference of Gaussians (edge detection)
        """
        # Making Dask cluster
        with cluster_proc_contxt(LocalCUDACluster()):
            # Getting configs
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
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
    @overwrite_check_decorator(pfm_fp_ls=("adaptv",))
    def cellc3(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 3

        Gaussian subtraction with large sigma for adaptive thresholding
        """
        # Making Dask cluster
        with cluster_proc_contxt(LocalCUDACluster()):
            # Getting configs
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
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
    @overwrite_check_decorator(pfm_fp_ls=("threshd",))
    def cellc4(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 4

        Currently, manual thresholding.
        Ideally, mean thresholding with standard deviation offset
        """
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster()):
            # Getting configs
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
            # # Visually inspect sd offset
            # t_p =adaptv_arr.sum() / (np.prod(adaptv_arr.shape) - (adaptv_arr == 0).sum())
            # t_p = t_p.compute()
            # logging.debug(t_p)
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
    @overwrite_check_decorator(pfm_fp_ls=("threshd_volumes",))
    def cellc5(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 5

        Getting object sizes
        """
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
    @overwrite_check_decorator(pfm_fp_ls=("threshd_filt",))
    def cellc6(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 6

        Filter out large objects (likely outlines, not cells)
        """
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster()):
            # Getting configs
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
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
    @overwrite_check_decorator(pfm_fp_ls=("maxima",))
    def cellc7(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 7

        Get maxima of image masked by labels.
        """
        # Making Dask cluster
        with cluster_proc_contxt(LocalCUDACluster()):
            # Getting configs
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
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
    @overwrite_check_decorator(pfm_fp_ls=("wshed_volumes",))
    def cellc8(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 8

        Watershed segmentation volumes.
        """
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
    @overwrite_check_decorator(pfm_fp_ls=("wshed_filt",))
    def cellc9(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 9

        Filter out large watershed objects (again likely outlines, not cells).
        """
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster()):
            # Getting configs
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
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
    @overwrite_check_decorator(
        pfm_fp_ls=("maxima_final", "threshd_final", "wshed_final")
    )
    def cellc10(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 10

        Trimming filtered regions overlaps to make:
        - Trimmed maxima image
        - Trimmed threshold image
        - Trimmed watershed image
        """
        # Making Dask cluster
        with cluster_proc_contxt(LocalCluster()):
            # Getting configs
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
            # Reading input images
            maxima_arr = da.from_zarr(pfm.maxima)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt)
            wshed_volumes_arr = da.from_zarr(pfm.wshed_volumes)
            # Declaring processing instructions
            maxima_final_arr = da_trim(maxima_arr, d=configs.overlap_depth)
            threshd_final_arr = da_trim(threshd_filt_arr, d=configs.overlap_depth)
            wshed_final_arr = da_trim(wshed_volumes_arr, d=configs.overlap_depth)
            # Computing and saving
            disk_cache(maxima_final_arr, pfm.maxima_final)
            disk_cache(threshd_final_arr, pfm.threshd_final)
            disk_cache(wshed_final_arr, pfm.wshed_final)

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("cells_raw_df",))
    def cellc11(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 11

        From maxima and watershed, save the cells.
        """
        with cluster_proc_contxt(LocalCluster(n_workers=2, threads_per_worker=1)):
            # n_workers=2
            # Getting configs
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
            # Reading input images
            raw_arr = da.from_zarr(pfm.raw)
            overlap_arr = da.from_zarr(pfm.overlap)
            maxima_arr = da.from_zarr(pfm.maxima)
            wshed_filt_arr = da.from_zarr(pfm.wshed_filt)
            # Declaring processing instructions
            # Getting maxima coords and cell measures in table
            cells_df = block2coords(
                Cf.get_cells,
                raw_arr,
                overlap_arr,
                maxima_arr,
                wshed_filt_arr,
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
    @overwrite_check_decorator(pfm_fp_ls=("maxima_df",))
    def cellc_coords_only(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Get maxima coords.
        Very basic but faster version of cellc11_pipeline get_cells.
        """
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
    @overwrite_check_decorator(pfm_fp_ls=("cells_trfm_df",))
    def transform_coords(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        `in_id` and `out_id` are either maxima or region

        NOTE: saves the cells_trfm dataframe as pandas parquet.
        """
        # Getting configs
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
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

            cells_trfm_df = transformation_coords(cells_df, pfm.ref, pfm.regresult)
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_df = dd.from_pandas(cells_df, npartitions=1)
            # Fitting resampled space to atlas image with Transformix (from Elastix registration step)
            # cells_df = cells_df.repartition(
            #     npartitions=int(np.ceil(cells_df.shape[0].compute() / ROWSPPART))
            # )
            # cells_df = cells_df.map_partitions(
            #     transformation_coords, pfm.ref, pfm.regresult
            # )
            cells_trfm_df.to_parquet(pfm.cells_trfm_df)

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("cells_df",))
    def cell_mapping(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Using the transformed cell coordinates, get the region ID and name for each cell
        corresponding to the reference atlas.

        NOTE: saves the cells dataframe as pandas parquet.
        """
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
            annot_df = annot_dict2df(read_json(pfm.map))
            # Getting the annotation name for every cell (zyx coord)
            cells_df = df_map_ids(cells_df, annot_df)
            # Saving to disk
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_df = dd.from_pandas(cells_df)
            cells_df.to_parquet(pfm.cells_df)

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("cells_agg_df",))
    def group_cells(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Grouping cells by region name and aggregating total cell volume
        and cell count for each region.

        NOTE: saves the cells_agg dataframe as pandas parquet.
        """
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
            annot_df = annot_dict2df(read_json(pfm.map))
            # Combining (summing) the cells_agg_df values for parent regions using the annot_df
            cells_agg_df = combine_nested_regions(cells_agg_df, annot_df)
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
    @overwrite_check_decorator(pfm_fp_ls=("cells_agg_csv",))
    def cells2csv(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
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
    @overwrite_check_decorator(pfm_fp_ls=("points_raw",))
    def coords2points_raw(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        with cluster_proc_contxt(LocalCluster()):
            coords2points_dask(
                coords=pd.read_parquet(pfm.cells_raw_df),
                shape=da.from_zarr(pfm.raw).shape,
                out_fp=pfm.points_raw,
            )

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("heatmap_raw",))
    def coords2heatmap_raw(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        with cluster_proc_contxt(LocalCluster()):
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
            coords2heatmap_dask(
                coords=pd.read_parquet(pfm.cells_raw_df),
                shape=da.from_zarr(pfm.raw).shape,
                out_fp=pfm.heatmap_raw,
                radius=configs.heatmap_raw_radius,
            )

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("points_trfm",))
    def coords2points_trfm(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        with cluster_proc_contxt(LocalCluster()):
            coords2points_tiff(
                coords=pd.read_parquet(pfm.cells_trfm_df),
                shape=tifffile.imread(pfm.ref).shape,
                out_fp=pfm.points_trfm,
            )

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("heatmap_trfm",))
    def coords2heatmap_trfm(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        with cluster_proc_contxt(LocalCluster()):
            configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
            coords2heatmap_tiff(
                coords=pd.read_parquet(pfm.cells_trfm_df),
                shape=tifffile.imread(pfm.ref).shape,
                out_fp=pfm.heatmap_trfm,
                radius=configs.heatmap_trfm_radius,
            )

    ###################################################################################################
    # COMBINING/MERGING ARRAYS IN RGB LAYERS
    ###################################################################################################

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("combined_reg",))
    def combine_reg(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        combine_arrs(
            fp_in_ls=(pfm.trimmed, pfm.regresult, pfm.regresult),
            # 2nd regresult means the combining works in ImageJ
            fp_out=pfm.combined_reg,
        )

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("combined_cellc",))
    def combine_cellc(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        configs = ConfigParamsModel.model_validate(read_json(pfm.config_params))
        combine_arrs(
            fp_in_ls=(pfm.raw, pfm.threshd_final, pfm.wshed_final),
            fp_out=pfm.combined_cellc,
            trimmer=(
                slice(*configs.z_trim),
                slice(*configs.y_trim),
                slice(*configs.x_trim),
            ),
        )

    @classmethod
    @overwrite_check_decorator(pfm_fp_ls=("combined_points",))
    def combine_points(cls, pfm: ProjFpModel, overwrite: bool = False) -> None:
        combine_arrs(
            fp_in_ls=(pfm.ref, pfm.annot, pfm.heatmap_trfm),
            # 2nd regresult means the combining works in ImageJ
            fp_out=pfm.combined_points,
        )

    ###################################################################################################
    # ALL PIPELINE FUNCTION
    ###################################################################################################

    @classmethod
    def all_pipeline(in_fp: str, pfm: ProjFpModel, overwrite: bool = False) -> None:
        """
        Running all pipelines in order.
        """
        # Running all pipelines in order
        PipelineFuncs.tiff2zarr(pfm, in_fp, overwrite=overwrite)
        PipelineFuncs.ref_prepare(pfm, overwrite=overwrite)
        PipelineFuncs.img_rough(pfm, overwrite=overwrite)
        PipelineFuncs.img_fine(pfm, overwrite=overwrite)
        PipelineFuncs.img_trim(pfm, overwrite=overwrite)
        PipelineFuncs.elastix_registration(pfm, overwrite=overwrite)
        PipelineFuncs.make_mask(pfm, overwrite=overwrite)
        PipelineFuncs.img_overlap(pfm, overwrite=overwrite)
        PipelineFuncs.cellc1(pfm, overwrite=overwrite)
        PipelineFuncs.cellc2(pfm, overwrite=overwrite)
        PipelineFuncs.cellc3(pfm, overwrite=overwrite)
        PipelineFuncs.cellc4(pfm, overwrite=overwrite)
        PipelineFuncs.cellc5(pfm, overwrite=overwrite)
        PipelineFuncs.cellc6(pfm, overwrite=overwrite)
        PipelineFuncs.cellc7(pfm, overwrite=overwrite)
        PipelineFuncs.cellc8(pfm, overwrite=overwrite)
        PipelineFuncs.cellc9(pfm, overwrite=overwrite)
        PipelineFuncs.cellc10(pfm, overwrite=overwrite)
        PipelineFuncs.cellc11(pfm, overwrite=overwrite)
        PipelineFuncs.cellc_coords_only(pfm, overwrite=overwrite)
        PipelineFuncs.transform_coords(pfm, overwrite=overwrite)
        PipelineFuncs.cell_mapping(pfm, overwrite=overwrite)
        PipelineFuncs.group_cells(pfm, overwrite=overwrite)
        PipelineFuncs.cells2csv(pfm, overwrite=overwrite)
        PipelineFuncs.coords2points_raw(pfm, overwrite=overwrite)
        PipelineFuncs.coords2heatmap_raw(pfm, overwrite=overwrite)
        PipelineFuncs.coords2points_trfm(pfm, overwrite=overwrite)
        PipelineFuncs.coords2heatmap_trfm(pfm, overwrite=overwrite)
        PipelineFuncs.combine_reg(pfm, overwrite=overwrite)
        PipelineFuncs.combine_cellc(pfm, overwrite=overwrite)
        PipelineFuncs.combine_points(pfm, overwrite=overwrite)
