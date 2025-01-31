import os

import dask.array as da
import pandas as pd
import tifffile
from dask.distributed import LocalCluster

from microscopy_proc.funcs.viewer_funcs import ViewerFuncs
from microscopy_proc.funcs.visual_check_funcs_dask import VisualCheckFuncsDask
from microscopy_proc.funcs.visual_check_funcs_tiff import VisualCheckFuncsTiff
from microscopy_proc.pipeline.pipeline import Pipeline
from microscopy_proc.utils.config_params_model import ConfigParamsModel
from microscopy_proc.utils.dask_utils import (
    cluster_process,
    da_trim,
    disk_cache,
)
from microscopy_proc.utils.diagnostics_utils import file_exists_msg
from microscopy_proc.utils.logging_utils import init_logger_file
from microscopy_proc.utils.proj_org_utils import (
    ProjFpModelBase,
)


class VisualCheck:
    # Clusters
    # busy (many workers - carrying low RAM computations)
    n_workers = 6
    threads_per_worker = 2

    @classmethod
    def cluster(cls):
        return LocalCluster(n_workers=cls.n_workers, threads_per_worker=cls.threads_per_worker)

    ###################################################################################################
    # VISUAL CHECKS FROM DF POINTS
    ###################################################################################################

    @classmethod
    def cellc_trim_to_final(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        """
        Cell counting pipeline - Step 10

        Trimming filtered regions overlaps to make:
        - Trimmed maxima image
        - Trimmed threshold image
        - Trimmed watershed image
        """
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.maxima_final.val, pfm.threshd_final.val, pfm.wshed_final.val):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            # Reading input images
            maxima_arr = da.from_zarr(pfm.maxima.val)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt.val)
            wshed_volumes_arr = da.from_zarr(pfm.wshed_volumes.val)
            # Declaring processing instructions
            maxima_final_arr = da_trim(maxima_arr, d=configs.overlap_depth)
            threshd_final_arr = da_trim(threshd_filt_arr, d=configs.overlap_depth)
            wshed_final_arr = da_trim(wshed_volumes_arr, d=configs.overlap_depth)
            # Computing and saving
            maxima_final_arr = disk_cache(maxima_final_arr, pfm.maxima_final.val)
            threshd_final_arr = disk_cache(threshd_final_arr, pfm.threshd_final.val)
            wshed_final_arr = disk_cache(wshed_final_arr, pfm.wshed_final.val)

    @classmethod
    def coords2points_raw(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.points_raw.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.cluster()):
            VisualCheckFuncsDask.coords2points(
                coords=pd.read_parquet(pfm.cells_raw_df.val),
                shape=da.from_zarr(pfm.raw.val).shape,
                out_fp=pfm.points_raw.val,
            )

    @classmethod
    def coords2heatmap_raw(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.heatmap_raw.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.cluster()):
            configs = ConfigParamsModel.read_fp(pfm.config_params.val)
            VisualCheckFuncsDask.coords2heatmap(
                coords=pd.read_parquet(pfm.cells_raw_df.val),
                shape=da.from_zarr(pfm.raw.val).shape,
                out_fp=pfm.heatmap_raw.val,
                radius=configs.heatmap_raw_radius,
            )

    @classmethod
    def coords2points_trfm(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.points_trfm.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        VisualCheckFuncsTiff.coords2points(
            coords=pd.read_parquet(pfm.cells_trfm_df.val),
            shape=tifffile.imread(pfm.ref.val).shape,
            out_fp=pfm.points_trfm.val,
        )

    @classmethod
    def coords2heatmap_trfm(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.heatmap_trfm.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        VisualCheckFuncsTiff.coords2heatmap(
            coords=pd.read_parquet(pfm.cells_trfm_df.val),
            shape=tifffile.imread(pfm.ref.val).shape,
            out_fp=pfm.heatmap_trfm.val,
            radius=configs.heatmap_trfm_radius,
        )

    ###################################################################################################
    # COMBINING/MERGING ARRAYS IN RGB LAYERS
    ###################################################################################################

    @classmethod
    def combine_reg(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.comb_reg.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.trimmed.val, pfm.bounded.val, pfm.regresult.val),
            fp_out=pfm.comb_reg.val,
        )

    @classmethod
    def combine_cellc(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.comb_cellc.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        configs = ConfigParamsModel.read_fp(pfm.config_params.val)
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.raw.val, pfm.threshd_final.val, pfm.wshed_final.val),
            fp_out=pfm.comb_cellc.val,
            trimmer=(
                slice(*configs.combine_cellc_z_trim),
                slice(*configs.combine_cellc_y_trim),
                slice(*configs.combine_cellc_x_trim),
            ),
        )

    @classmethod
    def combine_trfm_points(cls, pfm: ProjFpModelBase, overwrite: bool = False) -> None:
        logger = init_logger_file()
        if not overwrite:
            for fp in (pfm.comb_points.val,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.ref.val, pfm.annot.val, pfm.heatmap_trfm.val),
            # 2nd regresult means the combining works in ImageJ
            fp_out=pfm.comb_points.val,
        )

    ###################################################################################################
    # ALL PIPELINE FUNCTION
    ###################################################################################################

    @classmethod
    def run_make_visual_checks(cls, proj_dir: str, overwrite: bool = False) -> None:
        """
        Running all visual check pipelines in order.
        """
        # Getting PFMs
        pfm = Pipeline.get_pfm(proj_dir)
        pfm_tuning = Pipeline.get_pfm_tuning(proj_dir)
        # Registration visual check
        cls.combine_reg(pfm, overwrite=overwrite)
        # Cell counting visual checks
        for pfm_i in [
            pfm_tuning,
            pfm,
        ]:
            cls.cellc_trim_to_final(pfm_i, overwrite=overwrite)
            cls.coords2points_raw(pfm_i, overwrite=overwrite)
            cls.combine_cellc(pfm_i, overwrite=overwrite)
        # Transformed space visual checks
        cls.coords2points_trfm(pfm, overwrite=overwrite)
        cls.coords2heatmap_trfm(pfm, overwrite=overwrite)
        cls.combine_trfm_points(pfm, overwrite=overwrite)
