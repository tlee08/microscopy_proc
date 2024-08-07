from microscopy_proc.pipelines.cellc_pipeline import (
    img_overlap_pipeline,
    img_proc_pipeline,
    img_to_cells_pipeline,
)
from microscopy_proc.pipelines.make_zarr import tiff_to_zarr
from microscopy_proc.pipelines.reg_pipeline import (
    prepare_img_fine,
    prepare_img_rough,
    prepare_img_trim,
    prepare_ref,
    registration,
)
from microscopy_proc.utils.proj_org_utils import (
    get_proj_fp_dict,
    get_ref_fp_dict,
    make_proj_dirs,
)

# from prefect import flow


if __name__ == "__main__":
    # Filenames
    # proj_dir = "/home/linux1/Desktop/A-1-1/cellcount"
    # in_fp = "/home/linux1/Desktop/A-1-1/cropped abcd_larger.tif"
    in_fp = "/home/linux1/Desktop/A-1-1/example"
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"

    ref_fp_dict = get_ref_fp_dict(atlas_rsc_dir)
    proj_fp_dict = get_proj_fp_dict(proj_dir)
    make_proj_dirs(proj_dir)

    # Making zarr from tiff file(s)
    tiff_to_zarr(in_fp, proj_fp_dict["raw"])

    # Preparing reference images
    prepare_ref(
        ref_fp_dict=ref_fp_dict,
        proj_fp_dict=proj_fp_dict,
        orient_ls=(2, 3, 1),
        z_trim=slice(None, None),
        y_trim=slice(None, None),
        x_trim=slice(None, None),
    )

    # Preparing image itself
    prepare_img_rough(
        proj_fp_dict,
        z_rough=3,
        y_rough=6,
        x_rough=6,
    )
    prepare_img_fine(
        proj_fp_dict,
        z_fine=1,
        y_fine=0.6,
        x_fine=0.6,
    )
    prepare_img_trim(
        proj_fp_dict,
        z_trim=slice(None, -5),
        y_trim=slice(80, -75),
        x_trim=slice(None, None),
    )

    # Running Elastix registration
    registration(
        fixed_img_fp=proj_fp_dict["trimmed"],
        moving_img_fp=proj_fp_dict["ref"],
        output_img_fp=proj_fp_dict["regresult"],
        affine_fp=proj_fp_dict["affine"],
        bspline_fp=proj_fp_dict["bspline"],
    )

    img_overlap_pipeline(proj_fp_dict)

    img_proc_pipeline(
        proj_fp_dict=proj_fp_dict,
        tophat_sigma=10,
        dog_sigma1=1,
        dog_sigma2=4,
        gauss_sigma=101,
        thresh_p=32,
        min_size=100,
        max_size=10000,
        maxima_sigma=10,
    )

    img_to_cells_pipeline(proj_fp_dict)
