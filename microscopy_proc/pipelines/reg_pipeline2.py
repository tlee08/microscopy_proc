import os

from microscopy_proc.utils.elastix_utils import registration

# logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # # Filenames
    # in_fp = "/home/linux1/Desktop/A-1-1/large_cellcount/raw.zarr"
    proj_dir = "/home/linux1/Desktop/A-1-1/large_cellcount"

    ################## CHANGE BELOW ##################
    # ROOT AND PROJECT DIRECTORIES
    atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"
    # Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/

    # NAMES OF SUBDIRECTORIES FOR PROJECT FILES
    reg_sdir = "registration"
    cellc_subdir = "cellcount"
    analysis_sdir = "analysis"

    # NAMES OF ATLAS FILES USED
    ref_v = "average_template_25"  # "average_template_25", "ara_nissl_25"
    annot_v = "ccf_2016_25"  # "ccf_2017_25", "ccf_2016_25", "ccf_2015_25"
    region_map_v = "ABA_annotations"  # "ABA_annotations", "CM_annotations"
    ################## CHANGE ABOVE ##################

    # ATLAS FILES ORIGIN
    ref_orig_fp = os.path.join(atlas_rsc_dir, "reference", f"{ref_v}.tif")
    annot_orig_fp = os.path.join(atlas_rsc_dir, "annotation", f"{annot_v}.tif")
    map_orig_fp = os.path.join(atlas_rsc_dir, "region_mapping", f"{region_map_v}.json")
    # ELASTIX PARAM FILES ORIGIN
    affine_orig_fp = os.path.join(atlas_rsc_dir, "elastix_params", "align_affine.txt")
    bspline_orig_fp = os.path.join(atlas_rsc_dir, "elastix_params", "align_bspline.txt")

    # MY ATLAS FILES
    ref_fp = os.path.join(proj_dir, reg_sdir, "0a_reference.tif")
    annot_fp = os.path.join(proj_dir, reg_sdir, "0b_annotation.tif")
    region_map_fp = os.path.join(proj_dir, reg_sdir, "0c_mapping.json")
    # MY ELASTIX PARAM FILES
    affine_fp = os.path.join(proj_dir, reg_sdir, "0d_align_affine.txt")
    bspline_fp = os.path.join(proj_dir, reg_sdir, "0e_align_bspline.txt")

    # # REGISTRATION FILENAMES
    # resampled_rough_fp = os.path.join(proj_dir, registration_subdir, "1_resampled_rough.tif")
    # resampled_fp = os.path.join(proj_dir, registration_subdir, "2_resampled.tif")
    # resampled_sliced_fp = os.path.join(proj_dir, registration_subdir, "3_resampled_trimmed.tif")
    # registration_result_fp = os.path.join(proj_dir, registration_subdir, "4_registration_result.tif")
    # # CELL COUNTING FILENAMES
    # # TODO: clearmap detect cells configs JSON
    # illumination_fp = os.path.join(proj_dir, cellcount_subdir, "1_cells_illumination.npy")
    # bgremove_fp = os.path.join(proj_dir, cellcount_subdir, "2_cells_bgremove.npy")
    # equalization_fp = os.path.join(proj_dir, cellcount_subdir, "3_cells_equalization.npy")
    # dog_fp = os.path.join(proj_dir, cellcount_subdir, "4_cells_dog.npy")
    # maxima_fp = os.path.join(proj_dir, cellcount_subdir, "5_cells_maxima.npy")
    # shape_fp = os.path.join(proj_dir, cellcount_subdir, "6_cells_shape.npy")
    # intensity_fp = os.path.join(proj_dir, cellcount_subdir, "7_cells_intensity.npy")
    # cells_raw_fp = os.path.join(proj_dir, cellcount_subdir, "8_cells_raw.csv")
    # cells_filtered_fp = os.path.join(proj_dir, cellcount_subdir, "9_cells_filtered.csv")
    # cells_filtered_space_fp = os.path.join(proj_dir, cellcount_subdir, "10_cells_filtered_space.npy")
    # cells_transformed_fp = os.path.join(proj_dir, cellcount_subdir, "11_cells_transformed.csv")
    # cells_transformed_space_fp = os.path.join(proj_dir, cellcount_subdir, "12_cells_transformed_space.npy")
    # CELL DF FILENAME
    # cells_df_fp = os.path.join(proj_dir, "cells.csv")
    # ANALYSIS SUBDIRS
    # voxel_space_fp = os.path.join(proj_dir, analysis_subdir, "voxel_space.npy")

    os.makedirs(os.path.join(proj_dir, reg_sdir), exist_ok=True)

    ####### SPECIFYING ORIENTATION OF ATLAS FILES ###########
    # The order of the axes. A negative number indicates to flip that axis.
    orient_ls = (2, 3, 1)

    # How much to trim off from the edges of the reference image.
    x_slice_ref = slice(None, None)
    y_slice_ref = slice(None, None)
    z_slice_ref = slice(None, None)

    ################ DO NOT CHANGE BELOW ####################

    # Making atlas images
    # for fp_i, fp_o in [
    #     (ref_orig_fp, ref_fp),
    #     (annot_orig_fp, annot_fp),
    # ]:
    #     arr = tifffile.imread(fp_i)
    #     # Reorienting
    #     arr = reorient_arr(arr, orient_ls)
    #     # Slicing
    #     arr = arr[z_slice_ref, y_slice_ref, x_slice_ref]
    #     tifffile.imwrite(fp_o, arr)
    # # Copying region mapping json to project folder
    # shutil.copyfile(map_orig_fp, region_map_fp)

    # # Getting transformation files
    # shutil.copyfile(affine_orig_fp, affine_fp)
    # shutil.copyfile(bspline_orig_fp, bspline_fp)

    # Registration
    registration(
        fixed_img_fp=os.path.join(proj_dir, reg_sdir, "3_trimmed.tif"),
        moving_img_fp=ref_fp,
        output_img_fp=os.path.join(proj_dir, reg_sdir, "4_regresult.tif"),
        affine_fp=affine_fp,
        bspline_fp=bspline_fp,
    )

    # # Closing client
    # client.close()
