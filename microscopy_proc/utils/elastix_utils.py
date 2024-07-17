import os
import re

import dask.dataframe as dd
import numpy as np
import pandas as pd
import SimpleITK as sitk

from microscopy_proc.utils.io_utils import silentremove


def registration(
    fixed_img_fp: str,
    moving_img_fp: str,
    output_img_fp: str,
    affine_fp: str = None,
    bspline_fp: str = None,
):
    """
    Uses SimpleElastix (a plugin for SimpleITK)

    Params:
        TODO
    """
    output_img_dir = os.path.split(output_img_fp)[0]
    # Setting up Elastix object
    elastix_img_filt = sitk.ElastixImageFilter()
    # Setting the fixed, moving, and output image filepaths
    elastix_img_filt.SetFixedImage(sitk.ReadImage(fixed_img_fp))
    elastix_img_filt.SetMovingImage(sitk.ReadImage(moving_img_fp))
    elastix_img_filt.SetOutputDirectory(output_img_dir)

    # Parameter maps: translation, affine, bspline
    # Translation
    # parameter_map_translation = sitk.GetDefaultParameterMap("translation")
    # elastix_img_filt.SetParameterMap(parameter_map_translation)
    # Affine
    if affine_fp:
        params_affine = sitk.ReadParameterFile(affine_fp)
    else:
        params_affine = sitk.GetDefaultParameterMap("affine")
    elastix_img_filt.SetParameterMap(params_affine)
    # Bspline
    if affine_fp:
        params_bspline = sitk.ReadParameterFile(bspline_fp)
    else:
        params_bspline = sitk.GetDefaultParameterMap("bspline")
    elastix_img_filt.AddParameterMap(params_bspline)

    # Setting feedback and logging settings
    elastix_img_filt.LogToFileOff()
    elastix_img_filt.LogToConsoleOn()
    # Running registration
    elastix_img_filt.Execute()
    # Saving output file
    sitk.WriteImage(elastix_img_filt.GetResultImage(), output_img_fp)
    # Removing temporary and unecessary elastix files
    for i in os.listdir(output_img_dir):
        # Removing IterationInfo files
        if re.search(r"^IterationInfo.(\d+).R(\d+).txt$", i):
            silentremove(os.path.join(output_img_dir, i))
    # Returning the moved image (output_img) array
    return elastix_img_filt.GetResultImage()


def transformation_coords(
    coords: pd.DataFrame | dd.DataFrame,
    moving_img_fp: str,
    output_img_fp: str,
):
    """
    Uses the transformation parameter output from registration to transform
    cell coordinates from the fixed image space to moving image space.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        moving_img_fp: Filepath of the moved image from registration (typically the reference image).
        output_img_fp: Filepath of the outputted image from registration (the fixed image after warping). Important that the TransformParameters files are in this folder.

    Returns:
        A pd.DataFrame of the transformed coordinated from the fixed image space to the moving image space.
    """
    output_img_dir = os.path.split(output_img_fp)[0]
    # Setting up Transformix object
    transformix_img_filt = sitk.TransformixImageFilter()
    # Setting the fixed points and moving and output image filepaths
    # Converting cells array to a fixed_points file
    # NOTE: xyz, NOT zyx
    make_fixed_points_file(
        coords[["x", "y", "z"]].values, os.path.join(output_img_dir, "temp.dat")
    )
    transformix_img_filt.SetFixedPointSetFileName(
        os.path.join(output_img_dir, "temp.dat")
    )
    transformix_img_filt.SetMovingImage(sitk.ReadImage(moving_img_fp))
    transformix_img_filt.SetOutputDirectory(output_img_dir)
    # Transform parameter maps: from registration - affine, bspline
    transformix_img_filt.SetTransformParameterMap(
        sitk.ReadParameterFile(
            os.path.join(output_img_dir, "TransformParameters.0.txt")
        )
    )
    transformix_img_filt.AddTransformParameterMap(
        sitk.ReadParameterFile(
            os.path.join(output_img_dir, "TransformParameters.1.txt")
        )
    )
    # Execute cell transformation
    transformix_img_filt.Execute()
    # Converting transformix output to df
    coords_transformed = transformix_file_to_coords(
        os.path.join(output_img_dir, "outputpoints.txt")
    )
    # Removing temporary and unecessary transformix files
    silentremove(os.path.join(output_img_dir, "temp.dat"))
    silentremove(os.path.join(output_img_dir, "outputpoints.txt"))
    # Returning transformed coords
    return coords_transformed


def make_fixed_points_file(coords, fixed_points_fp):
    """
    https://simpleelastix.readthedocs.io/PointBasedRegistration.html

    Takes a list of `(x, y, z, ...)` arrays and converts it to the .pts file format for transformix.
    """
    with open(fixed_points_fp, "w") as f:
        f.write("index\n")
        f.write(f"{coords.shape[0]}\n")
        for i in np.arange(coords.shape[0]):
            f.write(f"{coords[i, 0]} {coords[i, 1]} {coords[i, 2]}\n")


def transformix_file_to_coords(output_points_fp):
    """
    Takes filename of the transformix output points and converts it to a pd.DataFrame of points.

    Params:
        output_points_fp: Filename of the ouput points.

    Returns:
        pd.DataFrame of points with the columns `x`, `y`, and `z`.
    """
    df = pd.read_csv(output_points_fp, header=None, sep=";")
    df.columns = df.loc[0].str.strip().str.split(r"\s").str[0]
    # Try either "OutputIndexFixed" or "OutputPoint"
    df = df["OutputPoint"].apply(
        lambda x: [float(i) for i in x.replace(" ]", "").split("[ ")[1].split()]
    )
    return pd.DataFrame(df.values.tolist(), columns=["x", "y", "z"])


def transformation_img(
    moving_img_fp: str,
    output_img_fp: str,
):
    """
    Uses the transformation parameter output from registration to transform
    cell coordinates from the fixed image space to moving image space.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        moving_img_fp: Filepath of the moved image from registration (typically the reference image).
        output_img_fp: Filepath of the outputted image from registration (the fixed image after warping). Important that the TransformParameters files are in this folder.

    Returns:
        A pd.DataFrame of the transformed coordinated from the fixed image space to the moving image space.
    """
    output_img_dir = os.path.split(output_img_fp)[0]
    # Setting up Transformix object
    transformix_img_filt = sitk.TransformixImageFilter()
    # Setting the fixed points and moving and output image filepaths
    # Converting cells array to a fixed_points file
    transformix_img_filt.SetFixedPointSetFileName(
        os.path.join(output_img_dir, "temp.dat")
    )
    transformix_img_filt.SetMovingImage(sitk.ReadImage(moving_img_fp))
    transformix_img_filt.SetOutputDirectory(output_img_dir)
    # Transform parameter maps: from registration - affine, bspline
    transformix_img_filt.SetTransformParameterMap(
        sitk.ReadParameterFile(
            os.path.join(output_img_dir, "TransformParameters.0.txt")
        )
    )
    transformix_img_filt.AddTransformParameterMap(
        sitk.ReadParameterFile(
            os.path.join(output_img_dir, "TransformParameters.1.txt")
        )
    )
    # Execute cell transformation
    transformix_img_filt.Execute()
    # # Converting transformix output to df
    # coords_transformed = transformix_file_to_coords(
    #     os.path.join(output_img_dir, "outputpoints.txt")
    # )
    # # Removing temporary and unecessary transformix files
    # silentremove(os.path.join(output_img_dir, "temp.dat"))
    # silentremove(os.path.join(output_img_dir, "outputpoints.txt"))
    # # Returning transformed coords
    # return coords_transformed
    return transformix_img_filt.GetResultImage()
