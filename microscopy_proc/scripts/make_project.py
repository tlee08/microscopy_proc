import os

from microscopy_proc.utils.io_utils import save_template


def import_template(src_fp, dst_fp, overwrite):
    """
    Imports the template file to the project folder.
    """
    # If not overwrite and file exists, then don't overwrite
    if not overwrite and os.path.exists(dst_fp):
        print(f"File {dst_fp} already exists and overwriting set to False. Not overwriting.")
        return
    # Saving the template to the file
    save_template(
        src_fp,
        "microscopy_proc",
        "templates",
        dst_fp,
    )


def main(root_dir: str = ".", overwrite: bool = False, dialogue: bool = True) -> None:
    """
    Makes a script to run a behavysis analysis project.

    Copies the `run_project.py` script and `default_configs.json` to `root_dir`.
    """
    # Making the root folder
    os.makedirs(root_dir, exist_ok=True)
    if dialogue:
        # Dialogue to check if the user wants to make the files
        to_continue = input("Making project in current directory. Continue? [y/N]: ").lower() + " "
        if to_continue[0] != "y":
            print("Exiting.")
            return
        # Dialogue to check if the user wants to overwrite the files
        to_overwrite = input("Overwrite existing files? [y/N]: ").lower() + " "
        if to_overwrite[0] == "y":
            overwrite = True
        else:
            overwrite = False
    # Copying the Python files to the project folder
    for src_fp in ["batch_pipeline.py", "view_img.py"]:
        import_template(src_fp, os.path.join(root_dir, src_fp), overwrite)
    # # Copying default configs file to the project folder
    # write_json(os.path.join(root_dir, "default_configs.json"), ConfigParamsModel().model_dump())


if __name__ == "__main__":
    main()
