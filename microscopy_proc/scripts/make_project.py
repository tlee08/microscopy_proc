from microscopy_proc.utils.logging_utils import init_logger
from microscopy_proc.utils.template_utils import import_static_templates_script

logger = init_logger(__name__)


def main() -> None:
    """
    Makes a script to run a behavysis analysis project.
    """
    import_static_templates_script(
        description="Make Microscopy Pipeline Script",
        templates_ls=["run_pipeline.py", "view_img.py"],
        pkg_name="microscopy_proc",
        pkg_subdir="templates",
        root_dir=".",
        overwrite=False,
        dialogue=True,
    )
    # # Copying default configs file to the project folder
    # write_json(os.path.join(root_dir, "default_configs.json"), ConfigParamsModel().model_dump())


if __name__ == "__main__":
    main()
