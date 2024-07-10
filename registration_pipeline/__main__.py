"""
the command line interface to the pipeline
"""

from __future__ import annotations
from pathlib import Path
import os
import sys
from typing import cast
import platform

import click


HELP_MESSAGE = """
Starts the registration pipeline using the czi file at
CZI_PATH and the scene index at INDEX. Opens a new napari window with the
plugin docked
Index is zero based
"""

def get_config_path():
    """
    gets the path to the config file
    """
    if os.name == "nt":
        app_data_dir = os.getenv("APPDATA")
        if app_data_dir is None:
            raise ValueError("OS missing appdata dir")
        return Path(app_data_dir) / "registration-pipeline/config.json"
    else:
        return Path().home() / ".config/registration-pipeline/config.json"



@click.command(help=HELP_MESSAGE)
@click.argument("czi-path", type=click.Path(exists=True))
@click.argument("index", type=int)
@click.option(
    "-n",
    "--ncpu",
    help=("Number of CPU cores to use in CMTK commands. Defaults to all cores"),
    required=False,
)
@click.option(
    "-o",
    "--out-dir",
    help=(
        "The directory where all output and intermediate files will be "
        "stored. Defaults to a newly created folder in the working "
        "directory"
    ),
    required=False,
    type=click.Path(),
)
@click.option(
    "-c",
    "--cmtk-path",
    help="Path to the cmtk executable",
    required=False,
    type=click.Path(exists=True),
)
@click.option(
    "-t",
    "--template-path",
    help="Path to directory containing template files",
    required=False,
    type=click.Path(exists=True),
)
@click.option(
    "-s",
    "--save-config",
    help="Save current config to the config file",
    is_flag=True,
    default=False
)
@click.option(
    "-d",
    "--delete-config",
    help="deletes the previos config file",
    is_flag=True,
    default=False
)
def main(
    czi_path: str | Path,
    index: int,
    ncpu: int | None,
    out_dir: str | Path | None,
    cmtk_path: str | Path | None,
    template_path: str | Path | None,
    save_config: bool,
    delete_config: bool
):
    """
    cli of the program
    """
    # resolve arguemnts
    config_path = get_config_path()
    if delete_config:
        config_path.unlink()

    czi_path = Path(czi_path)
    if czi_path.suffix != ".czi":
        print(
            f"Only CZI files are accepted. {czi_path} does "
            "not appear to be a czi file",
            file=sys.stderr,
        )
        sys.exit(1)
    # do as much validation as possible before importing the library
    from registration_pipeline.registration_config import RegistrationConfig # pylint: disable=import-outside-toplevel
    saved_config = RegistrationConfig.from_file(config_path) if config_path.exists() else None
    if ncpu is None:
        if saved_config is not None:
            ncpu = saved_config.ncpu
        else:
            ncpu = os.cpu_count()
            if ncpu is None:
                print(
                    "Could not determine the number of CPUs. Please pass the number "
                    "of CPUs you want to use using the --ncpu option",
                    file=sys.stderr,
                )
                sys.exit(1)
    if out_dir is None:
        # not using saved out_dir
        out_dir = Path(".") / f"{czi_path.name}-{index:03d}"
        if out_dir.exists():
            print(f"{out_dir} exists. Refusing to overwright it", file=sys.stderr)
            sys.exit(1)
    else:
        out_dir = Path(out_dir)
    if template_path is None:
        if saved_config is not None:
            template_path = saved_config.template_path
        else:
            print(
                "Please specify the path to the "
                "template directory using the --template-path option",
                file=sys.stderr,
            )
            sys.exit(1)
    template_path = Path(template_path)
    if not template_path.exists():
        print(
            f"{template_path} does not exist. Please specify the path to the "
            "template directory using the --template-path option",
            file=sys.stderr,
        )
        sys.exit(1)
    from registration_pipeline.registration_config import find_cmtk # pylint: disable=import-outside-toplevel
    if cmtk_path is None:
        cmtk_path = find_cmtk()
        if saved_config is not None:
            cmtk_path = saved_config.cmtk_exe_dir
        else:
            cmtk_path = find_cmtk()
        if cmtk_path is None:
            print(
                "CMTK not found. Please install from https://www.nitrc.org/projects/cmtk\n"
                "then add to path or specify installed path with the --cmtk-path option",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        cmtk_path = Path(cmtk_path)
    # add to path for xform to work properly
    os.environ["PATH"] += os.pathsep + str(cmtk_path.resolve())
    # launch the pipeline
    # import napari if things havent failed
    from registration_pipeline import ( # pylint: disable=import-outside-toplevel
        napari_plugin,
        landmarks,
    )
    import napari_scripts as ns  # pylint: disable=import-outside-toplevel
    import napari # pylint: disable=import-outside-toplevel
    print(f"loading {czi_path}")
    viewer = ns.get_viewer_at_czi_scene(czi_path, index, False)
    config = RegistrationConfig(
        template_path=template_path, cmtk_exe_dir=cmtk_path, out_dir=out_dir, ncpu=ncpu
    )
    if save_config:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_file(config_path)
    napari_plugin.launch_pipeline(viewer, landmarks.landmark_infos, config)
    napari.run()

main()
