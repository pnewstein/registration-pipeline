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

def find_cmtk() -> Path | None:
    """
    returns the cmtk executable or None if it cannot be found
    """
    search_path = [p for p in os.environ["PATH"].split(os.pathsep) if len(p) > 0]
    search_path += ['~/bin',
                    '/usr/lib/cmtk/bin/',
                    '/usr/local/lib/cmtk/bin',
                    '/usr/local/bin/cmtk/bin',
                    '/opt/local/bin',
                    '/opt/local/bin/cmtk/bin',
                    '/Applications/IGSRegistrationTools/bin']
    if platform.system() == "Windows":
        search_path += [r'C:\cygwin64\usr\local\lib\cmtk\bin',
                        r'C:\Program Files\CMTK-3.3\CMTK\lib\cmtk\bin',
                        r'C:\Program Files\CMTK-3.3\CMTK\bin',]
    for path_str in search_path:
        path = Path(path_str)
        if not path.is_dir():
            continue
        try:
            return next(path.glob("cmtk"))
        except StopIteration:
            continue

    
@click.command(help=HELP_MESSAGE)
@click.argument("czi-path", type=click.Path(exists=True))
@click.argument("index", type=int)
@click.option(
    "-n",
    "--ncpu",
    help=("Number of CPU cores to use in CMTK commands. "
          "Defaults to all cores"),
    required=False,
)
@click.option(
    "-o",
    "--out-dir",
    help=("The directory where all output and intermediate files will be "
          "stored. Defaults to a newly created folder in the working "
          "directory"),
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
def main(
    czi_path: str | Path,
    index: int,
    ncpu: int | None,
    out_dir: str | Path | None,
    cmtk_path: str | Path | None,
    template_path: str | Path | None
):
    """
    cli of the program
    """
    # resolve arguemnts
    czi_path = Path(czi_path)
    if czi_path.suffix != ".czi":
        print(f"Only CZI files are accepted. {czi_path} does "
              "not appear to be a czi file", file=sys.stderr)
        sys.exit(1)
    if ncpu is None:
        ncpu = os.cpu_count()
        if ncpu is None:
            print("Could not determine the number of CPUs. Please pass the number "
                  "of CPUs you want to use using the --ncpu option", file=sys.stderr)
            sys.exit(1)
    if out_dir is None:
        out_dir = Path(".") / f"{czi_path.name}-{index:03d}"
        if out_dir.exists():
            print(f"{out_dir} exists. Refusing to overwright it", file=sys.stderr)
            sys.exit(1)
    else:
        out_dir = Path(out_dir)
    if cmtk_path is None:
        cmtk_path = find_cmtk()
        if cmtk_path is None:
            print("CMTK not found. Please install from https://www.nitrc.org/projects/cmtk\n"
                  "then add to path or specify installed path with the --cmtk-path option", 
                  file=sys.stderr)
            sys.exit(1)
    else:
        cmtk_path = Path(cmtk_path)
    if template_path is None:
        template_path = Path().home() / "templates/JRC2018_UNISEX"
        if not template_path.exists():
            print(f"{template_path} does not exist. Please specify the path to the "
                  "template directory using the --template-path option", file=sys.stderr)
    else:
        template_path = Path(template_path)

    # launch the pipeline
    # import napari if things havent failed
    from registration_pipeline import napari_plugin, registration_config, landmarks# pylint: disable=import-outside-toplevel
    import napari_scripts as ns # pylint: disable=import-outside-toplevel
    import napari
    print(f"loading {czi_path}")
    viewer = ns.get_viewer_at_czi_scene(czi_path, index, False)
    config = registration_config.RegistrationConfig(
        template_path=template_path,
        cmtk_exe=cmtk_path,
        out_dir=out_dir,
        ncpu=ncpu
    )
    napari_plugin.launch_pipeline(viewer, landmarks.landmark_infos, config)
    napari.run()


main()
