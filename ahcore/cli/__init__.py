# coding=utf-8
# Copyright (c) dlup contributors
"""Ahcore Command-line interface. This is the file which builds the main parser."""
from __future__ import annotations

import argparse
import pathlib


def dir_path(path: str) -> pathlib.Path:
    """Check if the path is a valid directory.

    Parameters
    ----------
    path : str

    Returns
    -------
    pathlib.Path
        The path as a pathlib.Path object.
    """
    _path = pathlib.Path(path)
    if _path.is_dir():
        return _path
    raise argparse.ArgumentTypeError(f"{path} is not a valid directory.")


def file_path(path: str) -> pathlib.Path:
    """Check if the path is a valid file.

    Parameters
    ----------
    path : str

    Returns
    -------
    pathlib.Path
        The path as a pathlib.Path object.

    """
    _path = pathlib.Path(path)
    if _path.is_file():
        return _path
    raise argparse.ArgumentTypeError(f"{path} is not a valid file.")


def main() -> None:
    """
    Main entrypoint for the CLI command of ahcore.
    """
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    root_subparsers = root_parser.add_subparsers(help="Possible ahcore CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"


if __name__ == "__main__":
    main()
