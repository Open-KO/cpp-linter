from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Optional, List, Dict, Tuple, cast
import shutil

from ..common_fs import FileObj, FileIOTimeout
from ..common_fs.file_filter import TidyFileFilter, FormatFileFilter
from ..loggers import start_log_group, end_log_group, worker_log_init, logger
from .clang_tidy import run_clang_tidy, TidyAdvice
from .clang_format import run_clang_format, FormatAdvice
from ..cli import Args


def assemble_version_exec(tool_name: str, specified_version: str) -> Optional[str]:
    """Assembles the command to the executable of the given clang tool based on given
    version information.

    :param tool_name: The name of the clang tool to be executed.
    :param specified_version: The version number or the installed path to a version of
        the tool's executable.
    """
    semver = specified_version.split(".")
    exe_path = None
    if semver and semver[0].isdigit():  # version info is not a path
        # let's assume the exe is in the PATH env var
        exe_path = shutil.which(f"{tool_name}-{specified_version}")
    elif specified_version:  # treat value as a path to binary executable
        exe_path = shutil.which(tool_name, path=specified_version)
    if exe_path is not None:
        return exe_path
    return shutil.which(tool_name)


def _run_on_single_file(
    file: FileObj,
    log_lvl: int,
    tidy_cmd: Optional[str],
    db_json: Optional[List[Dict[str, str]]],
    format_cmd: Optional[str],
    format_filter: Optional[FormatFileFilter],
    tidy_filter: Optional[TidyFileFilter],
    tidy_line_filter_json: Optional[str],
    args: Args,
) -> Tuple[str, str, Optional[TidyAdvice], Optional[FormatAdvice]]:
    log_stream = worker_log_init(log_lvl)
    filename = Path(file.name).as_posix()

    format_advice = None
    if format_cmd is not None and (
        format_filter is None or format_filter.is_source_or_ignored(file.name)
    ):
        try:
            format_advice = run_clang_format(
                command=format_cmd,
                file_obj=file,
                style=args.style,
                lines_changed_only=args.lines_changed_only,
                format_review=args.format_review,
            )
        except FileIOTimeout:  # pragma: no cover
            logger.error(
                "Failed to read or write contents of %s when running clang-format",
                filename,
            )
        except OSError:  # pragma: no cover
            logger.error(
                "Failed to open the file %s when running clang-format", filename
            )

    tidy_note = None
    if tidy_cmd is not None and (
        tidy_filter is None or tidy_filter.is_source_or_ignored(file.name)
    ):
        try:
            tidy_note = run_clang_tidy(
                command=tidy_cmd,
                file_obj=file,
                checks=args.tidy_checks,
                lines_changed_only=args.lines_changed_only,
                database=args.database,
                extra_args=args.extra_arg,
                db_json=db_json,
                tidy_review=args.tidy_review,
                style=args.style,
                line_filter_json=tidy_line_filter_json,
            )
        except FileIOTimeout:  # pragma: no cover
            logger.error(
                "Failed to Read/Write contents of %s when running clang-tidy", filename
            )
        except OSError:  # pragma: no cover
            logger.error("Failed to open the file %s when running clang-tidy", filename)

    return file.name, log_stream.getvalue(), tidy_note, format_advice


VERSION_PATTERN = re.compile(r"version\s(\d+\.\d+\.\d+)")


def _capture_tool_version(cmd: str) -> str:
    """Get version number from output for executable used."""
    version_out = subprocess.run(
        [cmd, "--version"], capture_output=True, check=True, text=True
    )
    matched = VERSION_PATTERN.search(version_out.stdout)
    if matched is None:  # pragma: no cover
        raise RuntimeError(
            f"Failed to get version numbers from `{cmd} --version` output"
        )
    ver = cast(str, matched.group(1))
    logger.info("`%s --version`: %s", cmd, ver)
    return ver


class ClangVersions:
    def __init__(self) -> None:
        self.tidy: Optional[str] = None
        self.format: Optional[str] = None


def capture_clang_tools_output(
    files: List[FileObj],
    args: Args,
    files_to_analyze_tidy: Optional[List[FileObj]] = None,
) -> ClangVersions:
    """Execute and capture all output from clang-tidy and clang-format. This aggregates
    results in the :attr:`~cpp_linter.Globals.OUTPUT`.

    :param files: A list of files to process.
    :param args: A namespace of parsed args from the :doc:`CLI <../cli_args>`.
    :param files_to_analyze_tidy: A list of files for clang-tidy to analyze.
    """

    tidy_cmd, format_cmd = (None, None)
    tidy_filter, format_filter = (None, None)
    clang_versions = ClangVersions()
    if args.style:  # if style is an empty value, then clang-format is skipped
        format_cmd = assemble_version_exec("clang-format", args.version)
        if format_cmd is None:  # pragma: no cover
            raise FileNotFoundError("clang-format executable was not found")
        clang_versions.format = _capture_tool_version(format_cmd)
        format_filter = FormatFileFilter(
            extensions=args.extensions,
            ignore_value=args.ignore_format,
        )
    if args.tidy_checks != "-*":
        # if all checks are disabled, then clang-tidy is skipped
        tidy_cmd = assemble_version_exec("clang-tidy", args.version)
        if tidy_cmd is None:  # pragma: no cover
            raise FileNotFoundError("clang-tidy executable was not found")
        clang_versions.tidy = _capture_tool_version(tidy_cmd)
        tidy_filter = TidyFileFilter(
            extensions=args.extensions,
            ignore_value=args.ignore_tidy,
        )

    db_json: Optional[List[Dict[str, str]]] = None
    if args.database:
        db = Path(args.database)
        if not db.is_absolute():
            args.database = str(db.resolve())
        db_path = (db / "compile_commands.json").resolve()
        if db_path.exists():
            db_json = json.loads(db_path.read_text(encoding="utf-8"))

    tidy_line_filter_json = None
    if args.lines_changed_only and files_to_analyze_tidy:
        all_line_ranges = []
        for file in files_to_analyze_tidy:
            line_ranges = {
                "name": file.name.replace("/", os.sep),
                "lines": file.range_of_changed_lines(
                    args.lines_changed_only, get_ranges=True
                ),
            }

            if line_ranges["lines"]:
                all_line_ranges.append(line_ranges)

        tidy_line_filter_json = json.dumps(all_line_ranges)

    with ProcessPoolExecutor(args.jobs) as executor:
        log_lvl = logger.getEffectiveLevel()
        futures = []
        for file in files:
            if args.lines_changed_only and not files_to_analyze_tidy:
                line_ranges = {
                    "name": file.name.replace("/", os.sep),
                    "lines": file.range_of_changed_lines(
                        args.lines_changed_only, get_ranges=True
                    ),
                }

                if line_ranges["lines"]:
                    tidy_line_filter_json = json.dumps([line_ranges])

            futures.append(
                executor.submit(
                    _run_on_single_file,
                    file,
                    log_lvl=log_lvl,
                    tidy_cmd=tidy_cmd,
                    db_json=db_json,
                    format_cmd=format_cmd,
                    format_filter=format_filter,
                    tidy_filter=tidy_filter,
                    tidy_line_filter_json=tidy_line_filter_json,
                    args=args,
                )
            )

        applicable_file_list = files_to_analyze_tidy or files
        files_by_name = {}
        for file in applicable_file_list:
            files_by_name[file.name] = file

        # temporary cache of parsed notifications for use in log commands
        for future in as_completed(futures):
            file_name, logs, tidy_advice, format_advice = future.result()

            start_log_group(f"Performing checkup on {file_name}")
            try:
                print(logs, flush=True)
            except UnicodeEncodeError:
                sys.stdout.buffer.write(logs.encode("utf-8", errors="replace"))
                sys.stdout.flush()
            end_log_group()

            if format_advice:
                for file in files:
                    if file.name == file_name:
                        file.format_advice = format_advice
                        break
                # else:
                #    raise ValueError(f"Failed to find {file_name} in list of files.")

            if tidy_advice:
                for note in tidy_advice.notes:
                    note_file = files_by_name.get(note.filename)
                    if note_file is None:
                        # raise ValueError(
                        #    f"Failed to find {note.filename} in list of files."
                        # )
                        continue

                    # already has advice
                    if note_file.tidy_advice:
                        # verify that we're not inserting a duplicate
                        if not note_file.tidy_advice.has_note(note):
                            note_file.tidy_advice.notes.append(note)
                    # no advice attached to this file yet
                    else:
                        note_file.tidy_advice = TidyAdvice(notes=[note])

    return clang_versions
