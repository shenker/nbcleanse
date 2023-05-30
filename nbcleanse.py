#!/usr/bin/env python3

"""
nbcleanse filter-repo should be run from the root of a git repo. It will format
.py and .ipynb files with black, as well as strip output from .ipynb files.
nbcleanse install/uninstall will enable/disable .ipynb clean/smudge filtering.

nbcleanse filter-repo modified from https://gist.github.com/JacobHayes/9c86cc650c17a776f4a592fe0b2e7645
See also https://github.com/newren/git-filter-repo/issues/45
and https://github.com/newren/git-filter-repo/blob/master/contrib/filter-repo-demos/lint-history
nbcleanse install/uninstall/status/filter modified from nbstripout (https://github.com/kynan/nbstripout/).
"""

import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Collection
from functools import partial
from pathlib import Path, PurePath
from subprocess import DEVNULL, STDOUT, CalledProcessError
from textwrap import dedent

import black
import cachetools
import click
import isort
import nbformat

from pktline import start_filter_server

filter_ = filter

DEFAULT_EXTRA_KEYS = [
    "metadata.signature",
    "metadata.widgets",
    "cell.metadata.collapsed",
    "cell.metadata.ExecuteTime",
    "cell.metadata.execution",
    "cell.metadata.heading_collapsed",
    "cell.metadata.hidden",
    "cell.metadata.scrolled",
    "cell.metadata.jupyter.outputs_hidden",
]

PARENT_DIR = Path(__file__).resolve().parent
TIMESTAMP_FILE = PARENT_DIR / ".last_updated"
UPDATE_INTERVAL = 5 * 60  # seconds

blobs_handled = {}
format_cache = cachetools.LRUCache(maxsize=10_000_000, getsizeof=sys.getsizeof)


def path_to_posix(path):
    return str(PurePath(path).as_posix())


def is_git_pull_needed():
    # FROM: https://stackoverflow.com/questions/3258243/check-if-pull-needed-in-git
    commands = {
        "local": ["git", "rev-parse", "@"],
        "remote": ["git", "rev-parse", "@{u}"],
        "base": ["git", "merge-base", "@", "@{u}"],
    }
    revs = {}
    try:
        subprocess.run(
            ["git", "remote", "update"],
            cwd=PARENT_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        for key, args in commands.items():
            res = subprocess.run(
                args, cwd=PARENT_DIR, text=True, capture_output=True, check=True
            )
            revs[key] = res.stdout.strip()
        if revs["local"] == revs["remote"]:
            # up to date
            return False
        elif revs["local"] == revs["base"]:
            # need to pull
            return True
        elif revs["local"] == revs["remote"]:
            # need to push
            click.secho(
                "WARNING: please push nbcleanse commits to remote", err=True, bold=True
            )
            return False
        else:
            # diverged
            click.secho(
                "WARNING: nbcleanse local and remote have diverged, cannot update",
                err=True,
                bold=True,
            )
            return False
    except CalledProcessError as e:
        click.secho("could not check for nbcleanse update", err=True, bold=True)
        click.secho("please try updating nbcleanse manually", err=True, bold=True)


def is_update_needed():
    last_updated = None
    try:
        if TIMESTAMP_FILE.exists():
            last_updated = float(TIMESTAMP_FILE.read_text())
    except:
        pass
    now = time.time()
    return not last_updated or now - last_updated >= UPDATE_INTERVAL


def git_pull_if_needed(
    pyproject_file=None,
    gitattrs_file=None,
    conda_env=None,
    autoupdate=None,
    long_running=None,
):
    if not autoupdate:
        return
    if not is_update_needed():
        return
    pulled = is_git_pull_needed()
    if pulled:
        click.secho(
            "nbcleanse update available, running git pull...", err=True, bold=True
        )
        subprocess.run(["git", "pull"], text=True, cwd=PARENT_DIR, check=True)
        if conda_env:
            click.echo(
                click.style("updating nbcleanse conda environment '", bold=True)
                + conda_env
                + click.style("' (if necessary)...", bold=True),
                err=True,
            )
            envyml = PARENT_DIR / "environment.yml"
            subprocess.run(
                ["mamba", "env", "update", "--prune", "-n", conda_env, "-f", envyml],
                cwd=PARENT_DIR,
                text=True,
                stdout=subprocess.STDERR,
                stderr=subprocess.STDERR,
                check=True,
            )
        click.secho("reinstalling nbcleanse...", err=True, bold=True)
        _install(
            pyproject_file=pyproject_file,
            gitattrs_file=gitattrs_file,
            conda_env=conda_env,
            autoupdate=autoupdate,
            long_running=long_running,
        )
    now = time.time()
    with open(TIMESTAMP_FILE, "w") as f:
        f.write(f"{now}\n")
    return pulled


def load_pyproject_configs(pyproject_file):
    isort_config = isort.settings.Config(pyproject_file)
    black_config = black.parse_pyproject_toml(pyproject_file)
    # SEE: https://github.com/akaihola/darker/blob/a7f586428f44d07795dba601fb21db0fb5ede22b/src/darker/black_diff.py#L190
    # SEE: https://github.com/psf/black/blob/c42178690e96f3bf061ad44f70dec52b1d8a299a/src/black/__init__.py#L542
    black_mode = {}
    if "line_length" in black_config:
        black_mode["line_length"] = black_config["line_length"]
    if "target_version" in black_config:
        if isinstance(black_config["target_version"], Collection):
            target_versions_in = set(black_config["target_version"])
        else:
            target_versions_in = {black_config["target_version"]}
        all_target_versions = {v.name.lower(): v for v in black.TargetVersion}
        bad_target_versions = target_versions_in - set(all_target_versions)
        if bad_target_versions:
            raise ValueError(f"Invalid target version(s): {bad_target_versions}")
        black_mode["target_versions"] = {
            all_target_versions[n] for n in target_versions_in
        }
    if "skip_source_first_line" in black_config:
        black_mode["skip_source_first_line"] = not black_config[
            "skip_source_first_line"
        ]
    if "skip_string_normalization" in black_config:
        black_mode["string_normalization"] = not black_config[
            "skip_string_normalization"
        ]
    if "skip_magic_trailing_comma" in black_config:
        black_mode["magic_trailing_comma"] = black_config["skip_magic_trailing_comma"]
    if "experimental_string_processing" in black_config:
        black_mode["experimental_string_processing"] = black_config[
            "experimental_string_processing"
        ]
    if "preview" in black_config:
        black_mode["preview"] = black_config["preview"]
    if "python_cell_magics" in black_config:
        black_mode["python_cell_magics"] = set(black_config["python_cell_magics"])
    return dict(
        isort=isort_config,
        black=dict(
            fast=True,
            mode=black.FileMode(**black_mode),
        ),
    )


class Formatter:
    def __init__(self, config):
        self.clear_exceptions()
        if config is not None:
            self.config = config
        else:
            self.config = dict(
                black=dict(fast=False, mode=black.FileMode()), isort=None
            )

    @cachetools.cached(format_cache)
    def format(self, content, enable_isort=True, enable_black=True):
        try:
            raise ValueError("foobar")
            if enable_isort:
                content = isort.api.sort_code_string(content, self.config["isort"])
            if enable_black:
                content = black.format_file_contents(content, **self.config["black"])
        except black.NothingChanged:
            return content
        except Exception as exc:
            self.exceptions.append(exc)
            return None
        return content

    @cachetools.cached(format_cache)
    def format_cell(self, content, enable_isort=True, enable_black=True):
        try:
            if enable_isort:
                content = isort.api.sort_code_string(content, self.config["isort"])
            if enable_black:
                content = black.format_cell(content, **self.config["black"])
        except black.NothingChanged:
            return content
        except Exception as exc:
            self.exceptions.append(exc)
            return None
        return content

    def clear_exceptions(self):
        self.exceptions = []

    def print_exceptions(self, limit=None, file=sys.stderr, chain=True):
        print(
            f"Generated {len(self.exceptions)} exceptions when formatting:", file=file
        )
        formatted_exceptions = Counter()
        for exc in self.exceptions:
            msg = "".join(traceback.format_exception(exc, limit=limit, chain=chain))
            formatted_exceptions[msg] += 1
        for msg, count in formatted_exceptions.most_common():
            print(file=file)
            print(f"{msg[:-1]} [{count} exceptions]", file=file)


def filter_py(formatter, content, filename):
    content = content
    new_content = formatter.format(content)
    if new_content is None:
        click.echo(f"Unable to format {filename}", err=True)
        return None
    return new_content


# FROM: https://github.com/kynan/nbstripout/blob/master/nbstripout/_utils.py
def pop_recursive(d, key, default=None):
    """dict.pop(key) where `key` is a `.`-delimited list of nested keys.

    >>> d = {'a': {'b': 1, 'c': 2}}
    >>> pop_recursive(d, 'a.c')
    2
    >>> d
    {'a': {'b': 1}}
    """
    if not isinstance(d, dict):
        return default
    if key in d:
        return d.pop(key, default)
    if "." not in key:
        return default
    key_head, key_tail = key.split(".", maxsplit=1)
    if key_head in d:
        return pop_recursive(d[key_head], key_tail, default)
    return default


# FROM: https://github.com/kynan/nbstripout/blob/master/nbstripout/_utils.py
def _cells(nb, conditionals):
    """Remove cells not satisfying any conditional in conditionals and yield all other cells."""
    if nb.nbformat < 4:
        for ws in nb.worksheets:
            for conditional in conditionals:
                ws.cells = list(filter_(conditional, ws.cells))
            for cell in ws.cells:
                yield cell
    else:
        for conditional in conditionals:
            nb.cells = list(filter_(conditional, nb.cells))
        for cell in nb.cells:
            yield cell


# FROM: https://github.com/kynan/nbstripout/blob/master/nbstripout/_utils.py
def determine_keep_output(cell, default, strip_init_cells=False):
    """Given a cell, determine whether output should be kept

    Based on whether the metadata has "init_cell": true,
    "keep_output": true, or the tags contain "keep_output" """
    if "metadata" not in cell:
        return default
    if "init_cell" in cell.metadata:
        return bool(cell.metadata.init_cell) and not strip_init_cells
    has_keep_output_metadata = "keep_output" in cell.metadata
    keep_output_metadata = bool(cell.metadata.get("keep_output", False))
    has_keep_output_tag = "keep_output" in cell.metadata.get("tags", [])
    # keep_output between metadata and tags should not contradict each other
    if has_keep_output_metadata and has_keep_output_tag and not keep_output_metadata:
        raise ValueError(
            "cell metadata contradicts tags: `keep_output` is false, but `keep_output` in tags"
        )

    if has_keep_output_metadata or has_keep_output_tag:
        return keep_output_metadata or has_keep_output_tag
    return default


# FROM: https://github.com/kynan/nbstripout/blob/master/nbstripout/_utils.py
def strip_jupyter(
    formatter,
    nb,
    filename,
    keep_output=False,
    keep_count=False,
    keep_id=True,
    extra_keys=[],
    drop_empty_cells=True,
    drop_tagged_cells=[],
    strip_init_cells=False,
):
    """
    Strip the outputs, execution count/prompt number and miscellaneous
    metadata from a notebook object, unless specified to keep either the outputs
    or counts.

    `extra_keys` could be 'metadata.foo cell.metadata.bar metadata.baz'
    """
    if keep_output is None and "keep_output" in nb.metadata:
        keep_output = bool(nb.metadata["keep_output"])
    keys = defaultdict(list)
    for key in extra_keys:
        if "." not in key or key.split(".")[0] not in ["cell", "metadata"]:
            sys.stderr.write(f"Ignoring invalid extra key `{key}`\n")
        else:
            namespace, subkey = key.split(".", maxsplit=1)
            keys[namespace].append(subkey)
    for field in keys["metadata"]:
        pop_recursive(nb.metadata, field)
    conditionals = []
    # Keep cells if they have any `source` line that contains non-whitespace
    if drop_empty_cells:
        conditionals.append(lambda c: any(line.strip() for line in c.get("source", [])))
    for tag_to_drop in drop_tagged_cells:
        conditionals.append(
            lambda c: tag_to_drop not in c.get("metadata", {}).get("tags", [])
        )
    failed_cells = 0
    for i, cell in enumerate(_cells(nb, conditionals)):
        keep_output_this_cell = determine_keep_output(
            cell, keep_output, strip_init_cells
        )
        # remove the outputs, unless directed otherwise
        if "outputs" in cell:
            # Default behavior (max_size == 0) strips all outputs.
            if not keep_output_this_cell:
                cell["outputs"] = []
            # strip the counts from the outputs that were kept if not keep_count
            if not keep_count:
                for output in cell["outputs"]:
                    if "execution_count" in output:
                        output["execution_count"] = None
            # if keep_output_this_cell and keep_count, do nothing.
        # remove the prompt_number/execution_count, unless directed otherwise
        if "prompt_number" in cell and not keep_count:
            cell["prompt_number"] = None
        if "execution_count" in cell and not keep_count:
            cell["execution_count"] = None
        # replace the cell id with an incremental value that will be consistent across runs
        if not ("id" in cell and keep_id):
            cell["id"] = str(i)
        for field in keys["cell"]:
            pop_recursive(cell, field)
        if cell["cell_type"] == "code":
            try:
                new_source = formatter.format_cell(cell["source"])
            except:
                new_source = None
            if new_source is not None:
                cell["source"] = new_source
            else:
                failed_cells += 1
    if failed_cells:
        err_msg = f"\nFailed to format {failed_cells} cells in notebook"
        if filename:
            err_msg += f" {filename}"
        click.echo(err_msg, err=True)
    return nb


def filter_jupyter(formatter, content, filename, **kwargs):
    try:
        nb = nbformat.reads(content + "\n\n", nbformat.NO_CONVERT)
    except:
        click.echo(f"\nUnable to parse notebook {filename}", err=True)
        return None
    nb = strip_jupyter(formatter, nb, filename, **kwargs)
    new_content = nbformat.writes(nb) + "\n"
    return new_content


filetype_filters = {"py": filter_py, "ipynb": filter_jupyter}


def filter_commit(
    formatter,
    commit,
    metadata,
    cat_file_process=None,
    repo_filter=None,
):
    import git_filter_repo as fr

    for change in commit.file_changes:
        filename = change.filename.decode()
        extension = filename.split(".")[-1].lower()
        if change.blob_id in blobs_handled:
            change.blob_id = blobs_handled[change.blob_id]
        elif extension in filetype_filters.keys():
            # change.blob_id is None for deleted files (ex: change.type=b'D')
            if change.blob_id is None:
                assert change.type == b"D"
                continue
            # Get the old blob content
            cat_file_process.stdin.write(change.blob_id + b"\n")
            cat_file_process.stdin.flush()
            objhash, objtype, objsize = cat_file_process.stdout.readline().split()
            content_plus_newline = cat_file_process.stdout.read(int(objsize) + 1)
            # Reformat into a new blob
            if extension in filetype_filters.keys():
                new_content = filetype_filters[extension](
                    formatter, content_plus_newline, filename
                )
                if new_content is None:
                    continue
            else:
                continue
            # Insert the new file into the filter's stream, and remove the tempfile
            blob = fr.Blob(new_content)
            repo_filter.insert(blob)
            # Record our handling of the blob and use it for this change
            blobs_handled[change.blob_id] = blob.id
            change.blob_id = blob.id


pyproject_option = click.option(
    "--pyproject",
    "pyproject_file",
    default=None,
    help="Path of the pyproject.toml file specifying formatter configurations.",
)
gitattrs_option = click.option(
    "--gitattrs", "gitattrs_file", default=None, help="Location of .gitattributes file"
)
conda_option = click.option(
    "--conda", "conda_env", default=None, help="Name of conda environment to run in"
)
autoupdate_option = click.option(
    "--autoupdate/--no-autoupdate",
    default=True,
    help="Whether to update nbcleanse automatically",
    show_default=True,
)
long_running_option_arg = "--long-running/--no-long-running"
long_running_option_kwargs = dict(
    help="Whether to invoke git filter in long-running process mode (faster).",
    show_default=True,
)
long_running_option = click.option(
    long_running_option_arg, default=True, **long_running_option_kwargs
)
long_running_option_false_by_default = click.option(
    long_running_option_arg, default=False, **long_running_option_kwargs
)


@click.group()
def cli():
    pass


@cli.command()
@pyproject_option
def filter_repo(pyproject_file):
    import git_filter_repo as fr

    if pyproject_file:
        config = load_pyproject_configs(pyproject_file)
    else:
        config = None
    formatter = Formatter(config)
    args = fr.FilteringOptions.default_options()
    args.force = True
    cat_file_process = subprocess.Popen(
        ["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    repo_filter = fr.RepoFilter(args)
    repo_filter._commit_callback = partial(
        filter_commit,
        formatter,
        cat_file_process=cat_file_process,
        repo_filter=repo_filter,
    )
    repo_filter.run()
    cat_file_process.stdin.close()
    cat_file_process.wait()


def _install(
    pyproject_file=None,
    gitattrs_file=None,
    conda_env=None,
    autoupdate=None,
    long_running=True,
):
    """Install the git filter and set the git attributes."""
    try:
        git_dir = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except WindowsError if name == "nt" else OSError:
        click.secho("Installation failed: git is not on path!", err=True, bold=True)
        sys.exit(1)
    except CalledProcessError:
        click.secho("Installation failed: not a git repository!", err=True, bold=True)
        sys.exit(1)
    filter_command = ["'{}'".format(path_to_posix(sys.executable))]
    filter_command.extend(["'{}'".format(Path(__file__).resolve()), "filter"])
    if pyproject_file:
        filter_command.extend(["--pyproject", f"'{pyproject_file}'"])
    if gitattrs_file:
        filter_command.extend(["--gitattrs", f"'{gitattrs_file}'"])
    if conda_env:
        filter_command.extend(["--conda", f"'{conda_env}'"])
    if autoupdate:
        filter_command.extend(["--autoupdate"])
    filter_command = " ".join(filter_command)
    if long_running:
        commands = [
            (["git", "config", "--remove-section", "filter.nbcleanse"], False),
            (["git", "config", "--remove-section", "diff.nbcleanse"], False),
            (
                [
                    "git",
                    "config",
                    "filter.nbcleanse.process",
                    filter_command + " --long-running",
                ],
                True,
            ),
            (["git", "config", "filter.nbcleanse.required", "true"], True),
            (
                ["git", "config", "diff.nbcleanse.textconv", filter_command + " -t"],
                True,
            ),
        ]
    else:
        commands = [
            (["git", "config", "--remove-section", "filter.nbcleanse"], False),
            (["git", "config", "--remove-section", "diff.nbcleanse"], False),
            (
                [
                    "git",
                    "config",
                    "filter.nbcleanse.clean",
                    filter_command + " -e ipynb",
                ],
                True,
            ),
            (["git", "config", "filter.nbcleanse.smudge", "cat"], True),
            (["git", "config", "filter.nbcleanse.required", "true"], True),
            (
                ["git", "config", "diff.nbcleanse.textconv", filter_command + " -t"],
                True,
            ),
        ]
    for command, check in commands:
        if check:
            subprocess.run(command, check=True)
        else:
            # silence stderr
            subprocess.run(command, stderr=subprocess.DEVNULL, check=False)
    if not gitattrs_file:
        gitattrs_file = os.path.join(git_dir, "info", "attributes")
    gitattrs_file = os.path.expanduser(gitattrs_file)

    # Check if there is already a filter for ipynb files
    filter_exists = False
    diff_exists = False
    if os.path.exists(gitattrs_file):
        with open(gitattrs_file, "r") as f:  # TODO
            attrs = f.read()
        filter_exists = "*.ipynb filter" in attrs
        diff_exists = "*.ipynb diff" in attrs
        if filter_exists and diff_exists:
            return

    with open(gitattrs_file, "a") as f:
        # If the file already exists, ensure it ends with a new line
        if f.tell():
            f.write("\n")
        if not filter_exists:
            print("*.ipynb filter=nbcleanse", file=f)
        if not diff_exists:
            print("*.ipynb diff=nbcleanse", file=f)


@cli.command()
@pyproject_option
@gitattrs_option
@conda_option
@autoupdate_option
@long_running_option
def install(pyproject_file, gitattrs_file, conda_env, autoupdate, long_running):
    return _install(
        pyproject_file=pyproject_file,
        gitattrs_file=gitattrs_file,
        conda_env=conda_env,
        autoupdate=autoupdate,
        long_running=long_running,
    )


@cli.command()
@gitattrs_option
def uninstall(gitattrs_file):
    """Uninstall the git filter and unset the git attributes."""
    try:
        git_dir = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except CalledProcessError:
        click.secho("Installation failed: not a git repository!", err=True, bold=True)
        sys.exit(1)
    commands = [
        ["git", "config", "--remove-section", "filter.nbcleanse"],
        ["git", "config", "--remove-section", "diff.nbcleanse"],
    ]
    for command in commands:
        subprocess.run(command, text=True, stdout=DEVNULL, stderr=STDOUT)
    if not gitattrs_file:
        gitattrs_file = os.path.join(git_dir, "info", "attributes")
    # Check if there is a filter for ipynb files
    if os.path.exists(gitattrs_file):
        with open(gitattrs_file, "r+") as f:
            lines = [
                l for l in f if not ("filter=nbcleanse" in l or "diff=nbcleanse" in l)
            ]
            f.seek(0)
            f.write("".join(lines))
            f.truncate()


@cli.command()
def status():
    """Checks whether nbcleanse is installed as a git filter in the current
    repository.
    """
    commands = {
        "clean": ["git", "config", "filter.nbcleanse.clean"],
        "smudge": ["git", "config", "filter.nbcleanse.smudge"],
        "process": ["git", "config", "filter.nbcleanse.process"],
        "required": ["git", "config", "filter.nbcleanse.required"],
        "diff": ["git", "config", "diff.nbcleanse.textconv"],
        "attributes": ["git", "check-attr", "filter", "--", "*.ipynb"],
        "diff_attributes": ["git", "check-attr", "diff", "--", "*.ipynb"],
    }
    info = {}
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            text=True,
            capture_output=True,
            check=True,
        )
        info["git_dir"] = os.path.dirname(os.path.abspath(res.stdout.strip()))
    except:
        click.secho("not in a git repository!", err=True, bold=True)
        sys.exit(1)
    not_installed = False
    for key, args in commands.items():
        res = subprocess.run(args, text=True, capture_output=True, check=False)
        info[key] = res.stdout.strip()
    if not (info["clean"] or info["process"]):
        not_installed = True
    if "attributes" not in info or info["attributes"].endswith("unspecified"):
        not_installed = True
    if not_installed:
        click.echo("nbcleanse is not installed in repository {git_dir}".format(**info))
        sys.exit(1)
    click.echo(
        dedent(
            """\
        nbcleanse is installed in repository {git_dir}

        Filter:
            clean={clean}
            smudge={smudge}
            process={process}
            required={required}
            diff={diff}

        Attributes:
        {attributes}

        Diff attributes:
        {diff_attributes}
        """.format(
                **info
            )
        )
    )


@cli.command()
@click.option(
    "-t",
    "--textconv",
    is_flag=True,
    default=False,
    help="Print filtered files to stdout",
    show_default=True,
)
@click.option(
    "--keep-count",
    is_flag=True,
    default=False,
    help="Do not strip execution count/prompt number",
    show_default=True,
)
@click.option(
    "--keep-output",
    is_flag=True,
    default=False,
    help="Do not strip output",
    show_default=True,
)
@click.option("-s", "--strip-key", multiple=True, help="Strip key from notebook JSON")
@click.option(
    "-e",
    "--extension",
    "default_extension",
    type=click.Choice(list(filetype_filters.keys()), case_sensitive=False),
    default=None,
    help="Interpret input as filetype when filtering. Required if filtering stdin, otherwise will use filename extension for each input file.",
)
@pyproject_option
@gitattrs_option
@conda_option
@autoupdate_option
@long_running_option_false_by_default
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Print exceptions that occur when formatting",
    show_default=True,
)
@click.argument("files", type=click.File("r", lazy=True), nargs=-1)
def filter(
    files,
    textconv,
    keep_count,
    keep_output,
    strip_key,
    default_extension,
    pyproject_file,
    gitattrs_file,
    conda_env,
    autoupdate,
    long_running,
    verbose,
):
    if os.environ.get("NBCLEANSE_VERBOSE"):
        verbose = True
    git_pull_if_needed(
        pyproject_file=pyproject_file,
        gitattrs_file=gitattrs_file,
        conda_env=conda_env,
        autoupdate=autoupdate,
        long_running=long_running,
    )
    if pyproject_file:
        config = load_pyproject_configs(pyproject_file)
    else:
        config = None
    if strip_key:
        extra_keys = strip_key
    else:
        extra_keys = DEFAULT_EXTRA_KEYS
    formatter = Formatter(config)
    if long_running:
        start_filter_server(
            sys.stdin,
            sys.stdout,
            {
                "clean": partial(
                    filter_jupyter,
                    formatter,
                    keep_output=keep_output,
                    keep_count=keep_count,
                    extra_keys=extra_keys,
                )
            },
        )
        if verbose:
            print(file=sys.stderr)
            formatter.print_exceptions(file=sys.stderr)
        sys.exit(0)
    # SEE: https://stackoverflow.com/a/16549381
    if sys.stdin:
        sys.stdin.reconfigure(encoding="utf-8")
        if not files:
            files = [sys.stdin]
            textconv = True
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    if textconv:
        out_file = sys.stdout
    for file in files:
        if not textconv:
            file.mode = "r+"
            out_file = file
        extension = default_extension
        if not extension:
            if file is sys.stdin:
                raise click.error("must specify --extension when filtering stdin")
            else:
                extension = Path(file.name).suffix[1:]
        content = file.read()
        if extension == "ipynb":
            kwargs = dict(
                keep_output=keep_output, keep_count=keep_count, extra_keys=strip_key
            )
        else:
            kwargs = {}
        new_content = filetype_filters[extension](
            formatter, content, file.name, **kwargs
        )
        if new_content is None:
            continue
        if not textconv:
            out_file.seek(0)
        out_file.write(new_content)
        if textconv:
            out_file.flush()
        else:
            out_file.truncate()
    if verbose:
        print(file=sys.stderr)
        formatter.print_exceptions(file=sys.stderr)


def main():
    cli()


if __name__ == "__main__":
    main()
