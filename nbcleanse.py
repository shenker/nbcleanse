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

import subprocess
from subprocess import CalledProcessError, STDOUT, DEVNULL
from pathlib import Path, PurePath
import sys
import os
import re
import time
from functools import partial
from textwrap import dedent
import click
import nbformat
import black
import docformatter
import cachetools

PARENT_DIR = Path(__file__).resolve().parent
TIMESTAMP_FILE = PARENT_DIR / ".last_updated"
UPDATE_INTERVAL = 5 * 60  # seconds

blobs_handled = {}
black_cache = cachetools.LRUCache(maxsize=10_000_000, getsizeof=sys.getsizeof)


def path_to_posix(path):
    return str(PurePath(path).as_posix())


def is_git_pull_needed():
    # FROM: https://stackoverflow.com/questions/3258243/check-if-pull-needed-in-git
    commands = {
        "local": ["git", "rev-parse", "@{0}"],
        "remote": ["git", "rev-parse", "@{u}"],
        "base": ["git", "merge-base", "@{0}", "@{u}"],
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


def git_pull_if_needed(conda_env=None):
    if not is_update_needed():
        return
    pulled = is_git_pull_needed()
    if pulled:
        click.secho(
            "nbcleanse update available, running git pull...", err=True, bold=True
        )
        subprocess.run(["git", "pull"], cwd=PARENT_DIR, check=True)
        if conda_env:
            click.echo(
                click.style("updating nbcleanse conda environment '", bold=True)
                + conda_env
                + click.style("' (if necessary)...", bold=True),
                err=True,
            )
            envyml = PARENT_DIR / "environment.yml"
            subprocess.run(
                ["conda", "env", "update", "--prune", "-n", conda_env, "-f", envyml],
                cwd=PARENT_DIR,
                check=True,
            )
        click.secho("reinstalling nbcleanse...", err=True, bold=True)
        _install(conda_env=conda_env)
    now = time.time()
    with open(TIMESTAMP_FILE, "w") as f:
        f.write(f"{now}\n")
    return pulled


@cachetools.cached(black_cache)
def blacken(contents, format_docstrings=True):
    try:
        if format_docstrings:
            new_contents = docformatter.format_code(
                contents, summary_wrap_length=79, description_wrap_length=72
            )
        new_contents = black.format_file_contents(
            contents,
            fast=True,
            mode=black.FileMode(
                line_length=88, target_versions={black.TargetVersion.PY38}
            ),
        )
    except black.NothingChanged:
        return contents
    except black.InvalidInput:
        return None
    except:  # once got a blib2to3.pgen2.tokenize.TokenError
        return None
    return new_contents


def filter_py(contents, filename):
    new_contents = blacken(contents.decode())
    if new_contents is None:
        print(f"\nUnable to format {filename}")
        return None
    return new_contents.encode()


def pop_recursive(d, key, default=None):
    """dict.pop(key) where `key` is a `.`-delimited list of nested keys.
    >>> d = {'a': {'b': 1, 'c': 2}}
    >>> pop_recursive(d, 'a.c')
    2
    >>> d
    {'a': {'b': 1}}
    """
    nested = key.split(".")
    current = d
    for k in nested[:-1]:
        if hasattr(current, "get"):
            current = current.get(k, {})
        else:
            return default
    if not hasattr(current, "pop"):
        return default
    return current.pop(nested[-1], default)


def _cells(nb):
    """Yield all cells in an nbformat-insensitive manner"""
    if nb.nbformat < 4:
        for ws in nb.worksheets:
            for cell in ws.cells:
                yield cell
    else:
        for cell in nb.cells:
            yield cell


def strip_jupyter(
    nb, keep_output=False, keep_count=False, extra_keys=(), filename=None
):
    """Strip the outputs, execution count/prompt number and miscellaneous
    metadata from a notebook object, unless specified to keep either the outputs
    or counts.
    `extra_keys` could be `('metadata.foo', 'cell.metadata.bar', 'metadata.baz')`
    """
    if keep_output is None and "keep_output" in nb.metadata:
        keep_output = bool(nb.metadata["keep_output"])
    keys = {"metadata": [], "cell": {"metadata": []}}
    for key in extra_keys:
        if key.startswith("metadata."):
            keys["metadata"].append(key[len("metadata.") :])
        elif key.startswith("cell.metadata."):
            keys["cell"]["metadata"].append(key[len("cell.metadata.") :])
        else:
            print(f"ignoring extra key {key}", file=sys.stderr)
    for field in keys["metadata"]:
        pop_recursive(nb.metadata, field)
    failed_cells = 0
    for cell in _cells(nb):
        if cell["cell_type"] == "code":
            # Remove the prompt_number/execution_count, unless directed otherwise
            if not keep_count:
                cell.pop("prompt_number", None)
                cell["execution_count"] = None
            source = cell["source"]
            source = re.sub("^%", "#%#", source, flags=re.M)
            new_source = blacken(source)
            if new_source is not None:
                new_source = new_source.rstrip()
                new_source = re.sub("^#%#", "%", new_source, flags=re.M)
                cell["source"] = new_source
            else:
                failed_cells += 1
        keep_output_this_cell = keep_output
        # Keep the output for these cells, but strip count and metadata
        if "keep_output" in cell.metadata:
            keep_output_this_cell = bool(cell.metadata["keep_output"])
        # Remove the outputs, unless directed otherwise
        if "outputs" in cell:
            # Default behavior strips outputs. With all outputs stripped,
            # there are no counts to keep and keep_count is ignored.
            if not keep_output_this_cell:
                cell["outputs"] = []
            # If keep_output_this_cell, but not keep_count, strip the counts
            # from the output.
            if keep_output_this_cell and not keep_count:
                for output in cell["outputs"]:
                    if "execution_count" in output:
                        output["execution_count"] = None
            # If keep_output_this_cell and keep_count, do nothing.
        # Always remove this metadata
        if "metadata" in cell:
            for field in ["collapsed", "scrolled"]:
                cell.metadata.pop(field, None)
        for (extra, fields) in keys["cell"].items():
            if extra in cell:
                for field in fields:
                    pop_recursive(getattr(cell, extra), field)
    if failed_cells:
        err_msg = f"\nFailed to format {failed_cells} cells in notebook"
        if filename:
            err_msg += f" {filename}"
        print(err_msg, file=sys.stderr)
    return nb


def filter_jupyter(contents, filename):
    try:
        nb = nbformat.reads(contents.decode(), nbformat.NO_CONVERT)
    except:
        print(f"\nUnable to parse notebook {filename}")
        return None
    nb = strip_jupyter(nb, filename=filename)
    new_contents = nbformat.writes(nb) + "\n"
    return new_contents.encode()


filetype_filters = {"py": filter_py, "ipynb": filter_jupyter}


def filter_commit(commit, metadata, cat_file_process=None, repo_filter=None):
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
            # Get the old blob contents
            cat_file_process.stdin.write(change.blob_id + b"\n")
            cat_file_process.stdin.flush()
            objhash, objtype, objsize = cat_file_process.stdout.readline().split()
            contents_plus_newline = cat_file_process.stdout.read(int(objsize) + 1)
            # Reformat into a new blob
            if extension in filetype_filters.keys():
                new_contents = filetype_filters[extension](
                    contents_plus_newline, filename
                )
                if new_contents is None:
                    continue
            else:
                continue
            # Insert the new file into the filter's stream, and remove the tempfile
            blob = fr.Blob(new_contents)
            repo_filter.insert(blob)
            # Record our handling of the blob and use it for this change
            blobs_handled[change.blob_id] = blob.id
            change.blob_id = blob.id


gitattrs_option = click.option(
    "--gitattrs", default=None, help="Location of .gitattributes file"
)
conda_option = click.option(
    "--conda", "conda_env", default=None, help="Name of conda environment to run in"
)
autoupdate_option = click.option(
    "--autoupdate/--no-autoupdate",
    default=True,
    help="Whether to update nbcleanse automatically",
)


@click.group()
def cli():
    pass


@cli.command()
def filter_repo():
    import git_filter_repo as fr

    args = fr.FilteringOptions.default_options()
    args.force = True
    cat_file_process = subprocess.Popen(
        ["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    repo_filter = fr.RepoFilter(args)
    repo_filter._commit_callback = partial(
        filter_commit, cat_file_process=cat_file_process, repo_filter=repo_filter
    )
    repo_filter.run()
    cat_file_process.stdin.close()
    cat_file_process.wait()


def _install(gitattrs=None, conda_env=None, autoupdate=None):
    """Install the git filter and set the git attributes."""
    try:
        git_dir = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (WindowsError if name == "nt" else OSError):
        print("Installation failed: git is not on path!", file=sys.stderr)
        sys.exit(1)
    except CalledProcessError:
        print("Installation failed: not a git repository!", file=sys.stderr)
        sys.exit(1)
    if conda_env:
        if not os.environ["CONDA_EXE"]:
            click.secho(
                "$CONDA_EXE not set, cannot install with --conda set",
                err=True,
                bold=True,
            )
            sys.exit(1)
        filter_command = [
            "'{}'".format(path_to_posix(os.environ["CONDA_EXE"])),
            "run",
            "-n",
            f"'{conda_env}'",
        ]
    else:
        filter_command = ["'{}'".format(path_to_posix(sys.executable))]
    filter_command.extend(["'{}'".format(Path(__file__).resolve()), "filter"])
    if gitattrs:
        filter_command.extend(["--gitattrs", f"'{gitattrs}'"])
    if conda_env:
        filter_command.extend(["--conda", f"'{conda_env}'"])
    if autoupdate:
        filter_command.extend(["--autoupdate"])
    filter_command = " ".join(filter_command)
    commands = [
        ["git", "config", "filter.nbcleanse.clean", filter_command],
        ["git", "config", "filter.nbcleanse.smudge", "cat"],
        ["git", "config", "diff.ipynb.textconv", filter_command + " -t"],
    ]
    for command in commands:
        subprocess.run(command, check=True)

    if not gitattrs:
        gitattrs = os.path.join(git_dir, "info", "attributes")
    gitattrs = os.path.expanduser(gitattrs)

    # Check if there is already a filter for ipynb files
    filter_exists = False
    diff_exists = False
    if os.path.exists(gitattrs):
        with open(gitattrs, "r") as f:  # TODO
            attrs = f.read()
        filter_exists = "*.ipynb filter" in attrs
        diff_exists = "*.ipynb diff" in attrs
        if filter_exists and diff_exists:
            return

    with open(gitattrs, "a") as f:
        # If the file already exists, ensure it ends with a new line
        if f.tell():
            f.write("\n")
        if not filter_exists:
            print("*.ipynb filter=nbcleanse", file=f)
        if not diff_exists:
            print("*.ipynb diff=ipynb", file=f)


@cli.command()
@gitattrs_option
@conda_option
@autoupdate_option
def install(gitattrs, conda_env, autoupdate):
    return _install(gitattrs=gitattrs, conda_env=conda_env, autoupdate=autoupdate)


@cli.command()
@click.option("--gitattrs", default=None, help="Location of .gitattributes file")
def uninstall(gitattrs):
    """Uninstall the git filter and unset the git attributes."""
    try:
        git_dir = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except CalledProcessError:
        print("Installation failed: not a git repository!", file=sys.stderr)
        sys.exit(1)
    commands = [
        ["git", "config", "--unset", "filter.nbcleanse.clean"],
        ["git", "config", "--unset", "filter.nbcleanse.smudge"],
        ["git", "config", "--remove-section", "diff.ipynb"],
    ]
    for command in commands:
        subprocess.run(command, stdout=DEVNULL, stderr=STDOUT)
    if not gitattrs:
        gitattrs = os.path.join(git_dir, "info", "attributes")
    # Check if there is a filter for ipynb files
    if os.path.exists(gitattrs):
        with open(gitattrs, "r+") as f:
            lines = [
                l
                for l in f
                if not (l.startswith("*.ipynb filter") or l.startswith("*.ipynb diff"))
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
        "diff": ["git", "config", "diff.ipynb.textconv"],
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
        print("not in a git repository!", file=sys.stderr)
        sys.exit(1)
    try:
        for key, args in commands.items():
            res = subprocess.run(args, text=True, capture_output=True, check=True)
            info[key] = res.stdout.strip()
    except CalledProcessError:
        if "git_dir" in info:
            print("nbcleanse is not installed in repository {git_dir}".format(**info))
        else:
            print("could not find git repository", file=sys.stderr)
        sys.exit(1)
    if info["attributes"].endswith("unspecified"):
        print("nbcleanse is not installed in repository {git_dir}".format(**info))
        sys.exit(1)
    print(
        dedent(
            """\
        nbcleanse is installed in repository {git_dir}

        Filter:
            clean={clean}
            smudge={smudge}
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
)
@click.option(
    "--keep-count",
    is_flag=True,
    default=False,
    help="Do not strip execution count/prompt number",
)
@click.option("--keep-output", is_flag=True, default=False, help="Do not strip output")
@click.option("-s", "--strip-key", multiple=True, help="Strip key from notebook JSON")
@gitattrs_option
@conda_option
@autoupdate_option
@click.argument("files", type=click.File("r", lazy=True), nargs=-1)
def filter(files, textconv, keep_count, keep_output, strip_key, conda_env, autoupdate):
    git_pull_if_needed(gitattrs=gitattrs, conda_env=conda_env, autoupdate=autoupdate)
    # https://stackoverflow.com/a/16549381
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
        try:
            nb = nbformat.read(file, as_version=nbformat.NO_CONVERT)
            nb = strip_jupyter(
                nb, keep_output, keep_count, extra_keys=strip_key, filename=file.name
            )
            if not textconv:
                out_file.seek(0)
            nbformat.write(nb, out_file)
            if textconv:
                out_file.flush()
            else:
                out_file.truncate()
        # except nbformat.NotJSONError:
        #     print(f"Not a valid notebook: '{file.name}'", file=sys.stderr)
        #     sys.exit(1)
        except Exception:
            # Ignore exceptions for non-notebook files.
            print(f"Could not strip '{file.name}'", file=sys.stderr)
            raise


def main():
    cli()


if __name__ == "__main__":
    main()
