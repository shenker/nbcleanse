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
import sys
import re
import json
from textwrap import dedent
import click
import nbformat
import black
import cachetools

blobs_handled = {}
cat_file_process = None
black_cache = cachetools.LRUCache(maxsize=10_000_000, getsizeof=sys.getsizeof)


@click.group()
def cli():
    pass


@cachetools.cached(black_cache)
def blacken(contents):
    try:
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
    nested = key.split('.')
    current = d
    for k in nested[:-1]:
        if hasattr(current, 'get'):
            current = current.get(k, {})
        else:
            return default
    if not hasattr(current, 'pop'):
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
    """
    Strip the outputs, execution count/prompt number and miscellaneous
    metadata from a notebook object, unless specified to keep either the outputs
    or counts.
    `extra_keys` could be `('metadata.foo', 'cell.metadata.bar', 'metadata.baz')`
    """
    if keep_output is None and 'keep_output' in nb.metadata:
        keep_output = bool(nb.metadata['keep_output'])
    keys = {'metadata': [], 'cell': {'metadata': []}}
    for key in extra_keys:
        if key.startswith('metadata.'):
            keys['metadata'].append(key[len('metadata.') :])
        elif key.startswith('cell.metadata.'):
            keys['cell']['metadata'].append(key[len('cell.metadata.') :])
        else:
            print(f"ignoring extra key {key}", file=sys.stderr)
    for field in keys['metadata']:
        pop_recursive(nb.metadata, field)
    failed_cells = 0
    for cell in _cells(nb):
        if cell['cell_type'] == 'code':
            # Remove the prompt_number/execution_count, unless directed otherwise
            if not keep_count:
                cell.pop('prompt_number', None)
                cell['execution_count'] = None
            source = cell['source']
            source = re.sub('^%', '#%#', source, flags=re.M)
            new_source = blacken(source)
            if new_source is not None:
                new_source = new_source.rstrip()
                new_source = re.sub('^#%#', '%', new_source, flags=re.M)
                cell['source'] = new_source
            else:
                failed_cells += 1
        keep_output_this_cell = keep_output
        # Keep the output for these cells, but strip count and metadata
        if 'keep_output' in cell.metadata:
            keep_output_this_cell = bool(cell.metadata['keep_output'])
        # Remove the outputs, unless directed otherwise
        if 'outputs' in cell:
            # Default behavior strips outputs. With all outputs stripped,
            # there are no counts to keep and keep_count is ignored.
            if not keep_output_this_cell:
                cell['outputs'] = []
            # If keep_output_this_cell, but not keep_count, strip the counts
            # from the output.
            if keep_output_this_cell and not keep_count:
                for output in cell['outputs']:
                    if 'execution_count' in output:
                        output['execution_count'] = None
            # If keep_output_this_cell and keep_count, do nothing.
        # Always remove this metadata
        if 'metadata' in cell:
            for field in ['collapsed', 'scrolled']:
                cell.metadata.pop(field, None)
        for (extra, fields) in keys['cell'].items():
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
        # nb = json.loads(contents.decode())
    except:
        print(f"\nUnable to parse notebook {filename}")
        return None
    nb = strip_jupyter(nb, filename=filename)
    new_contents = nbformat.writes(nb) + "\n"
    # new_contents = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False)
    return new_contents.encode()


filetype_filters = {'py': filter_py, 'ipynb': filter_jupyter}


def filter_commit(commit, metadata):
    for change in commit.file_changes:
        filename = change.filename.decode()
        extension = filename.split('.')[-1].lower()
        if change.blob_id in blobs_handled:
            change.blob_id = blobs_handled[change.blob_id]
        elif extension in filetype_filters.keys():
            # change.blob_id is None for deleted files (ex: change.type=b'D')
            if change.blob_id is None:
                assert change.type == b'D'
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


@click.command()
def filter_repo():
    import git_filter_repo as fr

    args = fr.FilteringOptions.default_options()
    args.force = True
    cat_file_process = subprocess.Popen(
        ['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    repo_filter = fr.RepoFilter(args, commit_callback=filter_commit)
    repo_filter.run()
    cat_file_process.stdin.close()
    cat_file_process.wait()


@cli.command()
@click.option('--gitattrs', default=None, help='Location of .gitattributes file')
def install(gitattrs=None):
    """Install the git filter and set the git attributes."""
    from os import name, path
    from subprocess import check_call, check_output, CalledProcessError

    try:
        git_dir = check_output(['git', 'rev-parse', '--git-dir']).strip()
    except (WindowsError if name == 'nt' else OSError):
        print('Installation failed: git is not on path!', file=sys.stderr)
        sys.exit(1)
    except CalledProcessError:
        print('Installation failed: not a git repository!', file=sys.stderr)
        sys.exit(1)
    filepath = "'{}' '{}' filter".format(
        sys.executable.replace('\\', '/'), path.abspath(__file__)
    )
    check_call(['git', 'config', 'filter.nbcleanse.clean', filepath])
    check_call(['git', 'config', 'filter.nbcleanse.smudge', 'cat'])
    check_call(['git', 'config', 'diff.ipynb.textconv', filepath + ' -t'])

    if not gitattrs:
        gitattrs = path.join(git_dir.decode(), 'info', 'attributes')
    gitattrs = path.expanduser(gitattrs)

    # Check if there is already a filter for ipynb files
    filter_exists = False
    diff_exists = False
    if path.exists(gitattrs):
        with open(gitattrs, 'r') as f:
            attrs = f.read()
        filter_exists = '*.ipynb filter' in attrs
        diff_exists = '*.ipynb diff' in attrs
        if filter_exists and diff_exists:
            return

    with open(gitattrs, 'a') as f:
        # If the file already exists, ensure it ends with a new line
        if f.tell():
            f.write("\n")
        if not filter_exists:
            print("*.ipynb filter=nbcleanse", file=f)
        if not diff_exists:
            print("*.ipynb diff=ipynb", file=f)


@cli.command()
@click.option('--gitattrs', default=None, help='Location of .gitattributes file')
def uninstall(gitattrs):
    """Uninstall the git filter and unset the git attributes."""
    from os import devnull, path
    from subprocess import call, check_output, CalledProcessError, STDOUT, DEVNULL

    try:
        git_dir = check_output(['git', 'rev-parse', '--git-dir']).strip()
    except CalledProcessError:
        print('Installation failed: not a git repository!', file=sys.stderr)
        sys.exit(1)
    call(
        ['git', 'config', '--unset', 'filter.nbcleanse.clean'],
        stdout=DEVNULL,
        stderr=STDOUT,
    )
    call(
        ['git', 'config', '--unset', 'filter.nbcleanse.smudge'],
        stdout=DEVNULL,
        stderr=STDOUT,
    )
    call(
        ['git', 'config', '--remove-section', 'diff.ipynb'],
        stdout=DEVNULL,
        stderr=STDOUT,
    )
    if not gitattrs:
        gitattrs = path.join(git_dir.decode(), 'info', 'attributes')
    # Check if there is a filter for ipynb files
    if path.exists(gitattrs):
        with open(gitattrs, 'r+') as f:
            lines = [
                l
                for l in f
                if not (l.startswith('*.ipynb filter') or l.startswith('*.ipynb diff'))
            ]
            f.seek(0)
            f.write(''.join(lines))
            f.truncate()


@cli.command()
def status():
    """Checks whether nbcleanse is installed as a git filter in the current
    repository.
    """
    from os import path
    from subprocess import check_output, CalledProcessError

    try:
        git_dir = path.dirname(
            path.abspath(
                check_output(['git', 'rev-parse', '--git-dir']).strip().decode()
            )
        )
        clean = (
            check_output(['git', 'config', 'filter.nbcleanse.clean']).decode().strip()
        )
        smudge = (
            check_output(['git', 'config', 'filter.nbcleanse.smudge']).decode().strip()
        )
        diff = check_output(['git', 'config', 'diff.ipynb.textconv']).decode().strip()
        attributes = (
            check_output(['git', 'check-attr', 'filter', '--', '*.ipynb'])
            .decode()
            .strip()
        )
        diff_attributes = (
            check_output(['git', 'check-attr', 'diff', '--', '*.ipynb'])
            .decode()
            .strip()
        )
        if attributes.endswith('unspecified'):
            print(f"nbcleanse is not installed in repository {git_dir}")
            sys.exit(1)
        print(
            dedent(
                f"""\
            nbcleanse is installed in repository {git_dir}

            Filter:
                clean={clean}
                smudge={smudge}
                diff={diff}

            Attributes:
            {attributes}

            Diff attributes:
            {diff_attributes}
            """
            )
        )
    except CalledProcessError:
        print(f"nbcleanse is not installed in repository {git_dir}")
        sys.exit(1)


@cli.command()
@click.option(
    '-t',
    '--textconv',
    is_flag=True,
    default=False,
    help='Print filtered files to stdout',
)
@click.option(
    '--keep-count',
    is_flag=True,
    default=False,
    help='Do not strip execution count/prompt number',
)
@click.option('--keep-output', is_flag=True, default=False, help='Do not strip output')
@click.option('-s', '--strip-key', multiple=True, help='Strip key from notebook JSON')
@click.argument('files', type=click.File('r', lazy=True), nargs=-1)
def filter(files, textconv, keep_count, keep_output, strip_key):
    # https://stackoverflow.com/a/16549381
    if sys.stdin:
        sys.stdin.reconfigure(encoding='utf-8')
        if not files:
            files = [sys.stdin]
            textconv = True
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    if textconv:
        out_file = sys.stdout

    for file in files:
        if not textconv:
            file.mode = 'r+'
            out_file = file
        try:
            nb = nbformat.read(file, as_version=nbformat.NO_CONVERT)
            nb = strip_jupyter(
                nb, keep_output, keep_count, strip_key, filename=file.name
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


if __name__ == '__main__':
    main()
