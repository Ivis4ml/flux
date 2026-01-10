"""
MkDocs hook for generating API documentation.

This hook can be used to auto-generate API documentation from source code.
Currently a placeholder for future implementation.
"""


def on_pre_build(config):
    """Called before the build starts."""
    pass


def on_files(files, config):
    """Called after files are collected."""
    return files


def on_page_markdown(markdown, page, config, files):
    """Called on each page's markdown."""
    return markdown
