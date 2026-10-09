from __future__ import annotations

import importlib
import types

import discoursekit


def test_public_api_is_importable() -> None:
    for name in discoursekit.__all__:
        assert hasattr(discoursekit, name), name


def test_version_is_a_non_empty_string() -> None:
    assert isinstance(discoursekit.__version__, str)
    assert discoursekit.__version__


def test_no_public_name_hides_a_module() -> None:
    # A function exported under the name of the module that defines it would
    # make "import discoursekit.<name>" return the function.
    for name in ("embedding", "lexical", "semantic", "tokenization"):
        module = importlib.import_module(f"discoursekit.{name}")
        assert isinstance(module, types.ModuleType)
        assert getattr(discoursekit, name) is module
        assert name not in discoursekit.__all__
