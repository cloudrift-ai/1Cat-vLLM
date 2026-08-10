# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

import pytest

from vllm import platforms
from vllm.entrypoints.cli.main import main
from vllm.platforms.cpu import CpuPlatform
from vllm.version import __version__ as VLLM_VERSION


def test_cli_version_uses_package_version(monkeypatch, capsys):
    monkeypatch.setattr(platforms, "current_platform", CpuPlatform())
    monkeypatch.setattr(sys, "argv", ["vllm", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == VLLM_VERSION
