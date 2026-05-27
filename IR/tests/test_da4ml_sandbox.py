import importlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_da4ml_sandbox_import_is_lazy():
    sys.modules.pop("IR.da4ml_sandbox", None)

    module = importlib.import_module("IR.da4ml_sandbox")

    assert callable(module.inspect_dais_replay)
