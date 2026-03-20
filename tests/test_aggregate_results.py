import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from lnai.experiments.aggregate_results import _read_table


def test_read_table_supports_csv_and_lowercases_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("Model,MAE\nInformer,0.1\n")

    table = _read_table(csv_path)

    assert list(table.columns) == ["model", "mae"]
    assert table.loc[0, "model"] == "Informer"


def test_read_table_supports_json_objects(tmp_path: Path) -> None:
    json_path = tmp_path / "metrics.json"
    json_path.write_text(json.dumps({"Model": "CNN", "RMSE": 0.2}))

    table = _read_table(json_path)

    assert list(table.columns) == ["model", "rmse"]
    assert table.loc[0, "rmse"] == 0.2
