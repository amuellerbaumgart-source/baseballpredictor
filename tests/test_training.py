import joblib
import pandas as pd

from src.models.features import EARLY_FEATURE_COLUMNS
from src.models.trainer import train_model


def _training_frame(rows=40):
    values = {column: [float(index % 3) for index in range(rows)] for column in EARLY_FEATURE_COLUMNS}
    values["home_win"] = [index % 2 for index in range(rows)]
    values["game_pk"] = list(range(rows))
    values["game_datetime"] = pd.date_range("2025-04-01", periods=rows, freq="D").astype(str)
    return pd.DataFrame(values)


def test_training_does_not_replace_better_existing_artifact(tmp_path):
    artifact_path = tmp_path / "model.joblib"
    joblib.dump({"version": "better-existing-model", "evaluation": {"log_loss": 0.0}}, artifact_path)

    train_model(_training_frame(), str(artifact_path))

    assert joblib.load(artifact_path)["version"] == "better-existing-model"
