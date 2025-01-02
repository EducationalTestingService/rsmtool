"""Convert a simple RSMTool model to ONNX."""

from pathlib import Path

import numpy as np
from onnxruntime import InferenceSession


def convert(
    model_file: Path,
    feature_file: Path,
    calibration_file: Path,
    trim_min: float,
    trim_max: float,
    trim_tolerance: float,
    verify_correctness: bool = True,
) -> None:
    """Convert a simple rsmtool model to onnx.

    Parameters
    ----------
    model_file:
        Path to the file containing the SKLL learner.
    feature_file:
        Path to the file containing the feature statistics.
    calibration_file:
        Path to the file containing the label statistics.
    trim_min,trim_max,trim_tolerance:
        Trimming arguments for `fast_predict`.
    verify_correctness:
        Whether to verify that the converted model produces the same output.

    Raises
    ------
    AssertionError
        If an unsupported operation is encountered or the correctness test failed.
    """
    import json

    import pandas as pd
    from skl2onnx import to_onnx
    from skll.learner import Learner

    # load files
    learner = Learner.from_file(model_file)
    feature_information = pd.read_csv(feature_file, index_col=0)
    calibrated_values = json.loads(Path(calibration_file).read_text())

    # validate simplifying assumptions
    assert (feature_information["transform"] == "raw").all(), "Only transform=raw is implemented"
    assert (feature_information["sign"] == 1).all(), "Only sign=1 is implemented"
    assert learner.feat_selector.get_support().all(), "Remove features from df_feature_info"

    # sort features names (FeatureSet does that)
    feature_information = feature_information.sort_values(by="feature")

    # combine calibration values into one transformation
    scale = calibrated_values["human_labels_sd"] / calibrated_values["train_predictions_sd"]
    shift = (
        calibrated_values["human_labels_mean"] - calibrated_values["train_predictions_mean"] * scale
    )

    # export model and statistics
    onnx_model = to_onnx(
        learner.model,
        feature_information["train_mean"].to_numpy().astype(np.float32)[None],
        target_opset=20,
    )
    model_file.with_suffix(".onnx").write_bytes(onnx_model.SerializeToString())

    statistics = {
        "feature_names": feature_information.index.to_list(),
        "feature_outlier_min": (
            feature_information["train_mean"] - 4 * feature_information["train_sd"]
        ).to_list(),
        "feature_outlier_max": (
            feature_information["train_mean"] + 4 * feature_information["train_sd"]
        ).to_list(),
        "feature_means": feature_information["train_transformed_mean"].to_list(),
        "feature_stds": feature_information["train_transformed_sd"].to_list(),
        "label_mean": shift,
        "label_std": scale,
        "label_min": trim_min - trim_tolerance,
        "label_max": trim_max + trim_tolerance,
    }
    (model_file.parent / f"{model_file.with_suffix('').name}_statistics.json").write_text(
        json.dumps(statistics)
    )

    if not verify_correctness:
        return

    # verify that the converted model produces the same output
    from time import time

    from rsmtool import fast_predict
    from rsmtool.modeler import Modeler

    onnx_model = InferenceSession(model_file.with_suffix(".onnx"))
    rsm_model = Modeler.load_from_learner(learner)
    onnx_duration = 0
    rsm_duration = 0
    iterations = 1_000
    for _ in range(iterations):
        # sample random input data
        features = (
            feature_information["train_mean"]
            + (np.random.rand(feature_information.shape[0]) - 0.5)
            * 10
            * feature_information["train_sd"]
        ).to_dict()

        start = time()
        onnx_prediction = predict(features, model=onnx_model, statistics=statistics)
        onnx_duration += time() - start

        start = time()
        rsm_prediction = fast_predict(
            features,
            modeler=rsm_model,
            df_feature_info=feature_information,
            trim=True,
            trim_min=trim_min,
            trim_max=trim_max,
            trim_tolerance=trim_tolerance,
            scale=True,
            train_predictions_mean=calibrated_values["train_predictions_mean"],
            train_predictions_sd=calibrated_values["train_predictions_sd"],
            h1_mean=calibrated_values["human_labels_mean"],
            h1_sd=calibrated_values["human_labels_sd"],
        )["scale_trim"]
        rsm_duration += time() - start

        assert np.isclose(onnx_prediction, rsm_prediction), f"{onnx_prediction} vs {rsm_prediction}"

    print(f"ONNX duration: {round(onnx_duration/iterations, 5)}")
    print(f"RSMTool duration: {round(rsm_duration/iterations, 5)}")


def predict(
    features: dict[str, float],
    *,
    model: InferenceSession,
    statistics: dict[str, np.ndarray | float | list[str]],
) -> float:
    """Make a single prediction with the convered ONNX model.

    Parameters
    ----------
    features:
        Dictionary of the input features.
    model:
        ONNX inference session of the converted model.
    statistics:
        Dictionary containing the feature and label statistics.

    Returns
    -------
        A single prediction.
    """
    # get features in the expected order
    features = np.array([features[name] for name in statistics["feature_names"]])

    # clip outliers
    features = np.clip(
        features, a_min=statistics["feature_outlier_min"], a_max=statistics["feature_outlier_max"]
    )

    # normalize
    features = (features - statistics["feature_means"]) / statistics["feature_stds"]

    # predict
    prediction = model.run(None, {"X": features[None].astype(np.float32)})[0].item()

    # transform to human scale
    prediction = prediction * statistics["label_std"] + statistics["label_mean"]

    # trim prediction
    return np.clip(prediction, a_min=statistics["label_min"], a_max=statistics["label_max"])


if __name__ == "__main__":
    convert(
        Path("test.model"),
        Path("features.csv"),
        Path("calibrated_values.json"),
        trim_min=1,
        trim_max=3,
        trim_tolerance=0.49998,
    )
