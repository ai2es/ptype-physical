import logging
import yaml
import shutil
import os
import warnings
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

from ptype.reliability import (
    compute_calibration,
    reliability_diagram,
    reliability_diagrams,
)
from ptype.plotting import (
    plot_confusion_matrix,
    coverage_figures,
    labels_video,
    video,
)
from ptype.data import load_ptype_data_day, preprocess_data
from evml.keras.models import calc_prob_uncertainty

from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
from hagelslag.evaluation.MetricPlotter import roc_curve, performance_diagram

from collections import OrderedDict, defaultdict


warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def evaluate(conf, reevaluate=False):
    input_features = (
        conf["TEMP_C"] + conf["T_DEWPOINT_C"] + conf["UGRD_m/s"] + conf["VGRD_m/s"]
    )
    output_features = conf["ptypes"]
    save_loc = conf["save_loc"]
    labels = ["rain", "snow", "sleet", "frz-rain"]
    sym_colors = ["blue", "grey", "red", "purple"]
    symbols = ["s", "o", "v", "^"]
    if reevaluate:
        if conf["model"]["loss"] == "dirichlet":
            use_uncertainty = True
        else:
            use_uncertainty = False
        data = load_ptype_data_day(conf, data_split=0, verbose=1)
        scaled_data, scalers = preprocess_data(
            data,
            input_features,
            output_features,
            scaler_type="standard",
            encoder_type="onehot",
        )
        mlp = tf.keras.models.load_model(os.path.join(save_loc, "model"), compile=False)
        for name in data.keys():
            x = scaled_data[f"{name}_x"]
            pred_probs = mlp.predict(x)
            if use_uncertainty:
                pred_probs, u, ale, epi = calc_prob_uncertainty(pred_probs)
                pred_probs = pred_probs.numpy()
                u = u.numpy()
                ale = ale.numpy()
                epi = epi.numpy()
            true_labels = np.argmax(data[name][output_features].to_numpy(), 1)
            pred_labels = np.argmax(pred_probs, 1)
            confidences = np.take_along_axis(pred_probs, pred_labels[:, None], axis=1)
            data[name]["true_label"] = true_labels
            data[name]["pred_label"] = pred_labels
            data[name]["pred_conf"] = confidences
            for k in range(pred_probs.shape[-1]):
                data[name][f"pred_conf{k+1}"] = pred_probs[:, k]
            if use_uncertainty:
                data[name]["evidential"] = u
                data[name]["aleatoric"] = np.take_along_axis(
                    ale, pred_labels[:, None], axis=1
                )
                data[name]["epistemic"] = np.take_along_axis(
                    epi, pred_labels[:, None], axis=1
                )
            data[name].to_parquet(os.path.join(save_loc, f"{name}.parquet"))
    else:
        data = {
            name: pd.read_parquet(os.path.join(save_loc, f"{name}.parquet"))
            for name in ["train", "val", "test"]
        }

    # Compute categorical metrics
    metrics = defaultdict(list)
    for name in data.keys():
        outs = precision_recall_fscore_support(
            data[name]["true_label"].values,
            data[name]["pred_label"].values,
            average=None,
            labels=range(len(output_features)),
        )
        metrics["split"].append(name)
        for i, (p, r, f, s) in enumerate(zip(*list(outs))):
            class_name = output_features[i]
            metrics[f"{class_name}_precision"].append(p)
            metrics[f"{class_name}_recall"].append(r)
            metrics[f"{class_name}_f1"].append(f)
            metrics[f"{class_name}_support"].append(s)

    # Confusion matrix
    plot_confusion_matrix(
        data,
        labels,
        normalize=True,
        save_location=os.path.join(save_loc, "plots", "confusion_matrices.pdf"),
    )

    # Reliability
    metric_keys = [
        "avg_accuracy",
        "avg_confidence",
        "expected_calibration_error",
        "max_calibration_error",
    ]
    for name in data.keys():
        # Calibration stats
        results_calibration = compute_calibration(
            data[name]["true_label"].values,
            data[name]["pred_label"].values,
            data[name]["pred_conf"].values,
            num_bins=10,
        )
        for key in metric_keys:
            metrics[f"bulk_{key}"].append(results_calibration[key])
        # Bulk
        _ = reliability_diagram(
            data[name]["true_label"].values,
            data[name]["pred_label"].values,
            data[name]["pred_conf"].values,
            num_bins=10,
            dpi=300,
            return_fig=True,
        )
        fn = os.path.join(save_loc, "plots", f"bulk_reliability_{name}.pdf")
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        # Class by class
        results = OrderedDict()
        for label in range(len(output_features)):
            cond = data[name]["true_label"] == label
            results[output_features[label]] = {
                "true_labels": data[name][cond]["true_label"].values,
                "pred_labels": data[name][cond]["pred_label"].values,
                "confidences": data[name][cond]["pred_conf"].values,
            }
            results_calibration = compute_calibration(
                results[output_features[label]]["true_labels"],
                results[output_features[label]]["pred_labels"],
                results[output_features[label]]["confidences"],
                num_bins=10,
            )
            for key in metric_keys:
                metrics[f"{output_features[label]}_{key}"].append(
                    results_calibration[key]
                )

        _ = reliability_diagrams(
            results,
            num_bins=10,
            draw_bin_importance="alpha",
            num_cols=2,
            dpi=100,
            return_fig=True,
        )
        fn = os.path.join(save_loc, "plots", f"class_reliability_{name}.pdf")
        plt.savefig(fn, dpi=300, bbox_inches="tight")

    # Hagelslag
    for name in data.keys():
        rocs = []
        for i in range(len(output_features)):
            forecasts = data[name]["pred_conf"]
            obs = np.where(data[name]["true_label"] == i, 1, 0)
            roc = DistributedROC(
                thresholds=np.arange(0.0, 1.01, 0.01), obs_threshold=0.5
            )
            roc.update(forecasts, obs)
            rocs.append(roc)
            metrics[f"{output_features[i]}_auc"].append(roc.auc())
            metrics[f"{output_features[i]}_csi"].append(roc.max_csi())
        roc_curve(
            rocs,
            labels,
            sym_colors,
            symbols,
            os.path.join(save_loc, "plots", f"roc_curve_{name}.pdf"),
        )
        performance_diagram(
            rocs,
            labels,
            sym_colors,
            symbols,
            os.path.join(save_loc, "plots", f"performance_{name}.pdf"),
        )
    # Sorting curves
    for name in data.keys():
        coverage_figures(
            data[name],
            output_features,
            colors=sym_colors,
            save_location=os.path.join(save_loc, "plots", f"coverage_{name}.pdf"),
        )
    # Save metrics
    pd.DataFrame.from_dict(metrics).to_csv(
        os.path.join(save_loc, "metrics", "performance.csv")
    )
    #
    # # CONUS plots for test cases
    # for case_study, dates in conf["case_studies"].items():
    #     print(dates,)
    #     if data["test"]["day"].isin(dates[0:1]).sum():
    #         labels_video(
    #             data["test"],
    #             dates,
    #             "mping",
    #             "true_label",
    #             "True label",
    #             os.path.join(save_loc, "cases", f"{case_study}_true_label.gif"),
    #         )
    #         labels_video(
    #             data["test"],
    #             dates,
    #             "mping",
    #             "pred_label",
    #             "Predicted label",
    #             os.path.join(save_loc, "cases", f"{case_study}_pred_label.gif"),
    #         )
    #         video(
    #             data["test"],
    #             dates,
    #             "pred_conf",
    #             "Probability",
    #             case_study,
    #             os.path.join(save_loc, "cases", f"{case_study}_prob.gif"),
    #         )
    #         for col in ["evidential", "aleatoric", "epistemic"]:
    #             if col in data["test"]:
    #                 save_path = os.path.join(
    #                     save_loc, "cases", f"{case_study}_{col}.gif"
    #                 )
    #                 video(
    #                     data["test"],
    #                     dates,
    #                     col,
    #                     f"{col} uncertainty",
    #                     case_study,
    #                     save_path,
    #                 )
    #

if __name__ == "__main__":

    description = "Usage: python evaluate_mlp.py -c model.yml"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)
    for newdir in ["plots", "metrics", "cases"]:
        os.makedirs(os.path.join(save_loc, newdir), exist_ok=True)

    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config_file, os.path.join(save_loc, "model.yml"))
    else:
        with open(os.path.join(save_loc, "model.yml"), "w") as fid:
            yaml.dump(conf, fid)

    evaluate(conf)
