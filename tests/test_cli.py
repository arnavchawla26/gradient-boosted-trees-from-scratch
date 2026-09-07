import csv
import pickle
import subprocess
import sys

import numpy as np
import pytest

from gbdt.cli import build_parser, main


def run_cli(args):
    """Run the CLI in-process (fast) via main(); returns None, raises on error."""
    main(args)


class TestParser:
    def test_train_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["train"])
        assert args.dataset == "friedman1"
        assert args.n_estimators == 100
        assert args.command == "train"

    def test_compare_requires_no_extra_args(self):
        parser = build_parser()
        args = parser.parse_args(["compare", "--dataset", "moons"])
        assert args.dataset == "moons"
        assert args.baseline_max_depth == 6

    def test_missing_command_errors(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestEvaluateCommand:
    def test_regression_runs_and_prints_metric(self, capsys):
        run_cli([
            "evaluate", "--dataset", "friedman1", "--n-samples", "300",
            "--n-estimators", "10", "--max-depth", "2",
        ])
        out = capsys.readouterr().out
        assert "RMSE" in out

    def test_classification_runs_and_prints_metric(self, capsys):
        run_cli([
            "evaluate", "--dataset", "moons", "--n-samples", "300",
            "--n-estimators", "10", "--max-depth", "2",
        ])
        out = capsys.readouterr().out
        assert "accuracy" in out

    def test_early_stopping_reports_best_iteration(self, capsys):
        run_cli([
            "evaluate", "--dataset", "friedman1", "--n-samples", "200",
            "--n-estimators", "200", "--max-depth", "5",
            "--min-samples-leaf", "2", "--early-stopping-rounds", "3",
        ])
        out = capsys.readouterr().out
        assert "best_iteration" in out


class TestCompareCommand:
    def test_regression_compare_reports_both_models(self, capsys):
        run_cli([
            "compare", "--dataset", "friedman1", "--n-samples", "400",
            "--n-estimators", "20", "--max-depth", "3",
        ])
        out = capsys.readouterr().out
        assert "single CART tree" in out
        assert "GBDT" in out
        assert "RMSE" in out

    def test_classification_compare_reports_both_models(self, capsys):
        run_cli([
            "compare", "--dataset", "moons", "--n-samples", "400",
            "--n-estimators", "20", "--max-depth", "3",
        ])
        out = capsys.readouterr().out
        assert "single CART tree" in out
        assert "accuracy" in out


class TestTrainPredictRoundTrip:
    def test_train_saves_a_loadable_model(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        run_cli([
            "train", "--dataset", "friedman1", "--n-samples", "200",
            "--n-estimators", "10", "--max-depth", "2",
            "--out", str(model_path),
        ])
        assert model_path.exists()
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        assert len(model.trees_) == 10

    def test_train_on_csv_then_predict(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 150
        x0 = rng.uniform(size=n)
        x1 = rng.uniform(size=n)
        y = 3 * x0 - 2 * x1 + rng.normal(scale=0.05, size=n)

        train_csv = tmp_path / "train.csv"
        with open(train_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x0", "x1", "y"])
            for a, b, c in zip(x0, x1, y):
                writer.writerow([a, b, c])

        model_path = tmp_path / "model.pkl"
        run_cli([
            "train", "--csv", str(train_csv), "--target", "y",
            "--objective", "regression", "--n-estimators", "15",
            "--max-depth", "3", "--out", str(model_path),
        ])

        predict_csv = tmp_path / "predict_features.csv"
        with open(predict_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x0", "x1"])
            for a, b in zip(x0[:10], x1[:10]):
                writer.writerow([a, b])

        out_csv = tmp_path / "preds.csv"
        run_cli([
            "predict", "--model", str(model_path), "--csv", str(predict_csv),
            "--out", str(out_csv),
        ])
        assert out_csv.exists()
        with open(out_csv) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["prediction"]
        assert len(rows) == 11  # header + 10 predictions
        preds = np.array([float(r[0]) for r in rows[1:]])
        # Predictions should be in the right ballpark of the true linear trend.
        expected = 3 * x0[:10] - 2 * x1[:10]
        assert np.corrcoef(preds, expected)[0, 1] > 0.9

    def test_csv_missing_target_column_raises(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b"])
            writer.writerow([1.0, 2.0])
        with pytest.raises(ValueError):
            run_cli([
                "train", "--csv", str(csv_path), "--target", "not_a_column",
                "--out", str(tmp_path / "m.pkl"),
            ])


def test_cli_entry_point_via_subprocess():
    # Exercises the installed console-script entry point end to end, not
    # just the in-process argparse path.
    result = subprocess.run(
        [sys.executable, "-m", "gbdt.cli", "evaluate", "--dataset", "moons",
         "--n-samples", "100", "--n-estimators", "5"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "accuracy" in result.stdout
