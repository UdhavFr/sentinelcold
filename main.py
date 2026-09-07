"""
SentinelCold — Main Pipeline Orchestrator
End-to-end execution: Tier 1 → Tier 6 + experiment framework.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from utils.config_loader import load_config
from utils.logger import setup_logger, get_logger
from utils.reproducibility import set_global_seed

from tier1_ingestion.validator import DataValidator
from experiments.experiment_runner import ExperimentRunner
from experiments.experiment_registry import ExperimentRegistry

from tier4_modeling.baselines.knn_baseline import KNNBaseline
from tier4_modeling.baselines.kmeans_baseline import KMeansBaseline
from tier4_modeling.baselines.eiforest_baseline import EIForestBaseline


def parse_args():
    parser = argparse.ArgumentParser(description="SentinelCold Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", choices=["full", "baselines", "single", "ablation"], default="full",
                        help="Execution mode")
    parser.add_argument("--stage", type=int, default=None,
                        help="Run a specific ablation stage (1-7)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_baselines(data: pd.DataFrame, config, figures_dir: Path, tables_dir: Path):
    """Run all baseline models and save results."""
    log = get_logger("sentinelcold.baselines")
    log.info("═══════════════════════════════════════════")
    log.info("  RUNNING BASELINE MODELS")
    log.info("═══════════════════════════════════════════")

    target_col = config.tier1["target_column"]
    id_col = config.tier1["id_column"]
    X = data.drop(columns=[target_col, id_col], errors="ignore")
    y = data[target_col]

    # Impute missing before baselines
    X = X.fillna(X.median())

    # KNN
    knn = KNNBaseline(k_values=config.tier4["baselines"]["knn"]["k_values"])
    knn_results = knn.evaluate(X, y)
    knn_results.to_csv(tables_dir / "baseline_knn.csv", index=False)
    log.info(knn.rejection_rationale())

    # K-Means
    km = KMeansBaseline(n_clusters=config.tier4["baselines"]["kmeans"]["n_clusters"])
    km_results = km.evaluate(X, y)
    log.info(km.rejection_rationale())

    # Isolation Forest
    eif = EIForestBaseline(
        n_estimators_list=config.tier4["baselines"]["isolation_forest"]["n_estimators"],
        contamination=config.tier4["baselines"]["isolation_forest"]["contamination"],
    )
    eif_results = eif.evaluate(X, y)
    eif_results.to_csv(tables_dir / "baseline_eiforest.csv", index=False)
    log.info(eif.benchmark_note())

    return {"knn": knn_results, "kmeans": km_results, "eiforest": eif_results}


def run_single_experiment(data: pd.DataFrame, config):
    """Run a single experiment with default config."""
    exp_config = {
        "seed": config.seed,
        "imputation": config.tier2["imputation"]["default"],
        "balancing": config.tier2["balancing"]["default"],
        "balance_ratio": config.tier2["balancing"]["default_target_ratio"],
        "feature_set": config.tier3["default_feature_set"],
        "model": "ensemble_tuned",
        "threshold": config.tier4["threshold"]["default"],
        "cv_splits": config.tier4["cv"]["n_splits"],
        "model_configs": config.tier4["models"],
        "threshold_config": config.tier4["threshold"],
    }
    runner = ExperimentRunner(exp_config, data)
    results = runner.run()

    # Register
    registry = ExperimentRegistry(config.experiments_dir / "experiment_registry.csv")
    registry.register(results["config"], results["aggregated"], results["runtime_s"])

    return results


def run_ablation(data: pd.DataFrame, config, stage: int = None):
    """Run the staged ablation study."""
    log = get_logger("sentinelcold.ablation")
    registry = ExperimentRegistry(config.experiments_dir / "experiment_registry.csv")

    stages = config.experiments.get("ablation_stages", [])
    if stage is not None:
        stages = [stages[stage - 1]]

    # Track best from each stage
    best_config = {
        "imputation": config.tier2["imputation"]["default"],
        "balancing": config.tier2["balancing"]["default"],
        "balance_ratio": config.tier2["balancing"]["default_target_ratio"],
        "feature_set": config.tier3["default_feature_set"],
        "model": "xgb_default",
        "threshold": config.tier4["threshold"]["default"],
        "cv_splits": config.tier4["cv"]["n_splits"],
        "model_configs": config.tier4["models"],
        "threshold_config": config.tier4["threshold"],
        "seed": config.seed,
    }

    for stage_def in stages:
        stage_name = stage_def["name"]
        vary_values = stage_def["vary"]
        fix = stage_def.get("fix", {})

        log.info("═══ ABLATION STAGE: %s (vary %d configs) ═══", stage_name, len(vary_values))

        stage_results = []
        for variant in vary_values:
            exp_config = dict(best_config)
            exp_config.update(fix)

            # Apply the variant
            if stage_name == "imputation":
                exp_config["imputation"] = variant
            elif stage_name == "balancing":
                parts = variant.rsplit("_", 1)
                if len(parts) == 2 and parts[1].replace(".", "").isdigit():
                    exp_config["balancing"] = parts[0]
                    exp_config["balance_ratio"] = float(parts[1])
                else:
                    exp_config["balancing"] = variant
            elif stage_name == "features":
                exp_config["feature_set"] = variant
            elif stage_name == "models":
                exp_config["model"] = variant
            elif stage_name == "threshold":
                exp_config["threshold"] = variant

            try:
                runner = ExperimentRunner(exp_config, data)
                results = runner.run()
                registry.register(results["config"], results["aggregated"], results["runtime_s"])
                stage_results.append({
                    "variant": variant,
                    **results["aggregated"],
                })
            except Exception as e:
                log.error("Experiment failed for variant '%s': %s", variant, e)
                continue

        # Pick best from this stage
        if stage_results:
            results_df = pd.DataFrame(stage_results)
            best_idx = results_df["f1_mean"].idxmax()
            best_variant = results_df.loc[best_idx, "variant"]
            log.info("Stage '%s' best: %s (F1=%.4f)", stage_name, best_variant, results_df.loc[best_idx, "f1_mean"])

            # Update best_config for next stage
            if stage_name == "imputation":
                best_config["imputation"] = best_variant
            elif stage_name == "balancing":
                parts = best_variant.rsplit("_", 1)
                if len(parts) == 2 and parts[1].replace(".", "").isdigit():
                    best_config["balancing"] = parts[0]
                    best_config["balance_ratio"] = float(parts[1])
            elif stage_name == "features":
                best_config["feature_set"] = best_variant
            elif stage_name == "models":
                best_config["model"] = best_variant
            elif stage_name == "threshold":
                best_config["threshold"] = best_variant

            # Save stage results
            results_df.to_csv(config.tables_dir / f"ablation_{stage_name}.csv", index=False)

    return best_config


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    cfg.ensure_directories()

    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_file=str(cfg.reports_dir / "pipeline.log"), level=level)
    log = get_logger("sentinelcold")

    # Set seed
    set_global_seed(cfg.seed)

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║         SentinelCold — Pipeline Execution               ║")
    log.info("║         Mode: %-40s  ║", args.mode)
    log.info("╚══════════════════════════════════════════════════════════╝")

    # Tier 1: Ingest & validate
    validator = DataValidator(cfg.tier1)
    data, metadata = validator.validate(
        cfg.raw_data_path,
        report_path=cfg.reports_dir / "profiling_report.json",
    )

    if args.mode == "baselines":
        run_baselines(data, cfg, cfg.figures_dir, cfg.tables_dir)

    elif args.mode == "single":
        run_single_experiment(data, cfg)

    elif args.mode == "ablation":
        run_ablation(data, cfg, stage=args.stage)

    elif args.mode == "full":
        # Run everything
        run_baselines(data, cfg, cfg.figures_dir, cfg.tables_dir)
        best_config = run_ablation(data, cfg)
        log.info("Best configuration: %s", best_config)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
