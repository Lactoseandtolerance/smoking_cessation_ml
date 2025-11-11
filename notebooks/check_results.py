"""Display model comparison results with graceful fallbacks.

Intended usage: run this in a notebook after the modeling pipeline has
produced variables like `comparison_df`, `best_model_name`, `best_model_idx`,
`best_metrics`, `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`,
and `feature_cols`.

If some variables are missing, this script will still render whatever is
available and list what is missing instead of failing.
"""

from typing import Any, Dict
import os

import pandas as pd  # noqa: F401  # may be useful in notebooks

from src.reporting import (
	ModelSummary,
	summarize_results,
	render_text_summary,
	write_markdown_summary,
)


def _get(name: str, default: Any = None) -> Any:
	"""Fetch a variable from the global namespace if present."""
	return globals().get(name, default)


def main() -> None:
	# Collect optional notebook variables
	comparison_df = _get("comparison_df")
	best_model_name = _get("best_model_name")
	best_model_idx = _get("best_model_idx")
	best_metrics = _get("best_metrics")

	X_train = _get("X_train")
	X_val = _get("X_val")
	X_test = _get("X_test")
	y_train = _get("y_train")
	y_val = _get("y_val")
	y_test = _get("y_test")
	feature_cols = _get("feature_cols")

	# Build best model summary if we have at least a name or metrics
	best_model = None
	if best_model_name is not None or best_metrics is not None:
		best_model = ModelSummary(
			model_name=str(best_model_name) if best_model_name is not None else "<unknown>",
			model_index=int(best_model_idx) if best_model_idx is not None else None,
			metrics=best_metrics if isinstance(best_metrics, dict) else None,
		)

	splits: Dict[str, Any] = {}
	for k, v in {
		"X_train": X_train,
		"X_val": X_val,
		"X_test": X_test,
		"y_train": y_train,
		"y_val": y_val,
		"y_test": y_test,
	}.items():
		if v is not None:
			splits[k] = v

	# Summarize and render
	summary = summarize_results(
		comparison_df=comparison_df if isinstance(comparison_df, pd.DataFrame) else None,
		best_model=best_model,
		splits=splits if splits else None,
		feature_cols=feature_cols,
	)
	text = render_text_summary(summary)
	print(text)

	# Persist a Markdown snapshot into reports/
	reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
	os.makedirs(reports_dir, exist_ok=True)
	out_md = os.path.join(reports_dir, "RESULTS_SUMMARY.md")
	write_markdown_summary(summary, out_md)
	print(f"\n[Saved Markdown summary to {out_md}]")


if __name__ == "__main__":
	main()
