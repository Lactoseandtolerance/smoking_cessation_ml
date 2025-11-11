"""Quick script to display model comparison results from notebook variables"""

# These variables should be in the notebook kernel
print("=== MODEL COMPARISON ON REAL DATA (47,882 transitions) ===\n")
print(comparison_df)

print("\n=== BEST MODEL ===")
print(f"Name: {best_model_name}")
print(f"Index: {best_model_idx}")
print("\nBest Model Metrics:")
print(best_metrics)

print("\n=== DATASET SIZES ===")
print(f"Training set: {len(X_train):,} samples")
print(f"Validation set: {len(X_val):,} samples")
print(f"Test set: {len(X_test):,} samples")
print(f"\nFeatures: {len(feature_cols)} columns")

print("\n=== QUIT RATE ===")
print(f"Training: {y_train.mean():.1%}")
print(f"Validation: {y_val.mean():.1%}")
print(f"Test: {y_test.mean():.1%}")
