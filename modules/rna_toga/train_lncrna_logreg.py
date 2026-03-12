#!/usr/bin/env python3
"""
Train simplified logistic regression model for lncRNA TOGA classification.
Uses 3 features: synteny, gl_exo, flank_cov
Saves model coefficients to JSON.
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


def load_data(classification_table, biotypes_table=None, sample_size=1000):
    """Load and filter TOGA classification data, sample diverse cases."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load classification table
    df = pd.read_csv(classification_table, sep=',', engine='python')
    print(f"\nLoaded {len(df)} rows from {classification_table}")

    # Filter for ORTH and PARA only
    df_filtered = df[df['label'].isin(['ORTH', 'PARA'])].copy()
    print(f"Filtered to ORTH/PARA: {len(df_filtered)} rows")
    print(f"  ORTH: {(df_filtered['label'] == 'ORTH').sum()}")
    print(f"  PARA: {(df_filtered['label'] == 'PARA').sum()}")

    # Filter for lncRNA if biotypes table provided
    if biotypes_table:
        biotypes = pd.read_csv(biotypes_table, sep='\t')
        lncrna_transcripts = set(biotypes[biotypes['biotype'] == 'lncRNA']['transcript_id'])
        print(f"\nFound {len(lncrna_transcripts)} lncRNA transcripts in {biotypes_table}")

        df_filtered = df_filtered[df_filtered['transcript_id'].isin(lncrna_transcripts)].copy()
        print(f"Filtered to lncRNA: {len(df_filtered)} rows")
        print(f"  ORTH: {(df_filtered['label'] == 'ORTH').sum()}")
        print(f"  PARA: {(df_filtered['label'] == 'PARA').sum()}")

    # Sample diverse cases if requested
    if sample_size and sample_size > 0:
        print(f"\n" + "=" * 80)
        print(f"SAMPLING {sample_size} DIVERSE CASES PER CLASS")
        print("=" * 80)

        # Separate by class
        orth_data = df_filtered[df_filtered['label'] == 'ORTH'].copy()
        para_data = df_filtered[df_filtered['label'] == 'PARA'].copy()

        # Create bins for diversity
        orth_data['gl_exo_bin'] = pd.cut(orth_data['gl_exo'], bins=[0, 0.2, 0.4, 0.6, 1.0])
        orth_data['synteny_bin'] = pd.cut(orth_data['synteny'], bins=[0, 10, 30, 100, 10000])
        orth_data['flank_bin'] = pd.cut(orth_data['flank_cov'], bins=[0, 0.2, 0.5, 1.0])

        para_data['gl_exo_bin'] = pd.cut(para_data['gl_exo'], bins=[0, 0.2, 0.4, 0.6, 1.0])
        para_data['synteny_bin'] = pd.cut(para_data['synteny'], bins=[0, 10, 30, 100, 10000])
        para_data['flank_bin'] = pd.cut(para_data['flank_cov'], bins=[0, 0.2, 0.5, 1.0])

        # Sample with diversity (stratified by bins)
        orth_sample = orth_data.groupby(['gl_exo_bin', 'synteny_bin', 'flank_bin'],
                                        observed=True).apply(
            lambda x: x.sample(min(len(x), max(1, sample_size // 20)), random_state=42),
            include_groups=False
        )
        # Flatten and sample to exact size
        orth_sample = orth_sample.sample(min(sample_size, len(orth_data)), random_state=42)

        para_sample = para_data.groupby(['gl_exo_bin', 'synteny_bin', 'flank_bin'],
                                        observed=True).apply(
            lambda x: x.sample(min(len(x), max(1, sample_size // 20)), random_state=42),
            include_groups=False
        )
        para_sample = para_sample.sample(min(sample_size, len(para_data)), random_state=42)

        # Combine
        df_filtered = pd.concat([orth_sample, para_sample])

        print(f"\nSampled data:")
        print(f"  ORTH: {(df_filtered['label'] == 'ORTH').sum()}")
        print(f"  PARA: {(df_filtered['label'] == 'PARA').sum()}")
        print(f"  Total: {len(df_filtered)}")

    return df_filtered


def train_model(df):
    """Train logistic regression model."""
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 80)

    # Add log-transformed synteny feature
    df['synteny_log1p'] = np.log1p(df['synteny'])

    # Prepare features and labels
    feature_names = ['synteny_log1p', 'gl_exo', 'flank_cov']
    X = df[feature_names].values
    y = (df['label'] == 'ORTH').astype(int)  # 1 for ORTH, 0 for PARA

    print(f"\nFeatures: {feature_names}")
    print(f"Samples: {len(X)}")
    print(f"  Class 0 (PARA): {(y == 0).sum()}")
    print(f"  Class 1 (ORTH): {(y == 1).sum()}")

    # Train model with balanced class weights
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X, y)

    # Extract coefficients
    coeffs = model.coef_[0]
    intercept = model.intercept_[0]

    print("\n" + "=" * 80)
    print("MODEL COEFFICIENTS")
    print("=" * 80)
    print(f"\nScore = {coeffs[0]:.6f} * log1p(synteny) + {coeffs[1]:.6f} * gl_exo + {coeffs[2]:.6f} * flank_cov + {intercept:.6f}")
    print("\nInterpretation:")
    print(f"  synteny_log1p: {coeffs[0]:+.6f} ({'higher synteny → ORTH' if coeffs[0] > 0 else 'higher synteny → PARA'})")
    print(f"  gl_exo:        {coeffs[1]:+.6f} ({'higher gl_exo → ORTH' if coeffs[1] > 0 else 'higher gl_exo → PARA'})")
    print(f"  flank_cov:     {coeffs[2]:+.6f} ({'higher flank_cov → ORTH' if coeffs[2] > 0 else 'higher flank_cov → PARA'})")

    return model, feature_names


def evaluate_model(model, df, feature_names):
    """Evaluate model performance."""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Prepare data
    df['synteny_log1p'] = np.log1p(df['synteny'])
    X = df[feature_names].values
    y = (df['label'] == 'ORTH').astype(int)

    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted PARA  Predicted ORTH")
    print(f"True PARA       {cm[0, 0]:14d}  {cm[0, 1]:14d}")
    print(f"True ORTH       {cm[1, 0]:14d}  {cm[1, 1]:14d}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['PARA', 'ORTH'], digits=4))

    # ROC AUC
    auc = roc_auc_score(y, y_pred_proba)
    print(f"ROC AUC Score: {auc:.4f}")

    return {
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(auc),
        'n_samples': len(y),
        'n_orth': int((y == 1).sum()),
        'n_para': int((y == 0).sum())
    }


def compare_with_original(model, df, feature_names):
    """Compare predictions with original TOGA model."""
    print("\n" + "=" * 80)
    print("COMPARISON WITH ORIGINAL TOGA MODEL")
    print("=" * 80)

    # Prepare data
    df['synteny_log1p'] = np.log1p(df['synteny'])
    X = df[feature_names].values

    # New model predictions
    logreg_pred = model.predict(X)
    logreg_pred_label = np.where(logreg_pred == 1, 'ORTH', 'PARA')

    # Original model predictions
    orig_pred_label = np.where(df['pred'] > 0.5, 'ORTH', 'PARA')

    # Compare
    disagreements = logreg_pred_label != orig_pred_label
    n_disagree = disagreements.sum()

    print(f"\nTotal cases: {len(df)}")
    print(f"Disagreements: {n_disagree} ({100 * n_disagree / len(df):.2f}%)")
    print(f"Agreements: {(~disagreements).sum()} ({100 * (~disagreements).sum() / len(df):.2f}%)")

    # Breakdown by type
    orth_to_para = ((orig_pred_label == 'ORTH') & (logreg_pred_label == 'PARA')).sum()
    para_to_orth = ((orig_pred_label == 'PARA') & (logreg_pred_label == 'ORTH')).sum()

    print("\nDisagreement breakdown:")
    print(f"  ORTH→PARA (original says ORTH, logreg says PARA): {orth_to_para}")
    print(f"  PARA→ORTH (original says PARA, logreg says ORTH): {para_to_orth}")

    return {
        'total_cases': int(len(df)),
        'disagreements': int(n_disagree),
        'agreement_rate': float((~disagreements).sum() / len(df)),
        'orth_to_para': int(orth_to_para),
        'para_to_orth': int(para_to_orth)
    }


def save_model(model, feature_names, output_file, evaluation_stats, comparison_stats):
    """Save model coefficients and metadata to JSON."""
    coeffs = model.coef_[0]
    intercept = model.intercept_[0]

    model_data = {
        'model_type': 'logistic_regression',
        'features': feature_names,
        'coefficients': {
            'synteny_log1p': float(coeffs[0]),
            'gl_exo': float(coeffs[1]),
            'flank_cov': float(coeffs[2]),
            'intercept': float(intercept)
        },
        'formula': f'{coeffs[0]:.6f} * log1p(synteny) + {coeffs[1]:.6f} * gl_exo + {coeffs[2]:.6f} * flank_cov + {intercept:.6f}',
        'threshold': 0.5,
        'class_labels': {
            '0': 'PARA',
            '1': 'ORTH'
        },
        'evaluation': evaluation_stats,
        'comparison_with_original_toga': comparison_stats
    }

    with open(output_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"\n" + "=" * 80)
    print(f"Model saved to: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Train simplified logistic regression model for lncRNA TOGA classification'
    )
    parser.add_argument(
        'classification_table',
        help='Path to TOGA classification table (CSV format)'
    )
    parser.add_argument(
        '-b', '--biotypes',
        help='Path to biotypes table (TSV format) to filter for lncRNA only'
    )
    parser.add_argument(
        '-o', '--output',
        default='lncrna_logreg_model.json',
        help='Output JSON file for model (default: lncrna_logreg_model.json)'
    )
    parser.add_argument(
        '-s', '--sample-size',
        type=int,
        default=1000,
        help='Number of diverse samples per class (default: 1000, use 0 for all data)'
    )
    parser.add_argument(
        '--save-dataset',
        help='Save sampled training dataset to TSV file'
    )

    args = parser.parse_args()

    # Load data
    df = load_data(args.classification_table, args.biotypes, args.sample_size)

    if len(df) == 0:
        print("\nERROR: No data left after filtering!")
        sys.exit(1)

    # Save dataset if requested
    if args.save_dataset:
        # Remove binning columns if they exist
        cols_to_drop = ['gl_exo_bin', 'synteny_bin', 'flank_bin']
        df_to_save = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        df_to_save.to_csv(args.save_dataset, sep='\t', index=False)
        print(f"\n" + "=" * 80)
        print(f"Training dataset saved to: {args.save_dataset}")
        print("=" * 80)

    # Train model
    model, feature_names = train_model(df)

    # Evaluate model
    evaluation_stats = evaluate_model(model, df, feature_names)

    # Compare with original TOGA
    comparison_stats = compare_with_original(model, df, feature_names)

    # Save model
    save_model(model, feature_names, args.output, evaluation_stats, comparison_stats)

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
