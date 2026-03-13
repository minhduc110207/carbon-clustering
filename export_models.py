#!/usr/bin/env python3
"""
Export trained models from Kaggle checkpoint directory to portable models/ directory.

Usage:
    python export_models.py [--checkpoint-dir ./checkpoints] [--output-dir ./models]
"""

import os, json, pickle, shutil, argparse
import numpy as np


def export_models(checkpoint_dir='./checkpoints', output_dir='./models'):
    """Extract pipeline models from checkpoint to a clean directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint state
    state_file = os.path.join(checkpoint_dir, 'pipeline_state.json')
    if not os.path.exists(state_file):
        print(f"ERROR: No checkpoint found at {state_file}")
        return False

    with open(state_file, 'r') as f:
        state = json.load(f)

    print("Pipeline checkpoint status:")
    for stage, info in state.get('completed', {}).items():
        ts = info.get('timestamp', 'unknown')[:19]
        print(f"  ✓ {stage:<25s} ({ts})")

    # Models to export
    model_files = {
        'welford_scaler': 'scaler.pkl',
        'ipca_optimal': 'ipca.pkl',
        'ipca_full': 'ipca_full.pkl',
        'best_kmeans_model': 'kmeans.pkl',
        'iso_forest': 'isolation_forest.pkl',
        'ocsvm': 'ocsvm.pkl',
    }

    print(f"\nExporting to {output_dir}/:")
    for ckpt_name, output_name in model_files.items():
        src = os.path.join(checkpoint_dir, f'{ckpt_name}.pkl')
        dst = os.path.join(output_dir, output_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_kb = os.path.getsize(dst) / 1024
            print(f"  ✓ {output_name:<25s} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {ckpt_name}.pkl not found — skipping")

    # Export config from checkpoint metadata
    pca_meta = state['completed'].get('stage4_pca', {}).get('metadata', {})
    kmeans_meta = state['completed'].get('stage6_kmeans', {}).get('metadata', {})
    carbon_meta = state['completed'].get('stage1_carbon_filter', {}).get('metadata', {})
    soap_meta = state['completed'].get('stage2_soap', {}).get('metadata', {})

    config = {
        'carbon_only': carbon_meta.get('carbon_only', True),
        'min_carbon_atoms': carbon_meta.get('min_carbon_atoms', 2),
        'soap_species': ['C'],
        'soap_rcut': 6.0,
        'soap_nmax': 8,
        'soap_lmax': 8,
        'soap_sigma': 1.0,
        'soap_average': 'off',
        'pca_n_components': pca_meta.get('n_components'),
        'best_k': kmeans_meta.get('best_k'),
        'n_carbon_molecules': carbon_meta.get('n_carbon_molecules'),
        'n_features': soap_meta.get('n_features'),
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ config.json")

    # Export cumulative variance if available
    if 'cumulative_variance' in pca_meta:
        cumvar = np.array(pca_meta['cumulative_variance'])
        np.save(os.path.join(output_dir, 'cumulative_variance.npy'), cumvar)
        print(f"  ✓ cumulative_variance.npy")

    print(f"\nDone! Models exported to {output_dir}/")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Carbon Classification models')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--output-dir', default='./models')
    args = parser.parse_args()
    export_models(args.checkpoint_dir, args.output_dir)
