#!/usr/bin/env python3
"""
Predict cluster label for new Carbon structures using trained models.

Usage:
    python predict.py structure.extxyz --models-dir ./models
    python predict.py structures/ --models-dir ./models
"""

import os, sys, json, pickle, argparse
import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from dscribe.descriptors import SOAP


def load_models(models_dir='./models'):
    """Load all trained pipeline models."""
    with open(os.path.join(models_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    models = {}
    for name in ['scaler', 'ipca', 'kmeans']:
        path = os.path.join(models_dir, f'{name}.pkl')
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)

    return config, models


def predict_structure(atoms, config, models):
    """Run full prediction pipeline on a single structure."""
    # Verify pure Carbon
    symbols = atoms.get_chemical_symbols()
    n_carbon = sum(1 for s in symbols if s == 'C')
    n_total = len(symbols)

    if n_carbon == 0:
        return None, "No Carbon atoms found"

    # SOAP (per-atom, then aggregate with mean + std)
    soap = SOAP(
        species=config['soap_species'],
        r_cut=config['soap_rcut'],
        n_max=config['soap_nmax'],
        l_max=config['soap_lmax'],
        sigma=config['soap_sigma'],
        average=config['soap_average'],
        periodic=True,
    )
    atom_features = soap.create(atoms)  # (n_atoms, D)
    if atom_features.ndim == 1:
        atom_features = atom_features.reshape(1, -1)
    feat_mean = atom_features.mean(axis=0)
    feat_std = atom_features.std(axis=0)
    features = np.concatenate([feat_mean, feat_std]).reshape(1, -1)

    # Scale
    scaled = models['scaler'].transform(features)

    # PCA
    projected = models['ipca'].transform(scaled)

    # Predict cluster
    label = models['kmeans'].predict(projected)[0]
    center = models['kmeans'].cluster_centers_[label]
    distance = np.linalg.norm(projected[0] - center)

    return {
        'cluster': int(label),
        'distance_to_center': float(distance),
        'n_atoms': n_total,
        'n_carbon': n_carbon,
        'pca_coords': projected[0][:3].tolist(),
    }, None


def main():
    parser = argparse.ArgumentParser(description='Carbon Structure Classifier')
    parser.add_argument('input', help='Path to .extxyz/.xyz file or directory')
    parser.add_argument('--models-dir', default='./models', help='Models directory')
    args = parser.parse_args()

    config, models = load_models(args.models_dir)
    print(f"Loaded models (K={config['best_k']}, "
          f"PCA dims={config['pca_n_components']})")

    # Collect files
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                 if f.endswith(('.extxyz', '.xyz'))]
    else:
        files = [args.input]

    print(f"\nProcessing {len(files)} file(s)...\n")
    print(f"{'File':<35s} {'Cluster':>8s} {'Atoms':>6s} {'Distance':>10s}")
    print("-" * 65)

    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        try:
            atoms = ase_read(fpath)
            result, error = predict_structure(atoms, config, models)
            if error:
                print(f"{fname:<35s} {'SKIP':>8s}   {error}")
            else:
                print(f"{fname:<35s} {result['cluster']:>8d} "
                      f"{result['n_atoms']:>6d} {result['distance_to_center']:>10.4f}")
        except Exception as e:
            print(f"{fname:<35s} {'ERROR':>8s}   {e}")


if __name__ == '__main__':
    main()
