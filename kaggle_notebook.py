#!/usr/bin/env python3
"""
Carbon Structure Classification — Unsupervised Pipeline (Kaggle Notebook)

Dataset: jla-gardner/carbon-data (22.9M carbon atoms)
  - Pure Carbon structures: CNT, graphene, diamond, buckyball, amorphous
  - Format: .extxyz with metadata (density, temperature)
  - Paper: arXiv:2211.16443

Pipeline:
  Stage 1: Load .extxyz → ASE Atoms (pure Carbon, ~200 atoms/structure)
  Stage 2: R_C → p ∈ ℝ^D                (SOAP descriptor encoding)
  Stage 3: X ∈ ℝ^(M×D) → Z ∈ ℝ^(M×k)  (StandardScaler + Incremental PCA)
  Stage 4: Z → L ∈ {1,...,K}^M           (K-Means++ clustering)
"""

# === SETUP ===

import subprocess, sys

def install_packages():
    pkgs = ['dscribe', 'ase', 'h5py']
    for p in pkgs:
        try:
            __import__(p.replace('-', '_').split('==')[0])
        except ImportError:
            print(f"[SETUP] Installing {p}...")
            subprocess.call([sys.executable, '-m', 'pip', 'install', '-q', p])

install_packages()

# === IMPORTS ===

import os, io, json, time, pickle, zipfile, glob, warnings, requests
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

# Helper to display plots reliably on Kaggle/Colab/Jupyter
def show_plot(save_path=None):
    """Save and display plot. Works in Kaggle notebooks and local scripts."""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved {os.path.basename(save_path)}")
    try:
        from IPython.display import display
        display(plt.gcf())
    except ImportError:
        plt.show()
    plt.close()

from collections import Counter
from datetime import datetime

from ase import Atoms
from ase.io import read as ase_read
from dscribe.descriptors import SOAP
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# === CONFIGURATION ===

class Config:
    # --- Data ---
    DATA_DIR         = './data'
    GITHUB_ZIP_URL   = 'https://github.com/jla-gardner/carbon-data/archive/refs/heads/main.zip'
    DATA_ZIP         = 'carbon-data.zip'
    EXTXYZ_DIR       = './data/carbon-data-main/results'
    MAX_TRAJECTORIES = None   # None = all, set to e.g. 30 for faster testing
    MAX_SNAPSHOTS    = 50     # Snapshots per trajectory (max 210)
    HDF5_PATH        = './data/carbon_soap_features.h5'

    # --- SOAP (Stage 2) ---
    # species=['C'] is academically valid: dataset is 100% Carbon
    SOAP_SPECIES     = ['C']
    SOAP_RCUT        = 6.0    # Cutoff radius (Å) — covers ~3rd neighbor shell in carbon
    SOAP_NMAX        = 8      # Radial basis resolution
    SOAP_LMAX        = 6      # Angular basis resolution
    SOAP_SIGMA       = 0.5    # Gaussian smearing width (Å) — tighter for crystalline C
    SOAP_AVERAGE     = 'off'  # No averaging → 1 descriptor per atom
    BATCH_SIZE       = 512

    # --- PCA (Stage 3) ---
    PCA_VARIANCE     = 0.95   # Keep components explaining ≥95% variance

    # --- Anomaly Detection (Stage 4 — supplementary, not in core framework) ---
    ANOMALY_ENABLED  = True
    IF_CONTAMINATION = 0.03
    OCSVM_NU         = 0.03
    OCSVM_KERNEL     = 'rbf'

    # --- K-Means (Stage 5) ---
    K_RANGE          = [3, 4, 5, 6, 8, 10, 15]
    KMEANS_BATCH     = 2048
    KMEANS_NINIT     = 10

    # --- Output ---
    RESULTS_DIR      = './results'
    CHECKPOINT_DIR   = './checkpoints'
    CHECKPOINT_FILE  = './checkpoints/pipeline_state.json'
    MODELS_DIR       = './models'
    FRESH_RUN        = True   # Set True to re-run from scratch

cfg = Config()

# Clean old data if FRESH_RUN
if cfg.FRESH_RUN:
    import shutil
    for d in [cfg.CHECKPOINT_DIR, cfg.RESULTS_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    for pattern in ['_scaled.h5', '_pca.h5', '_clean.h5']:
        p = cfg.HDF5_PATH.replace('.h5', pattern)
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(cfg.HDF5_PATH):
        os.remove(cfg.HDF5_PATH)
    print("[FRESH] Starting pipeline from scratch\n")

# === CHECKPOINT MANAGER ===

class CheckpointManager:
    """Saves pipeline state after each stage for crash-resilient execution."""

    STAGES = [
        'stage0_data', 'stage1_load', 'stage2_soap',
        'stage3_scaler', 'stage4_pca', 'stage5_anomaly', 'stage6_kmeans',
    ]

    def __init__(self, config):
        self.config = config
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.config.CHECKPOINT_FILE):
            try:
                with open(self.config.CHECKPOINT_FILE, 'r') as f:
                    self.state = json.load(f)
                print(f"[CKPT] Loaded checkpoint from {self.config.CHECKPOINT_FILE}")
            except (json.JSONDecodeError, IOError):
                self.state = {}
        else:
            self.state = {}
        if 'completed' not in self.state:
            self.state['completed'] = {}

    def _save(self):
        with open(self.config.CHECKPOINT_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def is_stage_done(self, stage_name):
        info = self.state['completed'].get(stage_name)
        if info is None:
            return False
        for fpath in info.get('output_files', []):
            if not os.path.exists(fpath):
                return False
        return True

    def mark_done(self, stage_name, output_files=None, metadata=None):
        self.state['completed'][stage_name] = {
            'timestamp': datetime.now().isoformat(),
            'output_files': output_files or [],
            'metadata': metadata or {},
        }
        self._save()
        print(f"[CKPT] Stage '{stage_name}' checkpointed")

    def get_metadata(self, stage_name):
        return self.state['completed'].get(stage_name, {}).get('metadata', {})

    def save_object(self, name, obj):
        path = os.path.join(self.config.CHECKPOINT_DIR, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def load_object(self, name):
        path = os.path.join(self.config.CHECKPOINT_DIR, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def get_resume_point(self):
        for stage in self.STAGES:
            if not self.is_stage_done(stage):
                return stage
        return None

    def print_status(self):
        print("\n" + "=" * 60)
        print("  CHECKPOINT STATUS")
        print("=" * 60)
        resume = self.get_resume_point()
        for stage in self.STAGES:
            done = self.is_stage_done(stage)
            icon = "[x]" if done else ("-->" if stage == resume else "[ ]")
            info = self.state['completed'].get(stage, {})
            ts = info.get('timestamp', '')
            if ts:
                ts = f"  ({ts[:19]})"
            print(f"  {icon}  {stage:<25s}{ts}")
        if resume:
            print(f"\n  Resume from: {resume}")
        else:
            print(f"\n  All stages complete!")
        print("=" * 60 + "\n")


ckpt = CheckpointManager(cfg)
ckpt.print_status()

print("=" * 60)
print("  Carbon Structure Classification Pipeline v2")
print("=" * 60)
print(f"  Dataset:  jla-gardner/carbon-data (pure Carbon)")
print(f"  SOAP:     n_max={cfg.SOAP_NMAX}, l_max={cfg.SOAP_LMAX}, "
      f"r_cut={cfg.SOAP_RCUT}A, sigma={cfg.SOAP_SIGMA}")
print(f"  SOAP species: {cfg.SOAP_SPECIES} (valid: 100% C dataset)")
print(f"  PCA variance: {cfg.PCA_VARIANCE*100:.0f}%")
print(f"  K candidates: {cfg.K_RANGE}")
print(f"  Anomaly detection: {'ON' if cfg.ANOMALY_ENABLED else 'OFF'} "
      f"(supplementary)")
print("=" * 60)


# ======================================================================
#  STAGE 0: DOWNLOAD CARBON-DATA FROM GITHUB
# ======================================================================

print("\n" + "-" * 50)
print("  STAGE 0: DOWNLOAD DATASET")
print("-" * 50)


def download_carbon_data(config):
    """Download jla-gardner/carbon-data from GitHub."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    zip_path = os.path.join(config.DATA_DIR, config.DATA_ZIP)

    # Check if already extracted
    if os.path.isdir(config.EXTXYZ_DIR):
        n_files = len(glob.glob(os.path.join(config.EXTXYZ_DIR, '*.extxyz')))
        if n_files > 0:
            print(f"[DATA] Found {n_files} .extxyz files in {config.EXTXYZ_DIR}")
            return n_files

    # Download zip
    if not os.path.exists(zip_path):
        print(f"[DATA] Downloading carbon-data from GitHub...")
        response = requests.get(config.GITHUB_ZIP_URL, stream=True, timeout=600)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r[DATA] {downloaded/(1024**2):.1f}/{total/(1024**2):.1f} MB "
                          f"({pct:.1f}%)", end='')
        print(f"\n[DATA] Downloaded ({downloaded/(1024**2):.1f} MB)")
    else:
        print(f"[DATA] Found cached zip at {zip_path}")

    # Extract
    print("[DATA] Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Extract only .extxyz files from results/
        members = [m for m in zf.namelist()
                   if m.endswith('.extxyz') and '/results/' in m]
        for i, member in enumerate(members):
            zf.extract(member, config.DATA_DIR)
            if (i + 1) % 50 == 0 or (i + 1) == len(members):
                print(f"\r[DATA] Extracted {i+1}/{len(members)}", end='')
        print()

    n_files = len(glob.glob(os.path.join(config.EXTXYZ_DIR, '*.extxyz')))
    print(f"[DATA] {n_files} trajectory files extracted")
    return n_files


if ckpt.is_stage_done('stage0_data'):
    print("[CKPT] Stage 0 done -- data cached")
    n_trajectory_files = ckpt.get_metadata('stage0_data').get('n_files', 0)
else:
    n_trajectory_files = download_carbon_data(cfg)
    ckpt.mark_done('stage0_data',
                   output_files=[cfg.EXTXYZ_DIR],
                   metadata={'n_files': n_trajectory_files})


# ======================================================================
#  STAGE 1: LOAD STRUCTURES WITH METADATA
#  (No Carbon filter needed -- dataset is 100% Carbon)
# ======================================================================

print("\n" + "-" * 50)
print("  STAGE 1: LOAD CARBON STRUCTURES")
print("-" * 50)


def load_carbon_structures(config):
    """
    Load .extxyz trajectories as ASE Atoms objects.

    No Carbon filter needed: dataset is guaranteed 100% Carbon.
    Extracts metadata: density (g/cm3) and temperature (K).
    """
    extxyz_files = sorted(glob.glob(os.path.join(config.EXTXYZ_DIR, '*.extxyz')))
    if config.MAX_TRAJECTORIES:
        # Select evenly spaced subset for density coverage
        step = max(1, len(extxyz_files) // config.MAX_TRAJECTORIES)
        extxyz_files = extxyz_files[::step][:config.MAX_TRAJECTORIES]
        print(f"[LOAD] Using {len(extxyz_files)} of {n_trajectory_files} trajectories")
    else:
        print(f"[LOAD] Using all {len(extxyz_files)} trajectories")

    structures = []
    metadata = {'density': [], 'temperature': [], 'n_atoms': [], 'trajectory_id': []}
    n_snapshots = config.MAX_SNAPSHOTS

    for i, fpath in enumerate(extxyz_files):
        fname = os.path.basename(fpath)
        try:
            # Parse density and temperature from filename
            # Format: density-X.X-T-YYYY.extxyz
            parts = fname.replace('.extxyz', '').split('-')
            density = float(parts[1])
            temperature = float(parts[3])

            # Load trajectory snapshots
            trajectory = ase_read(fpath, index=f':{n_snapshots}')
            if not isinstance(trajectory, list):
                trajectory = [trajectory]

            for snap in trajectory:
                structures.append(snap)
                metadata['density'].append(density)
                metadata['temperature'].append(temperature)
                metadata['n_atoms'].append(len(snap))
                metadata['trajectory_id'].append(i)

        except Exception as e:
            print(f"\n[LOAD] Warning: {fname}: {e}")
            continue

        if (i + 1) % 10 == 0 or (i + 1) == len(extxyz_files):
            print(f"\r[LOAD] {i+1}/{len(extxyz_files)} files, "
                  f"{len(structures)} structures", end='')

    for key in metadata:
        metadata[key] = np.array(metadata[key])

    # Verify 100% Carbon
    n_non_carbon = 0
    for s in structures[:100]:  # Spot check
        symbols = s.get_chemical_symbols()
        if not all(sym == 'C' for sym in symbols):
            n_non_carbon += 1

    print(f"\n[LOAD] =============================================")
    print(f"[LOAD]   Total structures:       {len(structures)}")
    print(f"[LOAD]   Atoms per structure:     {int(metadata['n_atoms'].mean())} "
          f"(range: {int(metadata['n_atoms'].min())}-{int(metadata['n_atoms'].max())})")
    print(f"[LOAD]   Density range:           {metadata['density'].min():.1f} - "
          f"{metadata['density'].max():.1f} g/cm3")
    print(f"[LOAD]   Temperature range:       {metadata['temperature'].min():.0f} - "
          f"{metadata['temperature'].max():.0f} K")
    print(f"[LOAD]   Unique densities:        {len(np.unique(metadata['density']))}")
    print(f"[LOAD]   Species verification:    "
          f"{'100% Carbon' if n_non_carbon == 0 else f'{n_non_carbon} non-C found!'}")
    print(f"[LOAD] =============================================")

    return structures, metadata


structures = None
meta_data = None

if ckpt.is_stage_done('stage1_load'):
    print("[CKPT] Stage 1 done -- loading metadata")
    meta = ckpt.get_metadata('stage1_load')
    n_structures = meta.get('n_structures', 0)
    print(f"[LOAD] Cached: {n_structures} structures")
else:
    structures, meta_data = load_carbon_structures(cfg)
    n_structures = len(structures)

    # Save metadata for later analysis
    np.savez(os.path.join(cfg.DATA_DIR, 'structure_metadata.npz'), **meta_data)

    ckpt.mark_done('stage1_load',
                   output_files=[os.path.join(cfg.DATA_DIR, 'structure_metadata.npz')],
                   metadata={
                       'n_structures': n_structures,
                       'density_range': [float(meta_data['density'].min()),
                                         float(meta_data['density'].max())],
                       'temperature_range': [float(meta_data['temperature'].min()),
                                             float(meta_data['temperature'].max())],
                   })


# ======================================================================
#  STAGE 2: SOAP FEATURE EXTRACTION
#  R_C -> p in R^D (rotation-invariant geometric fingerprint)
#
#  Academic justification: species=['C'] is valid because the dataset
#  contains ONLY Carbon atoms. SOAP encodes local Carbon-Carbon geometry
#  including bond distances, angles, and coordination environments.
# ======================================================================

print("\n" + "-" * 50)
print("  STAGE 2: SOAP FEATURE EXTRACTION")
print("-" * 50)


def compute_soap_to_hdf5(structures, config):
    """
    Compute SOAP descriptors for Carbon structures -> HDF5.

    SOAP encodes each structure's geometry via:
      1. Gaussian smearing of atomic positions (sigma)
      2. Radial Rn(r) + Spherical harmonics Ylm(theta, phi)
      3. Power spectrum (rotation-invariant)
      4. Per-atom descriptors (average='off') -> aggregate via mean+std
         concatenation -> single 2*D vector per structure
    """
    soap = SOAP(
        species=config.SOAP_SPECIES, r_cut=config.SOAP_RCUT,
        n_max=config.SOAP_NMAX, l_max=config.SOAP_LMAX,
        sigma=config.SOAP_SIGMA, average=config.SOAP_AVERAGE, periodic=True,
    )
    n_structs = len(structures)
    n_feat_raw = soap.get_number_of_features()
    n_feat = 2 * n_feat_raw  # mean + std concatenation
    batch_size = config.BATCH_SIZE

    print(f"[SOAP] {n_structs} structures, {n_feat_raw} raw features per atom")
    print(f"[SOAP] Aggregation: mean + std -> {n_feat} features per structure")
    print(f"[SOAP] species={config.SOAP_SPECIES}, n_max={config.SOAP_NMAX}, "
          f"l_max={config.SOAP_LMAX}, r_cut={config.SOAP_RCUT}A, "
          f"sigma={config.SOAP_SIGMA}")
    print(f"[SOAP] periodic=True (structures have periodic boundary conditions)")

    os.makedirs(os.path.dirname(config.HDF5_PATH), exist_ok=True)

    with h5py.File(config.HDF5_PATH, 'w') as hf:
        dset = hf.create_dataset(
            'soap_features', shape=(n_structs, n_feat), dtype='float32',
            chunks=(min(batch_size, n_structs), n_feat),
            compression='gzip', compression_opts=4,
        )
        start_time = time.time()
        for start in range(0, n_structs, batch_size):
            end = min(start + batch_size, n_structs)
            batch = structures[start:end]

            # Compute per-atom SOAP and aggregate per structure
            for i, struct in enumerate(batch):
                atom_features = soap.create(struct)  # (n_atoms, D)
                if atom_features.ndim == 1:
                    atom_features = atom_features.reshape(1, -1)
                feat_mean = atom_features.mean(axis=0)
                feat_std = atom_features.std(axis=0)
                dset[start + i] = np.concatenate([feat_mean, feat_std]).astype(np.float32)

            elapsed = time.time() - start_time
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n_structs - end) / rate if rate > 0 else 0
            print(f"\r[SOAP] {end}/{n_structs} ({end/n_structs*100:.1f}%) "
                  f"-- {rate:.0f} struct/s -- ETA: {eta:.0f}s", end='')

        print(f"\n[SOAP] Saved to {config.HDF5_PATH} "
              f"({os.path.getsize(config.HDF5_PATH)/(1024**2):.1f} MB)")

    return n_structs, n_feat


if ckpt.is_stage_done('stage2_soap'):
    print("[CKPT] Stage 2 done")
    meta = ckpt.get_metadata('stage2_soap')
    n_structures, n_feat = meta['n_structures'], meta['n_features']
    print(f"[SOAP] Cached: {n_structures} structures, {n_feat} features")
else:
    if structures is None:
        structures, meta_data = load_carbon_structures(cfg)
    n_structures, n_feat = compute_soap_to_hdf5(structures, cfg)
    ckpt.mark_done('stage2_soap',
                   output_files=[cfg.HDF5_PATH],
                   metadata={'n_structures': n_structures, 'n_features': n_feat})
    del structures
    structures = None


# ======================================================================
#  STAGE 3: ONLINE STANDARDSCALER (Welford)
#  z = (x - mu) / sigma
# ======================================================================

print("\n" + "-" * 50)
print("  STAGE 3: ONLINE STANDARDSCALER (Welford)")
print("-" * 50)


class WelfordScaler:
    """Online StandardScaler using Welford's algorithm for batch processing."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None
        self._fitted = False

    def partial_fit(self, batch):
        if self.mean is None:
            self.mean = np.zeros(batch.shape[1], dtype=np.float64)
            self.M2 = np.zeros(batch.shape[1], dtype=np.float64)
        for x in batch:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
        self._fitted = True
        return self

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n >= 2 else np.ones_like(self.mean)

    @property
    def std(self):
        return np.sqrt(np.maximum(self.variance, 1e-10))

    def transform(self, batch):
        assert self._fitted, "Scaler not fitted yet."
        return (batch - self.mean) / self.std

    def fit_transform_batched(self, hdf5_path, dataset_name, batch_size):
        """Two-pass: compute stats -> transform."""
        scaled_path = hdf5_path.replace('.h5', '_scaled.h5')

        print("[SCALER] Pass 1/2: Computing statistics...")
        with h5py.File(hdf5_path, 'r') as hf:
            dset = hf[dataset_name]
            n_samples, n_features = dset.shape
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                self.partial_fit(dset[start:end][:])
                print(f"\r  [{end}/{n_samples}]", end='')

        print(f"\n[SCALER] mean_range=[{self.mean.min():.4f}, {self.mean.max():.4f}], "
              f"std_range=[{self.std.min():.4f}, {self.std.max():.4f}]")

        print("[SCALER] Pass 2/2: Transforming...")
        with h5py.File(hdf5_path, 'r') as hf_in, \
             h5py.File(scaled_path, 'w') as hf_out:
            dset_in = hf_in[dataset_name]
            dset_out = hf_out.create_dataset(
                'scaled_features', shape=dset_in.shape, dtype='float32',
                chunks=(min(batch_size, n_samples), n_features), compression='gzip',
            )
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                dset_out[start:end] = self.transform(dset_in[start:end][:]).astype(np.float32)
                print(f"\r  [{end}/{n_samples}]", end='')

        print(f"\n[SCALER] Saved to {scaled_path}")
        return scaled_path


scaled_hdf5 = cfg.HDF5_PATH.replace('.h5', '_scaled.h5')

if ckpt.is_stage_done('stage3_scaler'):
    print("[CKPT] Stage 3 done")
    scaler = ckpt.load_object('welford_scaler')
    if scaler is None:
        scaler = WelfordScaler()
else:
    scaler = WelfordScaler()
    scaled_hdf5 = scaler.fit_transform_batched(cfg.HDF5_PATH, 'soap_features', cfg.BATCH_SIZE)
    ckpt.save_object('welford_scaler', scaler)
    ckpt.mark_done('stage3_scaler',
                   output_files=[scaled_hdf5],
                   metadata={'scaled_path': scaled_hdf5})


# ======================================================================
#  STAGE 4: INCREMENTAL PCA
#  X in R^(M x D) -> Z in R^(M x k)
# ======================================================================

print("\n" + "-" * 50)
print("  STAGE 4: INCREMENTAL PCA")
print("-" * 50)


def fit_incremental_pca(scaled_path, config):
    """Two-pass IPCA: analyze variance -> fit optimal -> transform."""
    batch_size = config.BATCH_SIZE

    with h5py.File(scaled_path, 'r') as hf:
        n_samples, n_features = hf['scaled_features'].shape

    # Pass 1: Full IPCA for variance analysis
    print(f"[PCA] Pass 1: {n_samples} samples, {n_features} features")
    max_components = min(n_features, batch_size, 200)
    ipca_full = IncrementalPCA(n_components=max_components)

    with h5py.File(scaled_path, 'r') as hf:
        dset = hf['scaled_features']
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = dset[start:end][:]
            if batch.shape[0] >= max_components:
                ipca_full.partial_fit(batch)
            print(f"\r  [{end}/{n_samples}]", end='')

    cumvar = np.cumsum(ipca_full.explained_variance_ratio_)
    n_components = min(int(np.searchsorted(cumvar, config.PCA_VARIANCE) + 1), max_components)

    print(f"\n[PCA] {n_features} -> {n_components} components "
          f"({cumvar[n_components-1]*100:.2f}% variance)")

    # Pass 2: Fit optimal and transform
    print(f"[PCA] Pass 2: Fitting n={n_components}...")
    ipca = IncrementalPCA(n_components=n_components)

    with h5py.File(scaled_path, 'r') as hf:
        dset = hf['scaled_features']
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = dset[start:end][:]
            if batch.shape[0] >= n_components:
                ipca.partial_fit(batch)
            print(f"\r  [{end}/{n_samples}]", end='')

    print(f"\n[PCA] Transforming...")
    pca_path = scaled_path.replace('_scaled.h5', '_pca.h5')

    with h5py.File(scaled_path, 'r') as hf_in, \
         h5py.File(pca_path, 'w') as hf_out:
        dset_in = hf_in['scaled_features']
        dset_out = hf_out.create_dataset(
            'pca_features', shape=(n_samples, n_components), dtype='float32',
            chunks=(min(batch_size, n_samples), n_components), compression='gzip',
        )
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            dset_out[start:end] = ipca.transform(dset_in[start:end][:]).astype(np.float32)
            print(f"\r  [{end}/{n_samples}]", end='')

    print(f"\n[PCA] Saved to {pca_path}")
    return pca_path, ipca, ipca_full, n_components, cumvar


pca_path = cfg.HDF5_PATH.replace('.h5', '_pca.h5')
n_pca_components = None
cumvar = None

if ckpt.is_stage_done('stage4_pca'):
    print("[CKPT] Stage 4 done")
    meta = ckpt.get_metadata('stage4_pca')
    n_pca_components = meta['n_components']
    cumvar = np.array(meta['cumulative_variance'])
    ipca_full = ckpt.load_object('ipca_full')
    ipca = ckpt.load_object('ipca_optimal')
else:
    pca_path, ipca, ipca_full, n_pca_components, cumvar = fit_incremental_pca(scaled_hdf5, cfg)
    ckpt.save_object('ipca_full', ipca_full)
    ckpt.save_object('ipca_optimal', ipca)
    ckpt.mark_done('stage4_pca',
                   output_files=[pca_path],
                   metadata={
                       'pca_path': pca_path,
                       'n_components': n_pca_components,
                       'cumulative_variance': cumvar.tolist(),
                   })

# PCA Variance Plot
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(range(1, len(cumvar) + 1), cumvar * 100, 'b-', linewidth=2)
ax.axhline(y=cfg.PCA_VARIANCE * 100, color='r', linestyle='--',
           label=f'{cfg.PCA_VARIANCE*100:.0f}% threshold')
ax.axvline(x=n_pca_components, color='g', linestyle='--',
           label=f'n_components = {n_pca_components}')
ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
ax.set_title('Incremental PCA -- Cumulative Variance', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
show_plot(os.path.join(cfg.RESULTS_DIR, 'pca_variance.png'))


# ======================================================================
#  STAGE 5: ANOMALY DETECTION (SUPPLEMENTARY)
#
#  Note: This stage is NOT part of the core 4-stage theoretical framework.
#  It is included as a supplementary step for robustness. It can be
#  disabled by setting Config.ANOMALY_ENABLED = False.
# ======================================================================

print("\n" + "-" * 50)
print("  STAGE 5: ANOMALY DETECTION (supplementary)")
print("-" * 50)


def detect_anomalies(pca_path, config):
    """Consensus anomaly detection: Isolation Forest + One-class SVM."""
    with h5py.File(pca_path, 'r') as hf:
        X_pca = hf['pca_features'][:]

    n_samples = X_pca.shape[0]

    if not config.ANOMALY_ENABLED:
        print("[ANOMALY] Disabled -- skipping")
        normal_mask = np.ones(n_samples, dtype=bool)
        return X_pca, normal_mask, {
            'if_anomalies': 0, 'ocsvm_anomalies': 0,
            'consensus_anomalies': 0, 'clean_samples': n_samples,
        }

    print(f"[ANOMALY] {n_samples} samples, {X_pca.shape[1]} dims")

    print("[ANOMALY] Isolation Forest...")
    iso_forest = IsolationForest(
        contamination=config.IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    labels_if = iso_forest.fit_predict(X_pca)
    n_if = np.sum(labels_if == -1)
    print(f"  -> IF anomalies: {n_if} ({n_if/n_samples*100:.2f}%)")

    MAX_OCSVM = 20000
    if n_samples > MAX_OCSVM:
        print(f"[ANOMALY] OCSVM (subsampling {MAX_OCSVM})...")
        idx = np.random.choice(n_samples, MAX_OCSVM, replace=False)
        ocsvm = OneClassSVM(kernel=config.OCSVM_KERNEL, nu=config.OCSVM_NU)
        ocsvm.fit(X_pca[idx])
        labels_ocsvm = ocsvm.predict(X_pca)
    else:
        print("[ANOMALY] One-class SVM...")
        ocsvm = OneClassSVM(kernel=config.OCSVM_KERNEL, nu=config.OCSVM_NU)
        labels_ocsvm = ocsvm.fit_predict(X_pca)

    n_svm = np.sum(labels_ocsvm == -1)
    print(f"  -> OCSVM anomalies: {n_svm} ({n_svm/n_samples*100:.2f}%)")

    consensus = (labels_if == -1) & (labels_ocsvm == -1)
    n_cons = np.sum(consensus)
    normal_mask = ~consensus

    print(f"  -> Consensus removed: {n_cons} ({n_cons/n_samples*100:.2f}%)")
    print(f"  -> Clean remaining: {np.sum(normal_mask)}")

    X_clean = X_pca[normal_mask]
    clean_path = pca_path.replace('_pca.h5', '_clean.h5')
    with h5py.File(clean_path, 'w') as hf:
        hf.create_dataset('clean_features', data=X_clean, compression='gzip')

    anomaly_stats = {
        'if_anomalies': int(n_if), 'ocsvm_anomalies': int(n_svm),
        'consensus_anomalies': int(n_cons), 'clean_samples': int(np.sum(normal_mask)),
    }

    ckpt.save_object('iso_forest', iso_forest)
    ckpt.save_object('ocsvm', ocsvm)

    return X_pca, normal_mask, anomaly_stats


X_clean = None
normal_mask = None
anomaly_stats = None

if ckpt.is_stage_done('stage5_anomaly'):
    print("[CKPT] Stage 5 done")
    anomaly_stats = ckpt.get_metadata('stage5_anomaly')
    clean_path = pca_path.replace('_pca.h5', '_clean.h5')
    if os.path.exists(clean_path):
        with h5py.File(clean_path, 'r') as hf:
            X_clean = hf['clean_features'][:]
    else:
        with h5py.File(pca_path, 'r') as hf:
            X_clean = hf['pca_features'][:]
    normal_mask = np.ones(X_clean.shape[0], dtype=bool)
    # Reconstruct mask from sizes
    total = anomaly_stats.get('clean_samples', 0) + anomaly_stats.get('consensus_anomalies', 0)
    if total > 0:
        normal_mask_full = np.ones(total, dtype=bool)
        # We can't fully reconstruct, so re-read pca for use later
        with h5py.File(pca_path, 'r') as hf:
            total_pca = hf['pca_features'].shape[0]
        normal_mask = np.ones(total_pca, dtype=bool)
else:
    _, normal_mask, anomaly_stats = detect_anomalies(pca_path, cfg)
    clean_path = pca_path.replace('_pca.h5', '_clean.h5')
    if os.path.exists(clean_path):
        with h5py.File(clean_path, 'r') as hf:
            X_clean = hf['clean_features'][:]
    else:
        with h5py.File(pca_path, 'r') as hf:
            X_clean = hf['pca_features'][:]
    ckpt.mark_done('stage5_anomaly',
                   output_files=[clean_path] if os.path.exists(clean_path) else [],
                   metadata=anomaly_stats)


# ======================================================================
#  STAGE 6: K-MEANS++ CLUSTERING
#  Z -> L in {1,...,K}^M
# ======================================================================

print("\n" + "-" * 50)
print("  STAGE 6: K-MEANS++ CLUSTERING")
print("-" * 50)


def optimize_kmeans(X, config):
    """Search over K values for optimal clustering."""
    n_samples = X.shape[0]
    sil_sample = min(10000, n_samples)
    print(f"[KMEANS] K={config.K_RANGE}, {n_samples} samples "
          f"(sil sample={sil_sample})")

    results = {'k_values': [], 'inertia': [], 'silhouette': [], 'dbi': []}
    best_k, best_sil, best_model = None, -1, None

    for k in config.K_RANGE:
        if k >= n_samples:
            continue
        km = MiniBatchKMeans(
            n_clusters=k, batch_size=config.KMEANS_BATCH,
            n_init=config.KMEANS_NINIT, random_state=RANDOM_STATE,
            init='k-means++',
        )
        labels = km.fit_predict(X)

        idx = np.random.choice(n_samples, sil_sample, replace=False) \
            if n_samples > sil_sample else np.arange(n_samples)
        sil = silhouette_score(X[idx], labels[idx])
        dbi = davies_bouldin_score(X, labels)

        results['k_values'].append(k)
        results['inertia'].append(float(km.inertia_))
        results['silhouette'].append(float(sil))
        results['dbi'].append(float(dbi))

        print(f"  K={k:>3d}  WCSS={km.inertia_:.2e}  Sil={sil:.4f}  "
              f"DBI={dbi:.4f}  ({sil:.2f}s)")

        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = km

    print(f"\n[KMEANS] Best K={best_k} (Sil={best_sil:.4f})")
    return best_k, best_model, results


best_k = None
best_model = None
kr = None
final_labels = None

if ckpt.is_stage_done('stage6_kmeans'):
    print("[CKPT] Stage 6 done")
    meta = ckpt.get_metadata('stage6_kmeans')
    best_k = meta['best_k']
    kr = meta.get('search_results', {})
    best_model = ckpt.load_object('best_kmeans_model')
    if best_model is not None:
        final_labels = best_model.predict(X_clean)
    else:
        final_labels = np.zeros(X_clean.shape[0], dtype=int)
else:
    best_k, best_model, kr = optimize_kmeans(X_clean, cfg)
    final_labels = best_model.predict(X_clean)
    ckpt.save_object('best_kmeans_model', best_model)
    ckpt.mark_done('stage6_kmeans',
                   metadata={
                       'best_k': best_k,
                       'best_silhouette': float(kr['silhouette'][
                           kr['k_values'].index(best_k)]),
                       'search_results': kr,
                   })

print(f"\n[KMEANS] Cluster distribution (K={best_k}):")
for c in range(best_k):
    n_c = np.sum(final_labels == c)
    print(f"    Cluster {c}: {n_c} ({n_c/len(final_labels)*100:.1f}%)")


# ======================================================================
#  VISUALIZATION
# ======================================================================

print("\n" + "-" * 50)
print("  VISUALIZATION")
print("-" * 50)

os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

# --- Clustering results ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax = axes[0, 0]
ax.plot(kr['k_values'], kr['inertia'], 'bo-', linewidth=2, markersize=8)
ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax.set_xlabel('K'); ax.set_ylabel('WCSS'); ax.set_title('Elbow Method')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(kr['k_values'], kr['silhouette'], 'go-', linewidth=2, markersize=8)
ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax.set_xlabel('K'); ax.set_ylabel('Silhouette'); ax.set_title('Silhouette Coefficient')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(kr['k_values'], kr['dbi'], 'rs-', linewidth=2, markersize=8)
ax.axvline(x=best_k, color='b', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax.set_xlabel('K'); ax.set_ylabel('DBI (lower=better)'); ax.set_title('Davies-Bouldin Index')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
scatter = ax.scatter(X_clean[:, 0], X_clean[:, 1], c=final_labels, cmap='tab10', s=2, alpha=0.5)
ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
ax.set_title(f'Carbon Clusters (K={best_k}) -- PCA Projection')
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.grid(True, alpha=0.3)

plt.suptitle('Carbon Structure Classification -- Results', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
show_plot(os.path.join(cfg.RESULTS_DIR, 'clustering_results.png'))

# --- Anomaly summary ---
if cfg.ANOMALY_ENABLED:
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Isolation\nForest', 'One-class\nSVM', 'Consensus\n(Both)', 'Clean\nData']
    counts = [anomaly_stats['if_anomalies'], anomaly_stats['ocsvm_anomalies'],
              anomaly_stats['consensus_anomalies'], anomaly_stats['clean_samples']]
    colors = ['#e74c3c', '#e67e22', '#c0392b', '#27ae60']
    bars = ax.bar(methods, counts, color=colors, edgecolor='black', linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Samples'); ax.set_title('Anomaly Detection Summary', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    show_plot(os.path.join(cfg.RESULTS_DIR, 'anomaly_summary.png'))

# --- 3D PCA scatter ---
if X_clean.shape[1] >= 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_clean[:, 0], X_clean[:, 1], X_clean[:, 2],
                         c=final_labels, cmap='tab10', s=2, alpha=0.4)
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2'); ax.set_zlabel('PC 3')
    ax.set_title(f'Carbon Clusters 3D (K={best_k})')
    plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.6)
    plt.tight_layout()
    show_plot(os.path.join(cfg.RESULTS_DIR, 'clusters_3d.png'))


# ======================================================================
#  CLUSTER ANALYSIS WITH METADATA
# ======================================================================

print("\n" + "-" * 50)
print("  CLUSTER ANALYSIS")
print("-" * 50)

# Load metadata
meta_path = os.path.join(cfg.DATA_DIR, 'structure_metadata.npz')
if os.path.exists(meta_path):
    meta_data = dict(np.load(meta_path))
else:
    print("[ANALYSIS] No metadata file found -- skipping metadata analysis")
    meta_data = None

if meta_data is not None:
    # Apply normal_mask to metadata
    clean_density = meta_data['density'][normal_mask]
    clean_temperature = meta_data['temperature'][normal_mask]
    clean_n_atoms = meta_data['n_atoms'][normal_mask]

    print(f"\n{'='*70}")
    print(f"  CARBON CLUSTER CHARACTERIZATION (K={best_k})")
    print(f"{'='*70}")
    print(f"{'Metric':<25s}", end='')
    for c in range(best_k):
        print(f"{'Cluster '+str(c):>12s}", end='')
    print()
    print("-" * 70)

    for label, data, fmt in [
        ('Samples', None, 'count'),
        ('Avg density (g/cm3)', clean_density, '.3f'),
        ('Std density', clean_density, '.3f'),
        ('Avg temperature (K)', clean_temperature, '.0f'),
        ('Avg atoms/structure', clean_n_atoms, '.0f'),
    ]:
        print(f"  {label:<23s}", end='')
        for c in range(best_k):
            mask_c = final_labels == c
            if fmt == 'count':
                print(f"{np.sum(mask_c):>12d}", end='')
            elif 'Std' in label:
                print(f"{np.std(data[mask_c]):>12{fmt}}", end='')
            else:
                print(f"{np.mean(data[mask_c]):>12{fmt}}", end='')
        print()
    print(f"{'='*70}")

    # Density distribution per cluster
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    cluster_sizes = [np.sum(final_labels == c) for c in range(best_k)]
    ax.bar(range(best_k), cluster_sizes,
           color=plt.cm.tab10(np.arange(best_k) / max(best_k, 1)))
    ax.set_xlabel('Cluster'); ax.set_ylabel('Count')
    ax.set_title('Cluster Size Distribution'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for c in range(best_k):
        mask_c = final_labels == c
        ax.hist(clean_density[mask_c], bins=20, alpha=0.5, label=f'C{c}')
    ax.set_xlabel('Density (g/cm3)'); ax.set_ylabel('Count')
    ax.set_title('Density Distribution per Cluster')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for c in range(best_k):
        mask_c = final_labels == c
        ax.hist(clean_temperature[mask_c], bins=20, alpha=0.5, label=f'C{c}')
    ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Count')
    ax.set_title('Temperature Distribution per Cluster')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Carbon Cluster Properties (Metadata)', fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    show_plot(os.path.join(cfg.RESULTS_DIR, 'cluster_properties.png'))

    # Density vs Temperature colored by cluster
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(clean_density, clean_temperature, c=final_labels,
                         cmap='tab10', s=10, alpha=0.5)
    ax.set_xlabel('Density (g/cm3)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(f'Density vs Temperature (colored by Cluster, K={best_k})', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    show_plot(os.path.join(cfg.RESULTS_DIR, 'density_temperature_clusters.png'))


# ======================================================================
#  MODEL EXPORT
# ======================================================================

print("\n" + "-" * 50)
print("  MODEL EXPORT")
print("-" * 50)

os.makedirs(cfg.MODELS_DIR, exist_ok=True)

for fname, obj in [
    ('scaler.pkl', scaler),
    ('ipca.pkl', ipca),
    ('kmeans.pkl', best_model),
]:
    path = os.path.join(cfg.MODELS_DIR, fname)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"[EXPORT] {fname} ({os.path.getsize(path)/(1024):.1f} KB)")

# Save config
config_dict = {
    'dataset': 'jla-gardner/carbon-data',
    'soap_species': cfg.SOAP_SPECIES,
    'soap_rcut': cfg.SOAP_RCUT,
    'soap_nmax': cfg.SOAP_NMAX,
    'soap_lmax': cfg.SOAP_LMAX,
    'soap_sigma': cfg.SOAP_SIGMA,
    'soap_average': cfg.SOAP_AVERAGE,
    'pca_n_components': n_pca_components,
    'best_k': best_k,
    'n_structures': n_structures,
    'n_features': n_feat,
}
with open(os.path.join(cfg.MODELS_DIR, 'config.json'), 'w') as f:
    json.dump(config_dict, f, indent=2)
print("[EXPORT] config.json")

np.save(os.path.join(cfg.MODELS_DIR, 'cumulative_variance.npy'), cumvar)
print("[EXPORT] cumulative_variance.npy")


# ======================================================================
#  FINAL SUMMARY
# ======================================================================

ckpt.print_status()

best_sil = kr['silhouette'][kr['k_values'].index(best_k)] if kr else 0
best_dbi = kr['dbi'][kr['k_values'].index(best_k)] if kr else 0

print("=" * 60)
print("  PIPELINE COMPLETE -- Carbon Structure Classification v2")
print("=" * 60)
print(f"""
  Dataset:      jla-gardner/carbon-data ({n_structures} structures)
  Species:      {cfg.SOAP_SPECIES} (pure Carbon dataset)
  SOAP:         {n_feat} dims (n_max={cfg.SOAP_NMAX}, l_max={cfg.SOAP_LMAX})
  PCA:          {n_pca_components} components ({cfg.PCA_VARIANCE*100:.0f}% variance)
  Anomaly:      {anomaly_stats['consensus_anomalies']} removed (supplementary)
  Clean data:   {anomaly_stats['clean_samples']} structures
  Best K:       {best_k}
  Silhouette:   {best_sil:.4f}
  DBI:          {best_dbi:.4f}
""")
print("=" * 60)
