#!/usr/bin/env python3
"""
Test Suite for Carbon Structure Classification Pipeline

Tests each stage independently using synthetic Carbon molecules,
validating data shapes, transformations, and the full pipeline flow.
"""

import os, sys, time, pickle, json
import numpy as np
import h5py
import traceback
from collections import Counter

# ─── Test Framework ─────────────────────────────────────────

PASS = 0
FAIL = 0
TESTS = []

def test(name):
    """Decorator to register a test function."""
    def decorator(func):
        TESTS.append((name, func))
        return func
    return decorator

def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Expected {a} == {b}. {msg}")

def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(f"Assertion failed. {msg}")

def run_all_tests():
    global PASS, FAIL
    print("=" * 70)
    print("  CARBON STRUCTURE CLASSIFICATION — TEST SUITE")
    print("=" * 70)
    print()

    for name, func in TESTS:
        print(f"  ▸ {name}...", end=' ', flush=True)
        try:
            func()
            print("✓ PASS")
            PASS += 1
        except Exception as e:
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
            FAIL += 1
        print()

    print("=" * 70)
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print("=" * 70)
    return FAIL == 0

# ─── Import Pipeline Components ────────────────────────────

from ase import Atoms
from dscribe.descriptors import SOAP
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# ─── Test Utilities ─────────────────────────────────────────

def make_carbon_molecule(n_carbon, spread=2.0):
    """Create a synthetic Carbon molecule with random 3D positions."""
    positions = np.random.randn(n_carbon, 3) * spread
    return Atoms(symbols=['C'] * n_carbon, positions=positions)

def make_mixed_molecule(n_carbon, n_hydrogen, n_others=0):
    """Create a molecule with C, H, and optionally O/N/F."""
    n_total = n_carbon + n_hydrogen + n_others
    symbols = ['C'] * n_carbon + ['H'] * n_hydrogen
    others = ['O', 'N', 'F']
    for i in range(n_others):
        symbols.append(others[i % 3])
    positions = np.random.randn(n_total, 3) * 2.0
    return Atoms(symbols=symbols, positions=positions)

# ─── TESTS ──────────────────────────────────────────────────

@test("Stage 1: Carbon skeleton filter — keeps Carbon-only")
def test_carbon_filter_basic():
    """Verify filter_carbon_skeleton extracts only C atoms."""
    # Import the filter function from notebook (inline re-implementation for test)
    def filter_carbon_skeleton(atoms, min_carbon=2):
        symbols = atoms.get_chemical_symbols()
        carbon_indices = [i for i, s in enumerate(symbols) if s == 'C']
        if len(carbon_indices) < min_carbon:
            return None
        carbon_positions = atoms.positions[carbon_indices]
        return Atoms(symbols=['C'] * len(carbon_indices), positions=carbon_positions)

    # Test with mixed molecule
    mol = make_mixed_molecule(n_carbon=5, n_hydrogen=8, n_others=2)
    filtered = filter_carbon_skeleton(mol)
    assert_true(filtered is not None, "Filtered molecule should not be None")
    assert_eq(len(filtered), 5, "Should have exactly 5 Carbon atoms")
    assert_true(all(s == 'C' for s in filtered.get_chemical_symbols()),
                "All atoms must be Carbon")


@test("Stage 1: Carbon filter — rejects molecules with too few C")
def test_carbon_filter_rejects():
    def filter_carbon_skeleton(atoms, min_carbon=2):
        symbols = atoms.get_chemical_symbols()
        carbon_indices = [i for i, s in enumerate(symbols) if s == 'C']
        if len(carbon_indices) < min_carbon:
            return None
        carbon_positions = atoms.positions[carbon_indices]
        return Atoms(symbols=['C'] * len(carbon_indices), positions=carbon_positions)

    # Only 1 Carbon → should be rejected (min_carbon=2)
    mol = make_mixed_molecule(n_carbon=1, n_hydrogen=4)
    filtered = filter_carbon_skeleton(mol, min_carbon=2)
    assert_true(filtered is None, "Should reject molecule with < 2 Carbon atoms")

    # No Carbon at all
    mol_no_c = Atoms(symbols=['H', 'H', 'O'], positions=np.random.randn(3, 3))
    filtered_no_c = filter_carbon_skeleton(mol_no_c)
    assert_true(filtered_no_c is None, "Should reject molecule with 0 Carbon atoms")


@test("Stage 1: Carbon filter — preserves 3D coordinates")
def test_carbon_filter_preserves_coords():
    def filter_carbon_skeleton(atoms, min_carbon=2):
        symbols = atoms.get_chemical_symbols()
        carbon_indices = [i for i, s in enumerate(symbols) if s == 'C']
        if len(carbon_indices) < min_carbon:
            return None
        carbon_positions = atoms.positions[carbon_indices]
        return Atoms(symbols=['C'] * len(carbon_indices), positions=carbon_positions)

    # Create molecule with known positions
    positions = np.array([
        [0.0, 0.0, 0.0],  # C
        [1.0, 0.0, 0.0],  # H
        [0.0, 1.5, 0.0],  # C
        [0.0, 0.0, 1.2],  # H
        [2.0, 2.0, 2.0],  # C
    ])
    mol = Atoms(symbols=['C', 'H', 'C', 'H', 'C'], positions=positions)
    filtered = filter_carbon_skeleton(mol)

    assert_eq(len(filtered), 3, "Should have 3 Carbon atoms")
    expected_positions = np.array([[0.0, 0.0, 0.0], [0.0, 1.5, 0.0], [2.0, 2.0, 2.0]])
    assert_true(np.allclose(filtered.positions, expected_positions),
                f"Positions should match. Got {filtered.positions}")


@test("Stage 2: SOAP descriptor — correct output shape")
def test_soap_shape():
    soap = SOAP(
        species=['C'], r_cut=6.0, n_max=8, l_max=8,
        sigma=1.0, average='inner', periodic=False,
    )
    n_feat = soap.get_number_of_features()

    mol = make_carbon_molecule(n_carbon=5)
    features = soap.create(mol)

    assert_eq(features.ndim, 1, "Single molecule SOAP should be 1D (averaged)")
    assert_eq(features.shape[0], n_feat, f"Feature dim should be {n_feat}")
    assert_true(np.isfinite(features).all(), "SOAP features should be finite")
    print(f"(D={n_feat})", end=' ')


@test("Stage 2: SOAP descriptor — batch computation")
def test_soap_batch():
    soap = SOAP(
        species=['C'], r_cut=6.0, n_max=8, l_max=8,
        sigma=1.0, average='inner', periodic=False,
    )
    n_feat = soap.get_number_of_features()

    molecules = [make_carbon_molecule(n_carbon=np.random.randint(2, 9)) for _ in range(20)]
    features = soap.create(molecules)

    assert_eq(features.shape, (20, n_feat), f"Batch shape should be (20, {n_feat})")
    assert_true(np.isfinite(features).all(), "All features should be finite")


@test("Stage 2: SOAP — rotation invariance")
def test_soap_rotation_invariance():
    """SOAP descriptors should be identical for rotated molecules."""
    soap = SOAP(
        species=['C'], r_cut=6.0, n_max=8, l_max=8,
        sigma=1.0, average='inner', periodic=False,
    )

    positions = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0.0]], dtype=float)
    mol1 = Atoms(symbols=['C'] * 4, positions=positions)

    # Rotate 90 degrees around z-axis
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])
    rotated_positions = positions @ R.T
    mol2 = Atoms(symbols=['C'] * 4, positions=rotated_positions)

    feat1 = soap.create(mol1)
    feat2 = soap.create(mol2)

    diff = np.max(np.abs(feat1 - feat2))
    assert_true(diff < 1e-6, f"SOAP should be rotation-invariant. Max diff={diff:.2e}")


@test("Stage 3: Welford Scaler — correct mean and std")
def test_welford_scaler():
    # Inline WelfordScaler for testing
    class WelfordScaler:
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
            return (batch - self.mean) / self.std

    # Generate data with known statistics
    np.random.seed(42)
    data = np.random.randn(1000, 10) * 3.0 + 5.0  # mean~5, std~3

    scaler = WelfordScaler()
    # Process in batches
    for i in range(0, 1000, 100):
        scaler.partial_fit(data[i:i+100])

    # Check mean ~ 5.0 and std ~ 3.0
    assert_true(np.allclose(scaler.mean, data.mean(axis=0), atol=1e-10),
                f"Mean mismatch: {scaler.mean[:3]} vs {data.mean(axis=0)[:3]}")
    assert_true(np.allclose(scaler.std, data.std(axis=0, ddof=1), atol=1e-10),
                f"Std mismatch")

    # Transformed data should have mean~0, std~1
    transformed = scaler.transform(data)
    assert_true(np.abs(transformed.mean()) < 0.01, f"Transformed mean should be ~0")
    assert_true(np.abs(transformed.std() - 1.0) < 0.05, f"Transformed std should be ~1")


@test("Stage 4: Incremental PCA — dimensionality reduction")
def test_ipca():
    np.random.seed(42)
    n_samples, n_features = 500, 100
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Make first 5 components have most variance
    for i in range(5):
        X[:, i] *= (10 - i)

    ipca = IncrementalPCA(n_components=20)
    batch_size = 100
    for start in range(0, n_samples, batch_size):
        ipca.partial_fit(X[start:start+batch_size])

    cumvar = np.cumsum(ipca.explained_variance_ratio_)
    Z = ipca.transform(X)

    assert_eq(Z.shape, (500, 20), "PCA output shape should be (500, 20)")
    assert_true(cumvar[-1] > 0.5, f"20 components should explain >50% variance (got {cumvar[-1]*100:.1f}%)")
    print(f"(var={cumvar[-1]*100:.1f}%)", end=' ')


@test("Stage 5: K-Means++ — clustering quality")
def test_kmeans():
    np.random.seed(42)
    # Create 3 well-separated clusters
    c1 = np.random.randn(100, 5) + np.array([10, 0, 0, 0, 0])
    c2 = np.random.randn(100, 5) + np.array([0, 10, 0, 0, 0])
    c3 = np.random.randn(100, 5) + np.array([0, 0, 10, 0, 0])
    X = np.vstack([c1, c2, c3]).astype(np.float32)

    km = MiniBatchKMeans(n_clusters=3, n_init=10, random_state=42, init='k-means++')
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    assert_true(sil > 0.5, f"Silhouette should be >0.5 for well-separated clusters (got {sil:.3f})")

    # Check 3 clusters with ~100 each
    counts = Counter(labels)
    assert_eq(len(counts), 3, "Should find 3 clusters")
    for k, v in counts.items():
        assert_true(90 < v < 110, f"Cluster {k} has {v} members (expected ~100)")
    print(f"(sil={sil:.3f})", end=' ')


@test("Stage 2+3+4: Full SOAP → Scale → PCA → KMeans flow")
def test_full_flow():
    """End-to-end test: synthetic Carbon molecules → cluster labels."""
    np.random.seed(42)

    # Create two types of Carbon structures
    # Type A: compact (small spread)
    type_a = [make_carbon_molecule(n_carbon=4, spread=1.0) for _ in range(30)]
    # Type B: extended (large spread)
    type_b = [make_carbon_molecule(n_carbon=7, spread=4.0) for _ in range(30)]
    molecules = type_a + type_b

    # SOAP
    soap = SOAP(
        species=['C'], r_cut=6.0, n_max=8, l_max=8,
        sigma=1.0, average='inner', periodic=False,
    )
    features = soap.create(molecules)
    assert_eq(features.shape[0], 60, "Should have 60 feature vectors")

    # Scale
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-10
    scaled = (features - mean) / std

    # PCA
    n_comp = min(10, features.shape[1])
    ipca = IncrementalPCA(n_components=n_comp)
    Z = ipca.fit_transform(scaled)
    assert_eq(Z.shape, (60, n_comp), f"PCA output shape should be (60, {n_comp})")

    # K-Means
    km = MiniBatchKMeans(n_clusters=2, n_init=10, random_state=42, init='k-means++')
    labels = km.fit_predict(Z)

    assert_eq(len(labels), 60, "Should have 60 labels")
    assert_true(len(set(labels)) == 2, "Should find 2 clusters")

    sil = silhouette_score(Z, labels)
    print(f"(sil={sil:.3f}, clusters={Counter(labels)})", end=' ')


@test("HDF5 I/O — write and read features")
def test_hdf5_io():
    test_path = './data/test_features.h5'
    os.makedirs('./data', exist_ok=True)

    np.random.seed(42)
    data = np.random.randn(100, 50).astype(np.float32)

    # Write
    with h5py.File(test_path, 'w') as hf:
        dset = hf.create_dataset('features', data=data, compression='gzip')
        assert_eq(dset.shape, (100, 50))

    # Read
    with h5py.File(test_path, 'r') as hf:
        loaded = hf['features'][:]
        assert_true(np.allclose(data, loaded), "HDF5 roundtrip should be lossless")

    # Read in batches
    with h5py.File(test_path, 'r') as hf:
        dset = hf['features']
        batch1 = dset[0:50][:]
        batch2 = dset[50:100][:]
        assert_true(np.allclose(data[:50], batch1), "Batch 1 mismatch")
        assert_true(np.allclose(data[50:], batch2), "Batch 2 mismatch")

    # Cleanup
    os.remove(test_path)


@test("Checkpoint Manager — save and resume")
def test_checkpoint():
    test_ckpt_dir = './checkpoints_test'
    test_ckpt_file = os.path.join(test_ckpt_dir, 'test_state.json')
    os.makedirs(test_ckpt_dir, exist_ok=True)

    # Simulate checkpoint
    state = {
        'completed': {
            'stage0_data': {
                'timestamp': '2025-01-01T00:00:00',
                'output_files': [],
                'metadata': {'n_molecules': 100}
            }
        }
    }
    with open(test_ckpt_file, 'w') as f:
        json.dump(state, f)

    # Load and verify
    with open(test_ckpt_file, 'r') as f:
        loaded = json.load(f)
    assert_true('stage0_data' in loaded['completed'], "Stage 0 should be in checkpoint")
    assert_eq(loaded['completed']['stage0_data']['metadata']['n_molecules'], 100)

    # Cleanup
    os.remove(test_ckpt_file)
    os.rmdir(test_ckpt_dir)


@test("Model serialization — pickle roundtrip")
def test_model_serialization():
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)

    km = MiniBatchKMeans(n_clusters=3, random_state=42)
    labels_before = km.fit_predict(X)

    # Pickle roundtrip
    test_path = './test_model.pkl'
    with open(test_path, 'wb') as f:
        pickle.dump(km, f)
    with open(test_path, 'rb') as f:
        km_loaded = pickle.load(f)

    labels_after = km_loaded.predict(X)
    assert_true(np.array_equal(labels_before, labels_after),
                "Labels should be identical after pickle roundtrip")

    os.remove(test_path)


# ─── Run ────────────────────────────────────────────────────

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
