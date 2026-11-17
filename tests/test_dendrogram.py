"""community_dendrogram.py : linkage et motifs sans transition."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from community_dendrogram import linkage_motifs, motif_distance_from_transitions  # noqa: E402


def _transitions(seed=0, n=12):
    rng = np.random.default_rng(seed)
    T = rng.integers(0, 40, size=(n, n)).astype(float)
    np.fill_diagonal(T, 0)
    return T


def test_distance_cosinus_symetrique_et_bornee():
    D = motif_distance_from_transitions(_transitions())
    assert np.allclose(D, D.T) and D.min() >= 0 and D.max() <= 1 + 1e-9


@pytest.mark.parametrize("method", ["ward", "average"])
def test_linkage_monotone_sans_inversion(method):
    """Des hauteurs de fusion croissantes : aucune branche ne redescend,
    donc aucun motif ne « flotte » hors de l'arbre."""
    pytest.importorskip("scipy")
    Z = linkage_motifs(_transitions(), method)
    hauteurs = Z[:, 2]
    assert np.all(np.diff(hauteurs) >= -1e-12)
    assert Z.shape[0] == 11
