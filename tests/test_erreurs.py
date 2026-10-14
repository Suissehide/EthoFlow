"""lib/erreurs.py : le journal d'un échec → cause et conseil en clair."""
from lib.erreurs import expliquer

VAME_OOM = """Epoch 15/200  train_loss 0.0409  test_loss 0.0441
Traceback (most recent call last):
  File "scripts/run_vame.py", line 212, in train
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"""


def test_memoire_gpu():
    e = expliquer(VAME_OOM)
    assert "carte graphique" in e["cause"] and "taille de lot" in e["conseil"]


def test_module_manquant_cite_le_module():
    e = expliquer("ModuleNotFoundError: No module named 'umap'")
    assert "« umap »" in e["cause"]


def test_fichier_introuvable_cite_le_chemin():
    e = expliquer("FileNotFoundError: [Errno 2] No such file or directory: 'D:/videos/970.mp4'")
    assert e["cause"].endswith("D:/videos/970.mp4")


def test_premier_motif_l_emporte():
    # Un OOM qui finit en FileNotFoundError de nettoyage : la vraie cause est l'OOM.
    e = expliquer(VAME_OOM + "\nFileNotFoundError: 'x.pkl'")
    assert "carte graphique" in e["cause"]


def test_sans_correspondance():
    assert expliquer("ValueError: quelque chose d'inattendu") is None
    assert expliquer("") is None
