"""Suit l'entraînement DLC en direct, dans une page web locale.

Répond à la question : « `02_train.py` tourne depuis 3 h, est-ce que la loss
descend encore ? » — sans attendre la fin ni relire un CSV à la main.

DeepLabCut 3 n'écrit pas de logs TensorBoard : son runner d'entraînement
loggue chaque epoch dans `learning_stats.csv`, déposé dans le dossier
`train/` du shuffle. Ce script lit ce fichier en boucle et le sert sous
forme de courbes dans une page qui se rafraîchit toute seule.

Tout est local et en stdlib : pas de compte, pas de dépendance en plus,
pas de connexion internet. Le script n'importe pas DeepLabCut, il peut
donc tourner dans n'importe quel env conda pendant que `dlc` s'occupe de
la GPU.

Usage :
    # Interactif — menu des dossiers de config trouvés sous D:/EthoFlow/models
    python scripts/dlc_model-training/watch_training.py

    # Sur un modèle précis (parcours B, même flag que 02_train.py)
    python scripts/dlc_model-training/watch_training.py ^
        --config-dir D:\\EthoFlow\\models\\souris-bottomview

    # Directement sur le dossier du projet DLC
    python scripts/dlc_model-training/watch_training.py ^
        --model-dir D:\\EthoFlow\\models\\souris-bottomview

    # Sur un CSV précis (utile si tu compares deux shuffles)
    python scripts/dlc_model-training/watch_training.py --stats <...>\\learning_stats.csv

    # Autre port, sans ouvrir le navigateur, accessible depuis le LAN
    python scripts/dlc_model-training/watch_training.py --port 9000 --no-browser
    python scripts/dlc_model-training/watch_training.py --host 0.0.0.0

Lance-le dans un **second terminal** : l'entraînement garde le premier.
Tu peux le démarrer avant `02_train.py` (la page attend le premier epoch)
et le laisser tourner entre deux entraînements.
"""
from __future__ import annotations

import argparse
import csv
import http.server
import json
import math
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

# `_load_config` vit à côté, `interactive` un cran au-dessus.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load_config import add_config_dir_arg, load_config  # noqa: E402

# Silence au-delà duquel on considère que plus rien n'écrit (secondes).
SILENCE_S = 180

# Dossiers de modèles produits par DLC, du plus récent au plus ancien.
RACINES_MODELES = ("dlc-models-pytorch", "dlc-models")


# ----------------------------------------------------------------------
# Localisation des CSV
# ----------------------------------------------------------------------
def trouver_stats(model_dir: Path) -> list[Path]:
    """Tous les `learning_stats*.csv` du projet DLC, du plus récent au plus vieux.

    DLC en écrit un par shuffle (`learning_stats.csv`) et, pour les modèles
    top-down, un second pour le détecteur (`learning_stats_detector.csv`).
    On les remonte tous : la page fait une section par fichier.
    """
    trouves: list[Path] = []
    for racine in RACINES_MODELES:
        base = model_dir / racine
        if base.is_dir():
            trouves.extend(base.glob("*/*/train/learning_stats*.csv"))
    # Dernier filet : un projet réorganisé à la main, ou un shuffle
    # rangé autrement par une version de DLC différente.
    if not trouves:
        trouves.extend(model_dir.rglob("learning_stats*.csv"))
    uniques = {p.resolve(): p for p in trouves}
    return sorted(uniques.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def epochs_prevues(csv_path: Path) -> int | None:
    """Nombre d'epochs demandé, lu dans le `pytorch_config.yaml` du shuffle.

    Sert juste à afficher « epoch 23 / 50 ». Absent ou illisible → None,
    la page affiche l'epoch courant tout seul.
    """
    cfg = csv_path.parent / "pytorch_config.yaml"
    if not cfg.is_file():
        return None
    try:
        import yaml  # noqa: PLC0415 — optionnel, on s'en passe s'il manque
    except ImportError:
        return None
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        valeur = (data.get("train_settings") or {}).get("epochs")
        return int(valeur) if valeur is not None else None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Lecture du CSV
# ----------------------------------------------------------------------
def lire_stats(csv_path: Path) -> dict | None:
    """Parse le CSV en {steps, valeurs}. None si la lecture est inexploitable.

    Piège : DLC réécrit le fichier **en entier** (`open("w")`) à chaque
    epoch. Une lecture peut donc tomber pile sur un fichier tronqué. On
    ne bricole pas — on renvoie None et l'appelant garde le dernier état
    valide jusqu'au prochain tour.
    """
    try:
        texte = csv_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    lignes = texte.splitlines()
    if len(lignes) < 2:
        return None

    lecteur = csv.DictReader(lignes)
    entetes = lecteur.fieldnames or []
    if "step" not in entetes:
        return None

    colonnes = [c for c in entetes if c and c != "step"]
    steps: list[int] = []
    valeurs: dict[str, list[float | None]] = {c: [] for c in colonnes}

    for ligne in lecteur:
        brut = (ligne.get("step") or "").strip()
        try:
            step = int(float(brut))
        except ValueError:
            continue  # ligne coupée en plein milieu d'une réécriture
        steps.append(step)
        for col in colonnes:
            valeurs[col].append(_nombre(ligne.get(col)))

    if not steps:
        return None

    # Une colonne entièrement vide (métrique loggée par un autre mode)
    # ne mérite pas un graphe vide.
    valeurs = {c: v for c, v in valeurs.items() if any(x is not None for x in v)}
    return {"steps": steps, "valeurs": valeurs}


def _nombre(brut: str | None) -> float | None:
    """Convertit une cellule en float, ou None (vide, NaN, texte)."""
    if brut is None:
        return None
    brut = brut.strip()
    if not brut:
        return None
    try:
        val = float(brut)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


# ----------------------------------------------------------------------
# Construction du payload servi à la page
# ----------------------------------------------------------------------
class Moniteur:
    """Relit les CSV à la demande et mémorise le dernier état valide."""

    def __init__(self, model_dir: Path | None, csvs: list[Path] | None) -> None:
        self.model_dir = model_dir
        self.csvs_fixes = csvs
        self.dernier_bon: dict[str, dict] = {}

    def _fichiers(self) -> list[Path]:
        if self.csvs_fixes:
            return [p for p in self.csvs_fixes if p.is_file()]
        # Re-scanne à chaque tour : le CSV n'existe pas encore quand on
        # lance le viewer avant l'entraînement.
        return trouver_stats(self.model_dir) if self.model_dir else []

    def payload(self) -> dict:
        maintenant = time.time()
        sources = []
        for chemin in self._fichiers():
            cle = str(chemin.resolve())
            donnees = lire_stats(chemin)
            partiel = donnees is None
            if partiel:
                donnees = self.dernier_bon.get(cle)
                if donnees is None:
                    continue
            else:
                self.dernier_bon[cle] = donnees

            try:
                mtime = chemin.stat().st_mtime
            except OSError:
                mtime = maintenant

            sources.append({
                "fichier": chemin.name,
                "chemin": str(chemin),
                "shuffle": chemin.parent.parent.name,
                "detecteur": "detector" in chemin.name,
                "silence_s": round(maintenant - mtime, 1),
                "actif": (maintenant - mtime) < SILENCE_S,
                "epochs_prevues": epochs_prevues(chemin),
                "partiel": partiel,
                "steps": donnees["steps"],
                "valeurs": donnees["valeurs"],
            })

        # Le CSV du modèle de pose d'abord, le détecteur ensuite.
        sources.sort(key=lambda s: (s["detecteur"], s["fichier"]))
        return {
            "genere": datetime.now().strftime("%H:%M:%S"),
            "racine": str(self.model_dir) if self.model_dir else "",
            "sources": sources,
        }


# ----------------------------------------------------------------------
# Page web
# ----------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entraînement DLC — suivi</title>
<style>
  :root {
    --fond: #f7f7f8; --carte: #ffffff; --texte: #1b1b1f; --doux: #6b6b76;
    --trait: #e3e3e8; --accent: #2f6fdd; --ok: #1f9d5c; --alerte: #c8781a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --fond: #15151a; --carte: #1e1e25; --texte: #ececf1; --doux: #9a9aa6;
      --trait: #2e2e38; --accent: #6ea0ff; --ok: #4fc98a; --alerte: #e0a95c;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--fond); color: var(--texte);
    font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  h1 { font-size: 19px; margin: 0 0 4px; }
  h2 { font-size: 15px; margin: 0 0 2px; }
  .doux { color: var(--doux); font-size: 12.5px; }
  .barre {
    display: flex; flex-wrap: wrap; gap: 16px; align-items: baseline;
    justify-content: space-between; margin-bottom: 18px;
  }
  .pastille {
    display: inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 12px; font-weight: 600; border: 1px solid var(--trait);
  }
  .pastille.actif { color: var(--ok); border-color: var(--ok); }
  .pastille.arret { color: var(--alerte); border-color: var(--alerte); }
  .carte {
    background: var(--carte); border: 1px solid var(--trait); border-radius: 10px;
    padding: 14px 16px; margin-bottom: 16px;
  }
  .grille { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px; }
  .chiffres { display: flex; flex-wrap: wrap; gap: 20px; margin: 10px 0 2px; }
  .chiffre b { display: block; font-size: 20px; font-variant-numeric: tabular-nums; }
  .chiffre span { color: var(--doux); font-size: 12px; }
  svg.chart { width: 100%; height: auto; display: block; touch-action: none; }
  .legende { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; margin-top: 6px; }
  .legende i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
  .lecture { font-size: 12px; color: var(--doux); min-height: 18px; margin-top: 4px;
             font-variant-numeric: tabular-nums; }
  label.echelle { font-size: 12px; color: var(--doux); cursor: pointer; user-select: none; }
  .vide { color: var(--doux); padding: 40px 0; text-align: center; }
  code { background: rgba(127,127,127,.14); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
<div class="barre">
  <div>
    <h1>Entraînement DLC — suivi en direct</h1>
    <div class="doux" id="racine"></div>
  </div>
  <div class="doux">
    <label class="echelle"><input type="checkbox" id="log"> échelle log (pertes)</label>
    &nbsp;·&nbsp; rafraîchi <span id="horloge">—</span>
  </div>
</div>
<div id="contenu"><div class="vide">Lecture de <code>learning_stats.csv</code>…</div></div>

<script>
const REFRESH_MS = __REFRESH__;
const COULEURS = ["#2f6fdd", "#e07b39", "#1f9d5c", "#a457c9", "#c94f4f",
                  "#0f9ba8", "#8a8f98", "#b8a02e"];
const etat = { data: null, log: false, metas: new Map() };

document.getElementById("log").addEventListener("change", (e) => {
  etat.log = e.target.checked;
  rendre();
});

function joli(v) {
  if (v === null || v === undefined) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 100000)) return v.toExponential(2);
  return (Math.round(v * 10000) / 10000).toString();
}

function titreCourt(col) {
  return col.replace(/^losses\\//, "").replace(/^metrics\\//, "");
}

/* ---- dessin d'un graphe ------------------------------------------- */
function graphe(id, steps, series, log, large) {
  const W = large ? 1040 : 620, H = large ? 260 : 250, m = { t: 12, r: 14, b: 26, l: 56 };
  const pts = series.map(s => s.valeurs.map((v, i) => [steps[i], v])
                                       .filter(p => p[1] !== null));
  const plats = pts.flat();
  if (!plats.length) return { html: '<div class="vide">pas encore de valeur</div>', meta: null };

  let xmin = Math.min(...plats.map(p => p[0])), xmax = Math.max(...plats.map(p => p[0]));
  let ymin = Math.min(...plats.map(p => p[1])), ymax = Math.max(...plats.map(p => p[1]));
  if (xmax === xmin) xmax = xmin + 1;
  const logOk = log && ymin > 0;
  if (logOk) { ymin = Math.log10(ymin); ymax = Math.log10(ymax); }
  if (ymax === ymin) { ymax = ymin + Math.abs(ymin || 1) * 0.1; ymin -= Math.abs(ymin || 1) * 0.1; }
  const marge = (ymax - ymin) * 0.08;
  const positifs = plats.every(p => p[1] >= 0);
  ymin = (positifs && !logOk) ? Math.max(0, ymin - marge) : ymin - marge;
  ymax += marge;

  const px = x => m.l + (x - xmin) / (xmax - xmin) * (W - m.l - m.r);
  const py = y => {
    const v = logOk ? Math.log10(y) : y;
    return m.t + (1 - (v - ymin) / (ymax - ymin)) * (H - m.t - m.b);
  };

  let svg = `<svg class="chart" id="${id}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
  for (let i = 0; i <= 4; i++) {
    const y = m.t + i * (H - m.t - m.b) / 4;
    const val = ymax - i * (ymax - ymin) / 4;
    svg += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="var(--trait)"/>`;
    svg += `<text x="${m.l - 7}" y="${y + 4}" text-anchor="end" font-size="10"
             fill="var(--doux)">${joli(logOk ? Math.pow(10, val) : val)}</text>`;
  }
  const nbX = Math.min(6, Math.round(xmax - xmin) + 1);
  for (let i = 0; i < nbX; i++) {
    const x = xmin + i * (xmax - xmin) / Math.max(1, nbX - 1);
    svg += `<text x="${px(x)}" y="${H - 8}" text-anchor="middle" font-size="10"
             fill="var(--doux)">${Math.round(x)}</text>`;
  }
  pts.forEach((p, i) => {
    const couleur = series[i].couleur;
    if (p.length === 1) {
      svg += `<circle cx="${px(p[0][0])}" cy="${py(p[0][1])}" r="3" fill="${couleur}"/>`;
    } else if (p.length > 1) {
      const d = p.map(q => `${px(q[0]).toFixed(1)},${py(q[1]).toFixed(1)}`).join(" ");
      svg += `<polyline points="${d}" fill="none" stroke="${couleur}" stroke-width="1.8"
               stroke-linejoin="round"/>`;
    }
  });
  svg += `<line class="curseur" x1="0" y1="${m.t}" x2="0" y2="${H - m.b}"
           stroke="var(--doux)" stroke-dasharray="3 3" style="display:none"/></svg>`;
  return { html: svg, meta: { steps, series, xmin, xmax, px, W, m } };
}

function bloc(titre, id, steps, series, log, large) {
  const g = graphe(id, steps, series, log, large);
  if (g.meta) etat.metas.set(id, g.meta);
  const legende = series.map(s =>
    `<span><i style="background:${s.couleur}"></i>${s.label}</span>`).join("");
  return `<div class="carte"><h2>${titre}</h2>${g.html}
          <div class="legende">${legende}</div>
          <div class="lecture" id="lec-${id}"></div></div>`;
}

/* ---- survol : valeurs à l'epoch le plus proche --------------------- */
document.addEventListener("mousemove", (e) => {
  const svg = e.target.closest ? e.target.closest("svg.chart") : null;
  if (!svg) return;
  const meta = etat.metas.get(svg.id);
  if (!meta) return;
  const boite = svg.getBoundingClientRect();
  const xVue = (e.clientX - boite.left) / boite.width * meta.W;
  const ratio = (xVue - meta.m.l) / (meta.px(meta.xmax) - meta.px(meta.xmin));
  const cible = meta.xmin + ratio * (meta.xmax - meta.xmin);
  let idx = 0, ecart = Infinity;
  meta.steps.forEach((s, i) => {
    const d = Math.abs(s - cible);
    if (d < ecart) { ecart = d; idx = i; }
  });
  const curseur = svg.querySelector(".curseur");
  if (curseur) {
    const x = meta.px(meta.steps[idx]);
    curseur.setAttribute("x1", x); curseur.setAttribute("x2", x);
    curseur.style.display = "";
  }
  const lec = document.getElementById("lec-" + svg.id);
  if (lec) {
    lec.textContent = "epoch " + meta.steps[idx] + " — " + meta.series
      .map(s => s.label + " " + joli(s.valeurs[idx])).join("  ·  ");
  }
});

/* ---- rendu --------------------------------------------------------- */
function rendre() {
  const data = etat.data;
  if (!data) return;
  etat.metas.clear();
  document.getElementById("racine").textContent = data.racine;
  document.getElementById("horloge").textContent = data.genere;

  if (!data.sources.length) {
    document.getElementById("contenu").innerHTML =
      `<div class="vide">Aucun <code>learning_stats.csv</code> pour l'instant.<br>
       La page se remplira dès le premier epoch de <code>02_train.py</code>.</div>`;
    return;
  }

  let html = "";
  for (const src of data.sources) {
    const cols = Object.keys(src.valeurs);
    const dernier = src.steps[src.steps.length - 1];
    const total = src.epochs_prevues ? " / " + src.epochs_prevues : "";
    const etatTxt = src.actif
      ? `<span class="pastille actif">en cours</span>`
      : `<span class="pastille arret">silencieux depuis ${Math.round(src.silence_s)} s</span>`;

    html += `<div class="carte">
      <div class="barre" style="margin:0">
        <div><h2>${src.detecteur ? "Détecteur" : "Modèle de pose"} — ${src.shuffle}</h2>
             <div class="doux">${src.chemin}</div></div>
        <div>${etatTxt}</div>
      </div>
      <div class="chiffres"><div class="chiffre"><b>${dernier}${total}</b><span>epoch</span></div>`;
    for (const col of cols) {
      const serie = src.valeurs[col];
      let v = null;
      for (let i = serie.length - 1; i >= 0; i--) { if (serie[i] !== null) { v = serie[i]; break; } }
      html += `<div class="chiffre"><b>${joli(v)}</b><span>${titreCourt(col)}</span></div>`;
    }
    html += `</div></div>`;

    // Graphe principal : train vs eval, les deux courbes qui décident.
    const principales = ["losses/train.total_loss", "losses/eval.total_loss"]
      .filter(c => cols.includes(c))
      .map((c, i) => ({ label: titreCourt(c), couleur: COULEURS[i], valeurs: src.valeurs[c] }));
    const id = (src.fichier + src.shuffle).replace(/[^a-zA-Z0-9]/g, "");
    if (principales.length) {
      html += bloc("Perte totale — train vs eval", "p" + id, src.steps, principales,
                   etat.log, true);
    }

    // Le reste en petits multiples : une courbe par colonne.
    const restantes = cols.filter(c => !principales.some(p => titreCourt(c) === p.label));
    if (restantes.length) {
      html += `<div class="grille">`;
      restantes.forEach((col, i) => {
        const serie = [{ label: titreCourt(col), couleur: COULEURS[i % COULEURS.length],
                         valeurs: src.valeurs[col] }];
        const log = etat.log && col.startsWith("losses/");
        html += bloc(titreCourt(col), "c" + id + i, src.steps, serie, log);
      });
      html += `</div>`;
    }
  }
  document.getElementById("contenu").innerHTML = html;
}

async function tick() {
  try {
    const r = await fetch("data.json", { cache: "no-store" });
    etat.data = await r.json();
    rendre();
  } catch (e) {
    document.getElementById("horloge").textContent = "serveur injoignable";
  }
}
tick();
setInterval(tick, REFRESH_MS);
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    """Deux routes : la page, et le JSON qu'elle relit toutes les N secondes."""

    moniteur: Moniteur
    refresh_ms: int

    def do_GET(self) -> None:  # noqa: N802 — nom imposé par BaseHTTPRequestHandler
        chemin = self.path.split("?", 1)[0]
        if chemin in ("/", "/index.html"):
            page = PAGE.replace("__REFRESH__", str(self.refresh_ms))
            self._repondre(page.encode("utf-8"), "text/html; charset=utf-8")
        elif chemin == "/data.json":
            corps = json.dumps(self.moniteur.payload()).encode("utf-8")
            self._repondre(corps, "application/json; charset=utf-8")
        else:
            self.send_error(404)

    def _repondre(self, corps: bytes, mime: str) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(corps)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corps)
        except (BrokenPipeError, ConnectionResetError):
            pass  # onglet fermé en plein chargement, rien à signaler

    def log_message(self, *args) -> None:
        """Pas de log par requête : la page interroge toutes les 5 s."""


def servir(moniteur: Moniteur, host: str, port: int, refresh_ms: int,
           ouvrir: bool) -> None:
    Handler.moniteur = moniteur
    Handler.refresh_ms = refresh_ms

    serveur = None
    for essai in range(port, port + 10):
        try:
            serveur = http.server.ThreadingHTTPServer((host, essai), Handler)
            port = essai
            break
        except OSError:
            continue
    if serveur is None:
        print(f"❌ Aucun port libre entre {port} et {port + 9}. "
              f"Relance avec --port <autre>.", file=sys.stderr)
        sys.exit(1)

    affiche = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{affiche}:{port}"
    print(f"📈 Suivi d'entraînement sur {url}")
    print(f"   Rafraîchissement toutes les {refresh_ms / 1000:g} s. Ctrl+C pour arrêter.")
    if ouvrir:
        webbrowser.open(url)
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du suivi. L'entraînement, lui, continue.")
    finally:
        serveur.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_dir_arg(parser)
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Dossier du projet DLC (celui qui contient "
                             "config.yaml et dlc-models-pytorch/). Court-circuite "
                             "--config-dir.")
    parser.add_argument("--stats", type=Path, nargs="+", default=None,
                        help="Un ou plusieurs learning_stats.csv à suivre "
                             "directement. Court-circuite --model-dir.")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port d'écoute (défaut 8765, +1 si occupé).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 pour ouvrir la page aux autres machines du LAN.")
    parser.add_argument("--refresh", type=float, default=5.0,
                        help="Intervalle de rafraîchissement en secondes (défaut 5).")
    parser.add_argument("--no-browser", action="store_true",
                        help="Ne pas ouvrir le navigateur au démarrage.")
    args = parser.parse_args()

    if args.stats:
        manquants = [p for p in args.stats if not p.is_file()]
        if manquants:
            print("❌ Fichier introuvable : "
                  + ", ".join(str(p) for p in manquants), file=sys.stderr)
            sys.exit(1)
        moniteur = Moniteur(None, [p.resolve() for p in args.stats])
    else:
        if args.model_dir is not None:
            model_dir = args.model_dir.resolve()
            if not model_dir.is_dir():
                print(f"❌ Dossier introuvable : {model_dir}", file=sys.stderr)
                sys.exit(1)
        else:
            # Même flag et même menu que 02_train.py : on suit le modèle
            # qu'on est en train d'entraîner, sans retaper un chemin.
            load_config(args)
            from _config import PROJECT_DIR  # noqa: E402 — après load_config
            model_dir = Path(PROJECT_DIR)
            if not model_dir.is_dir():
                print(f"❌ Le projet DLC n'existe pas encore : {model_dir}\n"
                      f"   Lance d'abord 01_setup_project.py.", file=sys.stderr)
                sys.exit(1)

        moniteur = Moniteur(model_dir, None)
        trouves = trouver_stats(model_dir)
        if trouves:
            print(f"ℹ  {len(trouves)} fichier(s) de stats trouvé(s) sous {model_dir}")
        else:
            print(f"ℹ  Pas encore de learning_stats.csv sous {model_dir} — "
                  f"la page se remplira au premier epoch.")

    servir(moniteur, args.host, args.port,
           int(max(1.0, args.refresh) * 1000), not args.no_browser)


if __name__ == "__main__":
    main()
