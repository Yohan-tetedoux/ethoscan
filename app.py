"""
EthoScan : application Streamlit de scan sampling pour l'éthologie.
Inspirée de SAWA et ZooMonitor ; pensée pour ordinateur, iPad et iPhone.

Lancer :  streamlit run app.py

Principes de conception
-----------------------
* TOUT l'état vit dans `st.session_state` (observations, scan, chrono, sélections,
  dernier clic) : un clic ou une mise à jour ne fait jamais perdre de données.
* Les actions (boutons) passent par des callbacks `on_click` : elles s'exécutent
  AVANT le rerun, donc l'état est cohérent au moment du redessin.
* Le chrono est calculé à partir de l'horloge (`time.time()`), jamais en
  décrémentant un compteur : il reste juste même si le rerun est lent.
* Chaque observation est aussi écrite dans un CSV d'autosauvegarde sur le serveur
  (dossier `autosave/`) au cas où l'onglet du navigateur serait rechargé.
"""

import io
import math
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
# Colonnes du CSV exporté (tidy data : une ligne = une observation)
COLUMNS = ["Session", "Observateur", "Scan_Num", "Timestamp", "Individu",
           "Comportement", "Modificateur", "X_norm", "Y_norm"]

DEFAULT_INDIVIDUS = "Ind_1\nInd_2\nInd_3"
DEFAULT_COMPORTEMENTS = ("Repos\nLocomotion\nAlimentation\nExploration\n"
                         "Agonistique\nStéréotypie")
DEFAULT_MODIFICATEURS = "Seul\nEn groupe\nSollicitation"

AUTOSAVE_DIR = Path("autosave")
MAX_DISPLAY_PX = 1400  # on réduit les grandes images (satellite, photos iPhone)

st.set_page_config(page_title="EthoScan", page_icon="🐾", layout="wide")

# CSS : gros boutons tactiles (≥ 4 rem de haut) et chrono très lisible
st.markdown(
    """
    <style>
    div.stButton > button, div.stDownloadButton > button {
        min-height: 4.2rem; font-size: 1.25rem; font-weight: 600;
        border-radius: 14px; touch-action: manipulation;
    }
    .timer-box {text-align:center; padding:.6rem; border-radius:16px;
                background: rgba(128,128,128,.12);}
    .timer-box .time {font-size:4rem; font-weight:700; line-height:1.1;
                      font-variant-numeric: tabular-nums;}
    .timer-box .label {font-size:1rem; opacity:.75;}
    .timer-box.expired {background: rgba(220,40,40,.22);}
    .timer-box.expired .time {color:#d62828;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Initialisation de l'état (exécutée à chaque rerun, n'écrase jamais l'existant)
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "observations": [],      # liste de dicts (une par observation)
        "scan_num": 0,           # 0 = scan sampling pas encore démarré
        "timer_running": False,
        "timer_deadline": 0.0,   # instant (time.time()) où le scan se termine
        "timer_remaining": None, # secondes restantes quand le chrono est en pause
        "sel_ind": None,         # sélections en cours
        "sel_beh": None,
        "sel_mod": None,
        "last_click": None,      # {"x": .., "y": ..} normalisés dans [0, 1]
        "click_seen_ts": None,   # horodatage du dernier clic déjà traité
        "image_bytes": None,     # image de l'enclos (octets)
        "flash": None,           # message (type, texte) affiché après une action
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()
ss = st.session_state


# --------------------------------------------------------------------------- #
# Fonctions utilitaires
# --------------------------------------------------------------------------- #
def parse_list(text: str) -> list[str]:
    """Une entrée par ligne ; lignes vides et doublons supprimés (ordre conservé)."""
    return list(dict.fromkeys(l.strip() for l in text.splitlines() if l.strip()))


def interval_s() -> int:
    return int(ss.get("cfg_interval", 60))


def remaining_s() -> float:
    """Secondes restantes dans le scan courant."""
    if ss.scan_num == 0:
        return float(interval_s())
    if ss.timer_running:
        return max(0.0, ss.timer_deadline - time.time())
    return float(ss.timer_remaining if ss.timer_remaining is not None else interval_s())


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()) or "session"


def observations_df() -> pd.DataFrame:
    return pd.DataFrame(ss.observations, columns=COLUMNS)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV « séparateur point-virgule » (format Excel FR / read.csv2 de R) :
    séparateur ';', décimale ',', UTF-8 avec BOM pour que Excel lise les accents."""
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def autosave() -> None:
    """Copie de secours sur disque (échec silencieux : ne doit jamais bloquer)."""
    try:
        AUTOSAVE_DIR.mkdir(exist_ok=True)
        path = AUTOSAVE_DIR / f"{slug(ss.get('cfg_session', 'session'))}.csv"
        path.write_bytes(to_csv_bytes(observations_df()))
    except OSError:
        pass


@st.cache_data(show_spinner=False)
def load_base_image(data: bytes) -> Image.Image:
    """Décode l'image, corrige l'orientation EXIF (photos iPhone) et la réduit."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    img = img.convert("RGB")
    img.thumbnail((MAX_DISPLAY_PX, MAX_DISPLAY_PX))
    return img


def render_image_with_markers(base: Image.Image, show_all: bool) -> Image.Image:
    """Copie de l'image avec le dernier point cliqué (rouge) et, option, les
    points déjà enregistrés (bleu)."""
    img = base.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    r = max(8, w // 60)
    if show_all:
        for o in ss.observations:
            if pd.notna(o["X_norm"]) and pd.notna(o["Y_norm"]):
                cx, cy = o["X_norm"] * w, o["Y_norm"] * h
                draw.ellipse([cx - r / 2, cy - r / 2, cx + r / 2, cy + r / 2],
                             fill=(30, 110, 255), outline="white", width=2)
    if ss.last_click:
        cx, cy = ss.last_click["x"] * w, ss.last_click["y"] * h
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(230, 30, 30), outline="white", width=3)
        draw.line([cx - 2 * r, cy, cx + 2 * r, cy], fill=(230, 30, 30), width=2)
        draw.line([cx, cy - 2 * r, cx, cy + 2 * r], fill=(230, 30, 30), width=2)
    return img


# --------------------------------------------------------------------------- #
# Callbacks (s'exécutent avant le rerun)
# --------------------------------------------------------------------------- #
def start_or_resume() -> None:
    if ss.scan_num == 0:                       # tout premier démarrage
        ss.scan_num = 1
        ss.timer_remaining = float(interval_s())
    ss.timer_deadline = time.time() + ss.timer_remaining
    ss.timer_running = True


def pause() -> None:
    ss.timer_remaining = remaining_s()
    ss.timer_running = False


def next_scan() -> None:
    ss.scan_num += 1
    ss.timer_remaining = float(interval_s())
    ss.timer_deadline = time.time() + ss.timer_remaining
    ss.timer_running = True


def choose(state_key: str, option: str, optional: bool) -> None:
    """Sélectionne une option ; re-cliquer sur une option facultative la désélectionne."""
    if optional and ss[state_key] == option:
        ss[state_key] = None
    else:
        ss[state_key] = option


def save_observation() -> None:
    if ss.scan_num == 0:
        ss.flash = ("error", "Démarrez le scan (▶ Démarrer) avant d'enregistrer.")
        return
    if not (ss.sel_ind and ss.sel_beh):
        ss.flash = ("error", "Choisissez au moins un individu et un comportement.")
        return
    pt = ss.last_click
    ss.observations.append({
        "Session": ss.cfg_session,
        "Observateur": ss.cfg_observer,
        "Scan_Num": ss.scan_num,
        "Timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "Individu": ss.sel_ind,
        "Comportement": ss.sel_beh,
        "Modificateur": ss.sel_mod,                      # None → NA dans R
        "X_norm": pt["x"] if pt else None,
        "Y_norm": pt["y"] if pt else None,
    })
    ss.flash = ("success", f"✅ Enregistré : {ss.sel_ind} · {ss.sel_beh}"
                           + (" · position" if pt else " · SANS position"))
    # On remet à zéro les sélections ET le point : évite de réutiliser par erreur
    # la position de l'individu précédent.
    ss.sel_ind = ss.sel_beh = ss.sel_mod = None
    ss.last_click = None
    autosave()


def delete_last() -> None:
    if ss.observations:
        ss.observations.pop()
        ss.flash = ("info", "Dernière ligne supprimée.")
        autosave()


def reset_session() -> None:
    ss.observations = []
    ss.scan_num = 0
    ss.timer_running = False
    ss.timer_remaining = None
    ss.sel_ind = ss.sel_beh = ss.sel_mod = None
    ss.last_click = None
    ss.cfg_confirm_reset = False
    ss.flash = ("info", "Session réinitialisée.")


# --------------------------------------------------------------------------- #
# Onglets
# --------------------------------------------------------------------------- #
tab_obs, tab_cfg = st.tabs(["👁️ Observation", "⚙️ Configuration"])

# ============================ CONFIGURATION ================================ #
with tab_cfg:
    st.subheader("Session")
    c1, c2, c3 = st.columns(3)
    c1.text_input("Nom de la session", value="Session_1", key="cfg_session")
    c2.text_input("Observateur", value="", key="cfg_observer")
    c3.number_input("Intervalle de scan (s)", min_value=5, max_value=3600,
                    value=60, step=5, key="cfg_interval")

    st.subheader("Listes (une entrée par ligne, modifiables à tout moment)")
    l1, l2, l3 = st.columns(3)
    l1.text_area("Individus", value=DEFAULT_INDIVIDUS, height=200, key="cfg_individus")
    l2.text_area("Comportements", value=DEFAULT_COMPORTEMENTS, height=200,
                 key="cfg_comportements")
    l3.text_area("Modificateurs / Contextes", value=DEFAULT_MODIFICATEURS,
                 height=200, key="cfg_modificateurs")

    st.subheader("Plan de l'enclos")
    up = st.file_uploader("Image (JPG / PNG) : plan ou capture satellite",
                          type=["jpg", "jpeg", "png"])
    if up is not None:
        ss.image_bytes = up.getvalue()   # copie dans session_state
    if ss.image_bytes and st.button("Retirer l'image"):
        ss.image_bytes = None
        ss.last_click = None
        st.rerun()

    with st.expander("⚠️ Nouvelle session (efface toutes les données)"):
        st.checkbox("Je confirme vouloir tout effacer", key="cfg_confirm_reset")
        st.button("Réinitialiser", on_click=reset_session,
                  disabled=not ss.get("cfg_confirm_reset", False))

# Listes courantes, relues à chaque rerun
individus = parse_list(ss.cfg_individus)
comportements = parse_list(ss.cfg_comportements)
modificateurs = parse_list(ss.cfg_modificateurs)

# Si une liste a été modifiée, on invalide une sélection devenue inexistante
for key, options in (("sel_ind", individus), ("sel_beh", comportements),
                     ("sel_mod", modificateurs)):
    if ss[key] not in options:
        ss[key] = None


def button_grid(options: list[str], state_key: str, optional: bool = False,
                n_cols: int = 3) -> None:
    """Grille de gros boutons ; le bouton sélectionné est en couleur pleine."""
    if not options:
        st.warning("Liste vide : ajoutez des entrées dans l'onglet Configuration.")
        return
    for start in range(0, len(options), n_cols):
        cols = st.columns(n_cols)
        for col, opt in zip(cols, options[start:start + n_cols]):
            col.button(opt, key=f"{state_key}__{opt}", width="stretch",
                       type="primary" if ss[state_key] == opt else "secondary",
                       on_click=choose, args=(state_key, opt, optional))


def timer_display() -> None:
    """Affichage du chrono (rafraîchi chaque seconde sans relancer toute la page)."""
    rem = remaining_s()
    expired = ss.scan_num > 0 and rem <= 0
    mm, sec = divmod(math.ceil(rem), 60)
    if ss.scan_num == 0:
        status = "Non démarré"
    elif expired:
        status = "⏰ Scan terminé : passez au scan suivant"
    else:
        status = "En cours" if ss.timer_running else "⏸ En pause"
    st.markdown(
        f'<div class="timer-box{" expired" if expired else ""}">'
        f'<div class="label">Scan n° {ss.scan_num} · {status}</div>'
        f'<div class="time">{mm:02d}:{sec:02d}</div></div>',
        unsafe_allow_html=True)
    st.progress(min(1.0, max(0.0, 1 - rem / max(1, interval_s()))))


# ============================== OBSERVATION ================================ #
with tab_obs:
    # --- Chrono + contrôles ------------------------------------------------ #
    t_col, b_col = st.columns([2, 3])
    with t_col:
        # run_every=1 uniquement quand le chrono tourne (pas de rerun inutile)
        st.fragment(timer_display, run_every=1 if ss.timer_running else None)()
    with b_col:
        b1, b2 = st.columns(2)
        if ss.timer_running:
            b1.button("⏸ Pause", on_click=pause, width="stretch")
        else:
            b1.button("▶ Démarrer" if ss.scan_num == 0 else "▶ Reprendre",
                      on_click=start_or_resume, width="stretch", type="primary")
        b2.button("⏭ Scan suivant", on_click=next_scan, width="stretch",
                  disabled=ss.scan_num == 0)

    if ss.flash:
        getattr(st, ss.flash[0])(ss.flash[1])
        ss.flash = None

    left, right = st.columns(2)

    # --- Image cliquable --------------------------------------------------- #
    with left:
        st.markdown("#### 📍 Position")
        if ss.image_bytes:
            show_all = st.checkbox("Afficher les points déjà enregistrés")
            base = load_base_image(ss.image_bytes)
            click = streamlit_image_coordinates(
                render_image_with_markers(base, show_all),
                key="enclos", width="stretch")
            # Le composant renvoie le même dict à chaque rerun tant qu'il n'y a pas
            # de nouveau clic : on ne traite donc que les clics au timestamp inédit.
            if click and click.get("unix_time") != ss.click_seen_ts:
                ss.click_seen_ts = click.get("unix_time")
                ss.last_click = {
                    "x": round(min(1.0, max(0.0, click["x"] / click["width"])), 5),
                    "y": round(min(1.0, max(0.0, click["y"] / click["height"])), 5),
                }
                st.rerun()   # redessine l'image avec le marqueur
            if ss.last_click:
                st.success(f"Dernier point : X = {ss.last_click['x']:.3f} · "
                           f"Y = {ss.last_click['y']:.3f}")
            else:
                st.info("Touchez l'image pour placer l'individu.")
        else:
            st.info("Aucune image : importez un plan dans l'onglet Configuration. "
                    "Les observations seront enregistrées sans position (X/Y = NA).")

    # --- Sélection Individu → Comportement → Modificateur ------------------ #
    with right:
        st.markdown("#### 1️⃣ Individu")
        button_grid(individus, "sel_ind")
        st.markdown("#### 2️⃣ Comportement")
        button_grid(comportements, "sel_beh")
        st.markdown("#### 3️⃣ Modificateur (facultatif)")
        button_grid(modificateurs, "sel_mod", optional=True)

    # Récapitulatif + enregistrement
    pt = ss.last_click
    st.markdown(
        f"**Sélection :** {ss.sel_ind or '—'} · {ss.sel_beh or '—'} · "
        f"{ss.sel_mod or '—'} · "
        + (f"({pt['x']:.3f}, {pt['y']:.3f})" if pt else "pas de position"))
    st.button("💾 Enregistrer l'observation", type="primary", width="stretch",
              on_click=save_observation,
              disabled=not (ss.sel_ind and ss.sel_beh))

    # --- Tableau + export -------------------------------------------------- #
    st.divider()
    df = observations_df()
    st.markdown(f"#### 📋 Observations ({len(df)})")
    if df.empty:
        st.caption("Aucune observation enregistrée pour l'instant.")
    else:
        st.dataframe(df.set_index(df.index + 1), width="stretch", height=300)

    d1, d2 = st.columns(2)
    d1.button("🗑️ Supprimer la dernière ligne", on_click=delete_last,
              width="stretch", disabled=df.empty)
    d2.download_button(
        "⬇️ Télécharger les données (CSV)",
        data=to_csv_bytes(df),
        file_name=f"{slug(ss.cfg_session)}_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv", width="stretch", disabled=df.empty)
