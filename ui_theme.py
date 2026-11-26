"""Shared Streamlit look-and-feel for app.py (local) and app_backend.py (API client).

Keeps both front ends visually identical and tidy: a gradient hero header, carded
sections, rounded media, and a consistent primary button. Theme-safe (works in both
Streamlit light and dark modes).
"""
import streamlit as st

_BASE_CSS = """
<style>
/* ---- hide Streamlit chrome ---- */
header[data-testid="stHeader"],
.stDeployButton,
div[data-testid="stToolbarActions"],
#MainMenu { display: none !important; }
section[data-testid="stSidebar"]                 { display: none !important; }
section[data-testid="stSidebarCollapsedControl"] { display: none !important; }
@media print { html, body, div { display: none !important; visibility: hidden !important; } }

/* ================================================================
   STARFIELD DEMO THEME — black sky, white stars, purple accents
   ================================================================ */

/* ---- full-page starfield ----------------------------------------
   Layer A: 3 cross-spike stars  (vertical + horizontal light spikes
            + glowing core — like bright stars through a telescope)
   Layer B: 8 glow stars         (bright core + soft coloured halo)
   Layer C: 3 tiled dot fields   (coprime tile sizes → no visible grid)
   ---------------------------------------------------------------- */
.stApp {
  background-color: #000005;
  background-image:

    /* === CROSS-SPIKE STARS (3 × 3 layers each) === */
    /* Star A — 18% 20% — cool blue-white */
    radial-gradient(ellipse 1.5px 12px at 18% 20%, rgba(255,255,255,.80) 0%, transparent 100%),
    radial-gradient(ellipse 12px 1.5px at 18% 20%, rgba(255,255,255,.80) 0%, transparent 100%),
    radial-gradient(circle at 18% 20%, #fff 0px, #fff 2.5px, rgba(190,215,255,.45) 7px, transparent 14px),

    /* Star B — 79% 13% — warm white */
    radial-gradient(ellipse 1.5px 12px at 79% 13%, rgba(255,255,255,.75) 0%, transparent 100%),
    radial-gradient(ellipse 12px 1.5px at 79% 13%, rgba(255,255,255,.75) 0%, transparent 100%),
    radial-gradient(circle at 79% 13%, #fff 0px, #fff 2px, rgba(255,240,200,.42) 6px, transparent 13px),

    /* Star C — 51% 91% — lavender-white */
    radial-gradient(ellipse 1.5px 10px at 51% 91%, rgba(255,255,255,.70) 0%, transparent 100%),
    radial-gradient(ellipse 10px 1.5px at 51% 91%, rgba(255,255,255,.70) 0%, transparent 100%),
    radial-gradient(circle at 51% 91%, #fff 0px, #fff 2px, rgba(210,190,255,.40) 6px, transparent 12px),

    /* === GLOW STARS (8 × 1 layer: core + coloured halo) === */
    radial-gradient(circle at  8% 14%, #fff 0px, #fff 1.5px, rgba(190,215,255,.48) 4px, transparent 10px),
    radial-gradient(circle at 27%  7%, #fff 0px, #fff 1px,   rgba(255,245,210,.42) 3px, transparent  8px),
    radial-gradient(circle at 65%  5%, #fff 0px, #fff 1.5px, rgba(190,215,255,.44) 4px, transparent  9px),
    radial-gradient(circle at 93% 26%, #fff 0px, #fff 1px,   rgba(220,195,255,.40) 3px, transparent  8px),
    radial-gradient(circle at 88% 70%, #fff 0px, #fff 1.5px, rgba(255,245,210,.44) 4px, transparent 10px),
    radial-gradient(circle at 42% 86%, #fff 0px, #fff 1px,   rgba(190,215,255,.38) 3px, transparent  8px),
    radial-gradient(circle at 11% 76%, #fff 0px, #fff 1.5px, rgba(210,190,255,.42) 4px, transparent  9px),
    radial-gradient(circle at 36% 54%, #fff 0px, #fff 1px,   rgba(190,215,255,.32) 3px, transparent  7px),

    /* === TILED DOT FIELDS — 8 layers, all coprime sizes, all different offsets ===
       Each layer contributes dots at a shifted grid position.  The aggregate of 8
       independent grids with incommensurable periods looks genuinely random.
       Brightness drops with density so the overall tone is a natural night sky. */
    radial-gradient(rgba(255,255,255,.92) 1px, transparent 0),  /* 557×373   sparse bright  */
    radial-gradient(rgba(255,255,255,.78) 1px, transparent 0),  /* 431×293                  */
    radial-gradient(rgba(255,255,255,.64) 1px, transparent 0),  /* 317×211                  */
    radial-gradient(rgba(255,255,255,.52) 1px, transparent 0),  /* 251×167                  */
    radial-gradient(rgba(255,255,255,.40) 1px, transparent 0),  /* 193×131                  */
    radial-gradient(rgba(255,255,255,.30) 1px, transparent 0),  /* 149×107                  */
    radial-gradient(rgba(255,255,255,.20) 1px, transparent 0),  /* 97×71                    */
    radial-gradient(rgba(255,255,255,.12) 1px, transparent 0);  /* 61×47    dense dim       */

  background-size:
    /* cross-spike + glow stars: each fills 100% of viewport */
    100% 100%, 100% 100%, 100% 100%,
    100% 100%, 100% 100%, 100% 100%,
    100% 100%, 100% 100%, 100% 100%,
    100% 100%, 100% 100%, 100% 100%, 100% 100%,
    100% 100%, 100% 100%, 100% 100%, 100% 100%,
    /* tiled layers — 8 different tile sizes */
    557px 373px, 431px 293px, 317px 211px, 251px 167px,
    193px 131px, 149px 107px, 97px 71px, 61px 47px;

  background-position:
    /* feature + glow stars */
    0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0,
    0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0,
    /* tiled layers: each at a unique offset so the 8 grids don't align */
    0 0, 61px 157px, 23px 47px, 113px 83px,
    37px 109px, 71px 29px, 53px 61px, 17px 41px;

  background-repeat:
    /* feature + glow: no-repeat (positioned by % inside the gradient) */
    no-repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat,
    no-repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat,
    no-repeat, no-repeat, no-repeat, no-repeat, no-repeat,
    /* tiled layers: repeat */
    repeat, repeat, repeat, repeat, repeat, repeat, repeat, repeat;

  min-height: 100vh;
}
section[data-testid="stMain"],
.main .block-container { background: transparent !important; }

/* ---- layout & type ---- */
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1150px; }
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

/* ---- body text: crisp white on black ---- */
p, li, span:not(.sts-pill):not(.sts-step),
div[data-testid="stMarkdownContainer"] p { color: #e8e8ff; }
label { color: #a5a5cc !important; font-weight: 500 !important; }
.stCaptionContainer p,
div[data-testid="stCaptionContainer"] { color: #5a5a7a !important; }

/* ---- hero banner: vivid gradient floats on black like a neon sign ---- */
.sts-hero {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
  color: #fff; padding: 1.4rem 1.8rem; border-radius: 18px; margin: 0 0 1.3rem 0;
  box-shadow:
    0 0 60px rgba(99,102,241,.60),
    0 0 20px rgba(236,72,153,.40),
    0 8px 32px rgba(0,0,0,.8);
}
.sts-hero h1 {
  color: #fff; font-size: 1.65rem; font-weight: 800;
  margin: 0 0 .3rem 0; letter-spacing: -.02em;
  text-shadow: 0 0 24px rgba(255,255,255,.4);
}
.sts-hero p { color: rgba(255,255,255,.88); margin: 0; font-size: .97rem; }

/* ---- section cards: dark glass panel floating above the stars ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid rgba(139,92,246,.28) !important;
  border-radius: 20px !important;
  background: rgba(8,4,20,.82) !important;
  box-shadow:
    0 0 40px rgba(139,92,246,.14),
    0 0 0 1px rgba(139,92,246,.07) inset,
    0 8px 40px rgba(0,0,0,.85) !important;
  backdrop-filter: blur(18px) !important;
}

/* ---- primary button: purple → pink ---- */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #7c3aed 0%, #c026d3 55%, #db2777 100%) !important;
  border: none !important; border-radius: 11px !important;
  color: #fff !important; font-weight: 700 !important; font-size: 1rem !important;
  letter-spacing: .02em !important;
  box-shadow: 0 4px 22px rgba(124,58,237,.55) !important;
  transition: transform .15s ease, box-shadow .15s ease !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 7px 30px rgba(124,58,237,.75) !important;
  filter: brightness(1.1) !important;
}
.stButton > button[kind="primary"]:active {
  transform: scale(0.96) translateY(1px) !important;
  box-shadow: 0 2px 10px rgba(124,58,237,.35) !important;
  filter: brightness(0.95) !important;
}

/* ---- secondary buttons ---- */
.stButton > button[kind="secondary"] {
  background: rgba(124,58,237,.1) !important;
  border: 1px solid rgba(124,58,237,.35) !important;
  color: #c084fc !important; border-radius: 9px !important;
  transition: background .15s, border-color .15s !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(124,58,237,.22) !important;
  border-color: rgba(192,132,252,.6) !important;
}

/* ---- expander ---- */
div[data-testid="stExpander"] > details {
  border: 1px solid rgba(124,58,237,.2) !important;
  border-radius: 11px !important;
  background: rgba(8,4,20,.6) !important;
}
div[data-testid="stExpander"] > details > summary {
  color: #a78bfa !important; font-weight: 600 !important;
}

/* ---- subheaders ---- */
h3 { color: #a78bfa !important; font-weight: 700 !important; }

/* ---- rounded media / inputs ---- */
audio { width: 100%; border-radius: 10px; }
.stTextArea textarea { border-radius: 10px; }
div[data-testid="stRadio"] label { font-weight: 500; }

/* ---- language / status pills ---- */
.sts-pill {
  display: inline-block; padding: .18rem .7rem; border-radius: 999px;
  font-size: .8rem; font-weight: 600; margin: 0 .3rem .3rem 0;
  background: rgba(99,102,241,.16); color: #818cf8;
  border: 1px solid rgba(99,102,241,.38);
}
.sts-pill.alt  { background: rgba(236,72,153,.16); color: #f472b6; border-color: rgba(236,72,153,.4); }
.sts-pill.good { background: rgba(16,185,129,.16);  color: #34d399; border-color: rgba(16,185,129,.4); }
.sts-step { font-weight: 700; font-size: 1.02rem; margin: 0 0 .5rem 0; }
</style>
"""


def configure_page(title: str = "Trilingual Emotion STS Engine"):
    """Call once, before any other Streamlit output: sets page config + injects the theme."""
    st.set_page_config(page_title=title, page_icon="🎙️", layout="wide",
                       initial_sidebar_state="auto")
    st.markdown(_BASE_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str = "", center: bool = False):
    """Gradient banner header. Pass subtitle='' and center=True for the post-login view."""
    align = "center" if center else "left"
    sub_html = (f"<p style='text-align:{align};margin:0'>{subtitle}</p>"
                if subtitle else "")
    st.markdown(
        f"<div class='sts-hero' style='text-align:{align}'>"
        f"<h1 style='text-align:{align}'>🎙️ {title}</h1>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def inject_audio_mutex():
    """Single JS injection that:
      1. Ensures only one <audio> plays at a time (mutual exclusion via 'play' listener).
      2. Hides the play/pause button that appears inside st.audio_input after recording
         — the mic/re-record button (always the FIRST button in the widget) is kept.
    Uses a zero-height component iframe so the script actually executes (st.markdown
    with unsafe_allow_html does not execute injected <script> tags via innerHTML).
    """
    import streamlit.components.v1 as _components
    _components.html(
        """<script>
        (function(){
            var d=window.parent.document;

            /* --- 1. Audio mutex: only one <audio> plays at once --- */
            function pauseOthers(x){
                d.querySelectorAll('audio').forEach(function(a){if(a!==x)a.pause();});
            }
            function attachMutex(el){
                if(!el._stsMutex){
                    el._stsMutex=true;
                    el.addEventListener('play',function(){pauseOthers(this);});
                }
            }

            /* --- 2. Hide playback controls inside st.audio_input ---
               The widget renders: [mic-button] [play-button] [waveform] [timer]
               We keep the FIRST button (mic / re-record) and hide everything after it. */
            function hideInputPlayback(w){
                var btns=w.querySelectorAll('button');
                for(var i=1;i<btns.length;i++){
                    btns[i].style.setProperty('display','none','important');
                }
            }
            function processAudioInputs(root){
                (root.querySelectorAll?root:d)
                    .querySelectorAll('[data-testid="stAudioInput"]')
                    .forEach(hideInputPlayback);
            }

            /* Run on existing DOM */
            d.querySelectorAll('audio').forEach(attachMutex);
            processAudioInputs(d);

            /* Watch for future elements */
            new window.parent.MutationObserver(function(ms){
                ms.forEach(function(m){
                    m.addedNodes.forEach(function(n){
                        if(n.nodeName==='AUDIO')attachMutex(n);
                        if(n.querySelectorAll){
                            n.querySelectorAll('audio').forEach(attachMutex);
                            processAudioInputs(n);
                        }
                    });
                });
            }).observe(d.body,{childList:true,subtree:true});
        })();
        </script>""",
        height=0,
    )


def pill(text: str, kind: str = "") -> str:
    """Return HTML for a small coloured pill (kind: '', 'alt', 'good')."""
    cls = f"sts-pill {kind}".strip()
    return f"<span class='{cls}'>{text}</span>"


def step(num: int, text: str):
    """Render a numbered step heading inside a section card."""
    st.markdown(f"<div class='sts-step'>{num}. {text}</div>", unsafe_allow_html=True)

