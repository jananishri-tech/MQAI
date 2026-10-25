from pathlib import Path
from datetime import datetime
import csv
import hashlib
import hmac
import json
import math
import os
import secrets
import smtplib
import threading
import time
from email.message import EmailMessage

import cv2
import numpy as np
from live_yolo import LiveYOLODetector
from detector import detect_defects
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ManufacturingQualityAI
# Credentials are unchanged: users.csv (email, salt, password_hash, role, created_at),
# PBKDF2-SHA256, 310,000 rounds. Existing accounts keep working.
# Everything in the app is complete except the trained ML model, which is not connected.
BASE = Path(__file__).resolve().parent
IMAGES = BASE / "inspection_images"
USERS = BASE / "users.csv"
REPORT = BASE / "inspection_report.csv"
ACTIONS = BASE / "action_log.csv"
PROCESS = BASE / "process_data.csv"
VERIFY = BASE / "verification_log.csv"
LIMITS_FILE = BASE / "process_limits.json"
IMAGES.mkdir(exist_ok=True)

FILE_LOCK = threading.Lock()
DEFECT = "POSSIBLE DEFECT"
CLEAR = "NO OBVIOUS DEFECT"
MAX_ATTEMPTS = 5
LOCK_SECONDS = 30

PARAMS = ["temperature", "pressure", "speed"]
LABELS = {"temperature": ("Temperature", "°C"), "pressure": ("Pressure", "bar"), "speed": ("Machine speed", "units/min")}
DEFAULT_LIMITS = {"temperature": [160.0, 200.0], "pressure": [4.0, 6.0], "speed": [80.0, 120.0]}
PROC_COLS = ["timestamp", "batch_id", "temperature", "pressure", "speed", "units_inspected", "defects_found"]

st.set_page_config(page_title="ManufacturingQualityAI", page_icon="🏭", layout="wide")

# ---------------------------------------------------------------- Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root{
  --canvas:#F5F8FA; --surface:#FFFFFF; --line:#DCE5EA; --ink:#1B2A33; --muted:#5B6F7C;
  --brand:#0F766E; --brand-dark:#0B5A54; --brand-soft:#E3F3F1;
  --slate:#16252E; --slate-soft:#233743;
  --ok:#15803D; --ok-soft:#E5F5EA; --warn:#B45309; --warn-soft:#FEF3DC; --bad:#B42318; --bad-soft:#FDE8E6;
}
html,body,.stApp,[class*="css"]{font-family:'Manrope',system-ui,-apple-system,'Segoe UI',sans-serif}
.stApp{background:var(--canvas);color:var(--ink)}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1280px;padding:1.6rem 2rem 3rem}
h1,h2,h3,h4{color:var(--ink)!important;letter-spacing:-0.01em}
p,li,label,.stMarkdown{color:var(--ink)}
.stCaption,small,[data-testid="stCaptionContainer"]{color:var(--muted)!important}

[data-testid="stSidebar"]{background:var(--slate);border-right:0}
[data-testid="stSidebar"] *{color:#E4EDF1}
[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#9FB4C0!important}
[data-testid="stSidebar"] [role="radiogroup"]{gap:2px}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:9px 12px;border-radius:10px;width:100%}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:var(--slate-soft)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:var(--brand)}
[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none}
[data-testid="stSidebar"] .stButton>button{background:transparent;border:1px solid #3A5361;color:#E4EDF1}
[data-testid="stSidebar"] .stButton>button:hover{background:var(--slate-soft);border-color:#5C7B8C}

.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{
  background:var(--brand);color:#fff;border:0;border-radius:10px;font-weight:700;padding:.55rem 1.2rem;min-height:2.7rem}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{background:var(--brand-dark);color:#fff}
.stButton>button:focus-visible,.stFormSubmitButton>button:focus-visible{outline:3px solid #7FD1C9;outline-offset:2px}
.stTextInput input,.stTextArea textarea,.stNumberInput input{
  background:#fff!important;color:var(--ink)!important;border:1px solid #C5D2DA!important;border-radius:10px!important}
.stTextInput input:focus,.stTextArea textarea:focus,.stNumberInput input:focus{border-color:var(--brand)!important;box-shadow:0 0 0 3px #CBEBE7!important}
.stSelectbox div[data-baseweb="select"]>div{background:#fff;color:var(--ink);border-radius:10px;border-color:#C5D2DA}

[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-weight:600}
[data-testid="stMetricValue"]{color:var(--ink)!important;font-weight:800}
[data-testid="stForm"]{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:26px 28px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px}
[data-testid="stExpander"]{background:var(--surface);border:1px solid var(--line);border-radius:12px}
button[data-baseweb="tab"]{font-weight:700}

.page-head{margin:0 0 1.2rem}
.page-head h1{font-size:1.75rem;font-weight:800;margin:0}
.page-head p{color:var(--muted);margin:.25rem 0 0}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px 20px}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:700;font-size:.85rem}
.pill.bad{background:var(--bad-soft);color:var(--bad)}
.pill.ok{background:var(--ok-soft);color:var(--ok)}
.pill.warn{background:var(--warn-soft);color:var(--warn)}
.step{display:flex;gap:12px;align-items:flex-start;margin:0 0 14px}
.step b{display:inline-flex;min-width:30px;height:30px;border-radius:50%;background:var(--brand-soft);color:var(--brand-dark);
  align-items:center;justify-content:center;font-weight:800}

.brand-panel{background:var(--slate);border-radius:20px;padding:40px 36px;min-height:430px;color:#E4EDF1}
.brand-panel h1{color:#fff!important;font-size:2rem;font-weight:800;margin:0 0 .5rem}
.brand-panel p{color:#B8CAD3;font-size:1.02rem;line-height:1.6;margin:0 0 1.4rem}
.brand-panel .step{color:#E4EDF1;margin-bottom:12px}
.brand-panel .step b{background:var(--brand);color:#fff}
.signin-title{font-size:1.5rem;font-weight:800;margin:0 0 .2rem}
.signin-sub{color:var(--muted);margin:0 0 1rem}
@media (max-width:900px){.block-container{padding:1rem}.brand-panel{min-height:0;padding:26px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- File helpers
def read_csv(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def append_row(path, row):
    """Append one row. If the row has new columns, the file is upgraded without losing data."""
    with FILE_LOCK:
        exists = path.exists() and path.stat().st_size > 0
        fieldnames = list(row)
        if exists:
            try:
                header = list(pd.read_csv(path, nrows=0).columns)
            except Exception:
                header = None
            if header is not None:
                if set(row) - set(header):
                    df = pd.read_csv(path)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    df.to_csv(path, index=False, encoding="utf-8")
                    return
                fieldnames = header
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def images():
    return sorted(
        [p for p in IMAGES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}],
        key=lambda p: p.stat().st_mtime, reverse=True
    )


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


# Work on older and newer Streamlit (use_container_width was replaced by width="stretch").
def show_image(img, **kw):
    try:
        st.image(img, width="stretch", **kw)
    except Exception:
        st.image(img, use_container_width=True, **kw)


def show_table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_chart(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#1B2A33",
                      margin=dict(t=30, b=10, l=5, r=5))
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def page_head(title, subtitle):
    st.markdown(f'<div class="page-head"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def pill(text, kind):
    return f'<span class="pill {kind}">{text}</span>'


def valid_email(email):
    email = email.strip()
    return "@" in email and "." in email.split("@")[-1] and " " not in email


# ---------------------------------------------------------------- Credentials (unchanged logic)
def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310_000).hex()
    return salt, digest


def users():
    return read_csv(USERS)


def create_user(email, password, role="user"):
    salt, digest = password_hash(password)
    append_row(USERS, {"email": email.strip().lower(), "salt": salt, "password_hash": digest,
                       "role": role, "created_at": datetime.now().isoformat(timespec="seconds")})


def authenticate(email, password):
    df = users()
    if df.empty:
        return None
    match = df[df["email"].astype(str).str.lower() == email.strip().lower()]
    if match.empty:
        return None
    row = match.iloc[0]
    _, digest = password_hash(password, str(row["salt"]))
    return str(row["role"]) if hmac.compare_digest(digest, str(row["password_hash"])) else None


# ---------------------------------------------------------------- Email
def smtp_configured():
    return bool(os.getenv("MQAI_SMTP_USER", "") and os.getenv("MQAI_SMTP_PASSWORD", ""))


def send_report(recipient, data, filename, subject="ManufacturingQualityAI report"):
    sender = os.getenv("MQAI_SMTP_USER", "")
    password = os.getenv("MQAI_SMTP_PASSWORD", "")
    host = os.getenv("MQAI_SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("MQAI_SMTP_PORT", "587"))
    if not sender or not password:
        raise RuntimeError("SMTP is not configured. Set MQAI_SMTP_USER and MQAI_SMTP_PASSWORD in the terminal.")
    msg = EmailMessage()
    msg["From"], msg["To"] = sender, recipient
    msg["Subject"] = subject
    msg.set_content("Attached is the requested report. Preliminary screening results require validation.")
    msg.add_attachment(data, maintype="text", subtype="csv", filename=filename)
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)


# ---------------------------------------------------------------- Crack screening (OpenCV)
KERNELS = [(15, 3), (3, 15), (11, 11)]  # horizontal, vertical and diagonal-ish dark lines


def analyze_image(img, threshold=30, min_length=40):
    """Find thin, elongated dark features (possible cracks) and describe them.
    This is a classical image-processing screen, not a trained model."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)  # even out lighting
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bh = None
    for kw, kh in KERNELS:
        r = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
        bh = r if bh is None else cv2.max(bh, r)
    _, mask = cv2.threshold(bh, int(threshold), 255, cv2.THRESH_BINARY)
    edge = max(4, int(min(h, w) * 0.02))  # ignore the frame border
    mask[:edge, :] = 0
    mask[-edge:, :] = 0
    mask[:, :edge] = 0
    mask[:, -edge:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img.copy()
    line_mask = np.zeros_like(mask)
    lengths = []
    for c in contours:
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        length, width = max(rw, rh), max(min(rw, rh), 1.0)
        if length >= min_length and length / width >= 4:  # long and thin = line-like
            lengths.append(length)
            cv2.drawContours(line_mask, [c], -1, 255, -1)
            x, y, bw, bh_ = cv2.boundingRect(c)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh_), (0, 0, 220), 2)

    ratio = cv2.countNonZero(line_mask) / line_mask.size * 100.0
    longest = max(lengths) if lengths else 0.0
    diag = math.hypot(w, h)
    score = 0
    if lengths:
        score = min(100, round(50 * min(1.0, longest / (0.3 * diag)) + 5 * min(len(lengths), 6) + 20 * min(ratio / 2, 1.0)))
    severity = "None" if not lengths else "Low" if score < 25 else "Medium" if score < 55 else "High"
    return {
        "result": DEFECT if lengths else CLEAR,
        "ratio": round(ratio, 3), "count": len(lengths), "longest_px": round(longest, 1),
        "score": score, "severity": severity, "overlay": overlay, "mask": line_mask,
    }


def result_row(name, a, notes):
    return {"timestamp": now_iso(), "image": name, "result": a["result"],
            "highlighted_pixel_ratio_percent": a["ratio"], "crack_count": a["count"],
            "longest_crack_px": a["longest_px"], "severity": a["severity"], "notes": notes}


class LiveCrackDetector(VideoProcessorBase):
    """Live screening with the same analysis as the Image check page."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last = {"result": "Waiting for camera...", "ratio": 0.0, "count": 0, "longest_px": 0.0, "severity": "None", "score": 0}
        self.last_saved_time = 0.0
        self.streak = 0
        self.threshold = 30
        self.min_length = 40

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        if image.shape[1] > 640:  # keep processing fast
            scale = 640 / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        a = analyze_image(image, self.threshold, self.min_length)
        self.streak = self.streak + 1 if a["result"] == DEFECT else 0
        flagged = self.streak >= 3  # needs 3 frames in a row, which avoids one-frame flicker
        out = a["overlay"] if flagged else image
        label = DEFECT if flagged else CLEAR
        color = (0, 0, 220) if flagged else (0, 130, 0)
        cv2.putText(out, f"{label} | cracks: {a['count']} | severity: {a['severity']}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        now = time.time()
        should_save = False
        with self.lock:  # short lock: only update shared values
            self.last = {**{k: a[k] for k in ("ratio", "count", "longest_px", "severity", "score")}, "result": label}
            if flagged and now - self.last_saved_time >= 10:
                self.last_saved_time = now
                should_save = True
        if should_save:  # disk work outside the lock so video never stalls
            name = f"live_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            try:
                if cv2.imwrite(str(IMAGES / name), out):
                    append_row(REPORT, result_row(name, {**a, "result": DEFECT},
                                                  "Automatic webcam flag; classical OpenCV screening, not a confirmed crack."))
            except OSError:
                pass
        return frame.from_ndarray(out, format="bgr24")


# ---------------------------------------------------------------- Process data, limits and statistics
def load_limits():
    try:
        data = json.loads(LIMITS_FILE.read_text(encoding="utf-8"))
        return {p: [float(data[p][0]), float(data[p][1])] for p in PARAMS}
    except Exception:
        return {p: list(v) for p, v in DEFAULT_LIMITS.items()}


def save_limits(limits):
    LIMITS_FILE.write_text(json.dumps(limits, indent=2), encoding="utf-8")


def clean_process(df):
    """Return a tidy, typed copy of the process log (or an empty frame)."""
    if df.empty or any(c not in df.columns for c in PROC_COLS):
        return pd.DataFrame(columns=PROC_COLS)
    d = df.copy()
    for c in PARAMS + ["units_inspected", "defects_found"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=PARAMS + ["units_inspected", "defects_found", "timestamp"])
    d = d[(d["units_inspected"] > 0) & (d["defects_found"] >= 0) & (d["defects_found"] <= d["units_inspected"])]
    d = d.sort_values("timestamp").reset_index(drop=True)
    d["rate"] = d["defects_found"] / d["units_inspected"]
    return d


def breach_severity(value, lo, hi):
    """0 when inside limits, otherwise the distance outside as a share of the allowed range."""
    span = max(hi - lo, 1e-9)
    if value < lo:
        return (lo - value) / span
    if value > hi:
        return (value - hi) / span
    return 0.0


def out_mask(series, lo, hi):
    return (series < lo) | (series > hi)


def wilson(k, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def compare_rates(bn, bd, an, ad):
    """Two-proportion comparison of defect rates (before vs after)."""
    pb, pa = bd / bn, ad / an
    pooled = (bd + ad) / (bn + an)
    se_pool = math.sqrt(pooled * (1 - pooled) * (1 / bn + 1 / an))
    z = (pb - pa) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - norm_cdf(abs(z))) if se_pool > 0 else 1.0
    se = math.sqrt(pb * (1 - pb) / bn + pa * (1 - pa) / an)
    diff = pb - pa
    return {"before": pb * 100, "after": pa * 100, "diff": diff * 100,
            "ci_low": (diff - 1.96 * se) * 100, "ci_high": (diff + 1.96 * se) * 100,
            "p": p_value, "relative": (diff / pb * 100) if pb > 0 else 0.0}


def risk_level(rate_pct):
    return ("Low", "ok") if rate_pct < 3 else ("Medium", "warn") if rate_pct < 8 else ("High", "bad")


# ---------------------------------------------------------------- First administrator setup
BRAND_STEPS = """
<div class="step"><b>1</b><div><strong>Capture</strong><br>Scan parts with your camera or upload images.</div></div>
<div class="step"><b>2</b><div><strong>Review</strong><br>See flagged areas and the reason they were flagged.</div></div>
<div class="step"><b>3</b><div><strong>Act</strong><br>Record operator decisions and check if they worked.</div></div>
"""


def auth_layout(title, subtitle):
    _, mid, _ = st.columns([0.05, 1, 0.05])
    with mid:
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            st.markdown('<div class="brand-panel"><h1>🏭 ManufacturingQualityAI</h1>'
                        '<p>Spot surface defects early and keep a clear record of every inspection.</p>'
                        f'{BRAND_STEPS}</div>', unsafe_allow_html=True)
        with right:
            st.markdown(f'<div class="signin-title">{title}</div><div class="signin-sub">{subtitle}</div>',
                        unsafe_allow_html=True)
    return right


if users().empty:
    setup_key = os.environ.get("MQAI_SETUP_KEY", "")

    if not setup_key:
        st.error("Administrator setup is not configured.")
        st.stop()

    entered_key = st.text_input("Workspace setup key", type="password")

    if not hmac.compare_digest(entered_key, setup_key):
        st.info("Enter the workspace setup key to create the administrator account.")
        st.stop()

    right = auth_layout(
        "Set up your workspace",
        "Create the first administrator account. You can add more users later."
    )

    with right:
        with st.form("first_admin"):
            email = st.text_input("Administrator email", placeholder="name@company.com")
            pw1 = st.text_input("Password", type="password", help="Use at least 12 characters.")
            pw2 = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create administrator", use_container_width=True)

        if submitted:
            if not valid_email(email):
                st.error("Enter a valid email address, like name@company.com.")
            elif len(pw1) < 12:
                st.error("Password must be at least 12 characters.")
            elif pw1 != pw2:
                st.error("The two passwords don't match.")
            else:
                create_user(email, pw1, "admin")
                st.session_state["flash"] = "Administrator created. Sign in below."
                st.rerun()

    st.stop()
# ---------------------------------------------------------------- Sign in
if not st.session_state.get("authenticated"):
    right = auth_layout("Welcome back", "Sign in to continue to your inspections.")
    with right:
        if st.session_state.get("flash"):
            st.success(st.session_state.pop("flash"))
        locked_for = int(st.session_state.get("locked_until", 0) - time.time())
        with st.form("login"):
            email = st.text_input("Email address", placeholder="name@company.com")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign in", use_container_width=True)
        if submit:
            if locked_for > 0:
                st.error(f"Too many attempts. Try again in {locked_for} seconds.")
            else:
                role = authenticate(email, password)
                if role:
                    st.session_state["authenticated"] = True
                    st.session_state["email"] = email.strip().lower()
                    st.session_state["role"] = role
                    st.session_state["failed"] = 0
                    st.rerun()
                st.session_state["failed"] = st.session_state.get("failed", 0) + 1
                if st.session_state["failed"] >= MAX_ATTEMPTS:
                    st.session_state["locked_until"] = time.time() + LOCK_SECONDS
                    st.session_state["failed"] = 0
                    st.error(f"Too many attempts. Wait {LOCK_SECONDS} seconds and try again.")
                else:
                    st.error("Email or password is incorrect. Check both and try again.")
        st.caption("Forgot your password? Ask an administrator to create a new account for you.")
    st.stop()

# ---------------------------------------------------------------- Workspace
saved_images = images()
records = read_csv(REPORT)
process_raw = read_csv(PROCESS)
action_data = read_csv(ACTIONS)
verify_data = read_csv(VERIFY)
limits = load_limits()
is_admin = st.session_state.get("role") == "admin"

PAGES = {
    "🏠  Overview": "Overview",
    "🖼️  Image check": "Image Inspection",
    "🎥  Live camera": "Live Webcam Detection",
    "🧾  Process log": "Process log",
    "🔎  Find the cause": "Explainable Diagnosis",
    "📦  Batch risk": "Batch Risk",
    "✅  Recommendations": "Recommendations",
    "📈  Did it work?": "Closed-loop Verification",
    "📄  Reports": "Reports & Email",
    "🕘  History": "History",
}
if is_admin:
    PAGES["👥  Users"] = "Manage users"

st.sidebar.markdown("## 🏭 MQAI")
st.sidebar.caption("Quality intelligence")
choice = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
page = PAGES[choice]
st.sidebar.divider()
st.sidebar.caption(f"Signed in as\n\n**{st.session_state['email']}** ({st.session_state['role']})")
if st.sidebar.button("Sign out", use_container_width=True):
    st.session_state.clear()
    st.rerun()

hist = clean_process(process_raw)

# ---------------- Overview
if page == "Overview":
    page_head("Overview", "A quick look at your inspection results.")
    total = len(records)
    flags = int((records["result"] == DEFECT).sum()) if total and "result" in records else 0
    a, b, c, d = st.columns(4)
    a.metric("Saved images", len(saved_images))
    b.metric("Inspections recorded", total)
    c.metric("Possible defects", flags)
    d.metric("Flag rate", f"{flags / total * 100:.1f}%" if total else "0.0%")
    st.caption("Flags are preliminary screening results, not confirmed defects.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Latest image")
        if saved_images:
            show_image(str(saved_images[0]), caption=saved_images[0].name)
        else:
            st.info("No images yet. Use Live camera or add images to the inspection_images folder.")
    with right:
        st.subheader("Results")
        if total and "result" in records:
            chart = records["result"].fillna("Unknown").value_counts().rename_axis("Outcome").reset_index(name="Count")
            fig = px.pie(chart, names="Outcome", values="Count", hole=.6, color="Outcome",
                         color_discrete_map={DEFECT: "#B42318", CLEAR: "#0F766E", "Unknown": "#8CA0AC"})
            fig.update_layout(legend=dict(orientation="h", y=-0.1))
            show_chart(fig)
        else:
            st.info("Save an inspection result to see the chart.")
    if total and "timestamp" in records:
        daily = records.assign(day=pd.to_datetime(records["timestamp"], errors="coerce").dt.date)
        daily = daily.dropna(subset=["day"]).groupby(["day", "result"]).size().reset_index(name="Count")
        if not daily.empty:
            st.subheader("Inspections per day")
            fig = px.bar(daily, x="day", y="Count", color="result", barmode="stack",
                         color_discrete_map={DEFECT: "#B42318", CLEAR: "#0F766E"})
            fig.update_layout(xaxis_title=None, legend_title=None)
            show_chart(fig)
    st.subheader("How it works")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="panel"><strong>1. Check</strong><br>Scan a part with the camera or pick a saved image.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="panel"><strong>2. Decide</strong><br>Review the likely causes and record what the operator approves.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="panel"><strong>3. Verify</strong><br>Compare defect rates before and after the change.</div>', unsafe_allow_html=True)

# ---------------- Image inspection
elif page == "Image Inspection":
    page_head("Image check", "Look for crack-like lines in a saved image, an upload, or every saved image at once.")
    c1, c2 = st.columns(2, gap="large")
    threshold = c1.slider("Sensitivity threshold", 5, 100, 30, help="Lower values flag more features. Higher values flag fewer.")
    min_len = c2.slider("Minimum crack length (px)", 10, 300, 40, help="Ignore marks shorter than this.")
    tab_saved, tab_upload, tab_all = st.tabs(["Saved image", "Upload an image", "Scan all saved images"])

    def show_analysis(img, name, key):
        a = analyze_image(img, threshold, min_len)
        x, y = st.columns(2, gap="large")
        with x:
            st.markdown("**Original**")
            show_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with y:
            st.markdown("**Detected crack-like features**")
            show_image(cv2.cvtColor(a["overlay"], cv2.COLOR_BGR2RGB))
        kind = "bad" if a["result"] == DEFECT else "ok"
        st.markdown(f"Result: {pill(a['result'], kind)}", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Crack-like features", a["count"])
        m2.metric("Longest (px)", f"{a['longest_px']:.0f}")
        m3.metric("Highlighted pixels", f"{a['ratio']:.2f}%")
        m4.metric("Severity", f"{a['severity']} ({a['score']})")
        st.warning("This is a classical image-processing screen. It can miss real defects and flag harmless marks.")
        if st.button("Save result", key=key):
            append_row(REPORT, result_row(name, a, "OpenCV screening; not a trained model or confirmed crack diagnosis."))
            st.success("Saved to the inspection report.")

    with tab_saved:
        if not saved_images:
            st.warning("No images found. Capture some with Live camera, upload one, or add files to the inspection_images folder.")
        else:
            selected = st.selectbox("Image", saved_images, format_func=lambda p: p.name)
            img = cv2.imread(str(selected))
            if img is None:
                st.error("This image can't be read. Try a different file.")
            else:
                show_analysis(img, selected.name, "save_saved")

    with tab_upload:
        up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            data = np.frombuffer(up.getvalue(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                st.error("This file isn't a readable image.")
            else:
                keep = st.checkbox("Also keep a copy in the inspection_images folder", value=True)
                name = f"upload_{datetime.now():%Y%m%d_%H%M%S}_{Path(up.name).stem[:30]}.jpg"
                if keep and st.session_state.get("last_upload") != (up.name, up.size):
                    cv2.imwrite(str(IMAGES / name), img)
                    st.session_state["last_upload"] = (up.name, up.size)
                    st.session_state["last_upload_name"] = name
                show_analysis(img, st.session_state.get("last_upload_name", name) if keep else up.name, "save_upload")

    with tab_all:
        if not saved_images:
            st.info("No saved images to scan.")
        elif st.button("Scan all saved images", type="primary"):
            rows, bar = [], st.progress(0.0)
            for i, p in enumerate(saved_images):
                im = cv2.imread(str(p))
                if im is not None:
                    rows.append(result_row(p.name, analyze_image(im, threshold, min_len), "Batch OpenCV scan; not confirmed."))
                bar.progress((i + 1) / len(saved_images))
            st.session_state["scan_all"] = rows
        rows = st.session_state.get("scan_all")
        if rows:
            df = pd.DataFrame(rows)
            flagged = int((df["result"] == DEFECT).sum())
            st.metric("Flagged images", f"{flagged} of {len(df)}")
            show_table(df.drop(columns=["timestamp"]))
            if st.button("Save all results to the report"):
                for r in rows:
                    append_row(REPORT, r)
                st.session_state.pop("scan_all")
                st.success(f"Saved {len(rows)} results.")
                st.rerun()


# ---------------- AI Image Detection
if page == "Image Inspection":
    st.divider()
    st.subheader("AI Surface Defect Detection")
    st.caption("Detect six metal surface defect types using the trained YOLO model.")

    ai_upload = st.file_uploader(
        "Upload an image for AI inspection",
        type=["jpg", "jpeg", "png"],
        key="yolo_upload"
    )

    if ai_upload is not None:
        image_bytes = np.frombuffer(ai_upload.getvalue(), np.uint8)
        ai_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if ai_image is None:
            st.error("Unable to read the uploaded image.")
        elif st.button("Run AI Detection"):
            with st.spinner("Analyzing surface..."):
                prediction = detect_defects(ai_image)
                name = f"yolo_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                saved = cv2.imwrite(str(IMAGES / name), prediction["annotated_image"])

                if saved:
                    detections = prediction["detections"]
                    defect_names = ", ".join(sorted(set(d["defect"] for d in detections)))
                    confidences = ", ".join(str(d["confidence"]) for d in detections)
                    append_row(REPORT, {
                        "timestamp": now_iso(),
                        "image": name,
                        "result": "AI DEFECT DETECTED" if detections else "NO AI DETECTION",
                        "inspection_method": "YOLOv8",
                        "defect_count": prediction["count"],
                        "defect_types": defect_names,
                        "confidence_scores": confidences,
                        "notes": "Automated YOLO surface inspection; predictions require verification."
                    })
                    st.success("AI inspection saved to History!")
                else:
                    st.error("Could not save the inspection image.")

            st.image(
                cv2.cvtColor(prediction["annotated_image"], cv2.COLOR_BGR2RGB),
                caption="YOLO detection result"
            )

            st.metric("Defects detected", prediction["count"])

            if prediction["detections"]:
                st.dataframe(
                    pd.DataFrame(prediction["detections"]),
                    hide_index=True
                )
            else:
                st.warning("No defects detected above the confidence threshold.")
elif page == "Live Webcam Detection":
    page_head("Live camera", "Screen parts in real time. Flagged frames are saved automatically.")
    st.warning("Screening only. Lighting, texture and reflections can cause false alerts or missed defects.")
    c1, c2 = st.columns(2, gap="large")
    sens = c1.slider("Sensitivity threshold", 5, 100, 30, help="Lower values flag more. Raise it if clean surfaces get flagged.")
    live_len = c2.slider("Minimum crack length (px)", 10, 200, 40)
    st.caption("Allow camera access when your browser asks. Use bright, even light and hold the part steady.")
    detection_mode = st.radio(
        "Choose live detection mode",
        ["OpenCV Crack Detection", "YOLO AI Defect Detection"],
        horizontal=True,
        key="live_detection_mode"
    )
    ctx = webrtc_streamer(
        key=f"manufacturing-quality-live-{detection_mode}",
        video_processor_factory=(
    LiveYOLODetector
    if detection_mode == "YOLO AI Defect Detection"
    else LiveCrackDetector
),
        desired_playing_state=True,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    if (
        detection_mode == "OpenCV Crack Detection"
        and ctx.state.playing
        and ctx.video_processor is not None
    ):
        ctx.video_processor.threshold = sens
        ctx.video_processor.min_length = live_len
        with ctx.video_processor.lock:
            live = dict(ctx.video_processor.last)
        a, b, c = st.columns(3)
        a.metric("Current result", live["result"])
        b.metric("Crack-like features", live["count"])
        c.metric("Severity", live["severity"])
        if live["result"] == DEFECT:
            st.error("Possible defect flagged. Confirm it with an approved inspection method.")
            st.caption("While a flag continues, one image is saved every 10 seconds.")
        else:
            st.success("Nothing obvious in this frame. That doesn't prove the part is defect-free.")
        time.sleep(1)
        st.rerun()
    else:
        st.info("Waiting for the camera. If it doesn't start, check your browser's camera permission and press Start.")

# ---------------- Process log
elif page == "Process log":
    page_head("Process log", "Record machine readings and results for each batch. Other pages learn from this data.")
    st.caption("Each batch needs its readings plus how many units were inspected and how many were confirmed defective.")
    with st.form("log_batch", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        batch_id = c1.text_input("Batch ID", placeholder="BATCH-001")
        units = c2.number_input("Units inspected", min_value=1, value=100)
        defects = c3.number_input("Confirmed defects", min_value=0, value=0)
        r1, r2, r3 = st.columns(3)
        vals = {}
        for col, p in zip((r1, r2, r3), PARAMS):
            lo, hi = limits[p]
            vals[p] = col.number_input(f"{LABELS[p][0]} ({LABELS[p][1]})", value=float((lo + hi) / 2))
        saved = st.form_submit_button("Save batch")
    if saved:
        if not batch_id.strip():
            st.error("Enter a batch ID.")
        elif defects > units:
            st.error("Defects can't be higher than the units inspected.")
        else:
            append_row(PROCESS, {"timestamp": now_iso(), "batch_id": batch_id.strip(), **vals,
                                 "units_inspected": int(units), "defects_found": int(defects)})
            st.success(f"Saved {batch_id.strip()}.")
            st.rerun()

    with st.expander("Import batches from a CSV file"):
        template = pd.DataFrame([{"timestamp": "2026-01-15T09:30:00", "batch_id": "BATCH-001", "temperature": 182,
                                  "pressure": 5.1, "speed": 100, "units_inspected": 100, "defects_found": 4}])
        st.download_button("Download a template", template.to_csv(index=False).encode("utf-8"), "process_template.csv", "text/csv")
        up = st.file_uploader("CSV file", type=["csv"], key="proc_upload")
        if up is not None:
            try:
                new = pd.read_csv(up)
            except Exception:
                new = pd.DataFrame()
            if "timestamp" not in new.columns and not new.empty:
                new["timestamp"] = now_iso()
            missing = [c for c in PROC_COLS if c not in new.columns]
            if new.empty or missing:
                st.error("The file needs these columns: " + ", ".join(PROC_COLS) + (f". Missing: {', '.join(missing)}." if missing else "."))
            else:
                good = clean_process(new)
                st.write(f"{len(good)} of {len(new)} rows are valid.")
                if len(good) and st.button("Import valid rows"):
                    for _, r in good.iterrows():
                        append_row(PROCESS, {"timestamp": r["timestamp"].isoformat(timespec="seconds"), "batch_id": r["batch_id"],
                                             **{p: r[p] for p in PARAMS},
                                             "units_inspected": int(r["units_inspected"]), "defects_found": int(r["defects_found"])})
                    st.success(f"Imported {len(good)} rows.")
                    st.rerun()

    if hist.empty:
        st.info("No valid batches yet. Add one above to unlock Find the cause, Batch risk and Did it work?")
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("Batches", len(hist))
        t2.metric("Units inspected", f"{int(hist['units_inspected'].sum()):,}")
        t3.metric("Overall defect rate", f"{hist['defects_found'].sum() / hist['units_inspected'].sum() * 100:.2f}%")
        fig = px.line(hist, x="timestamp", y=hist["rate"] * 100, markers=True, labels={"y": "Defect rate (%)", "timestamp": ""})
        fig.update_traces(line_color="#0F766E")
        fig.update_layout(yaxis_title="Defect rate (%)")
        show_chart(fig)
        show_table(hist.drop(columns=["rate"]).sort_values("timestamp", ascending=False))

# ---------------- Explainable diagnosis
elif page == "Explainable Diagnosis":
    page_head("Find the cause", "Enter current readings. See which ones are out of range and how they line up with past defects.")

    with st.expander("Allowed ranges"):
        st.caption("These limits drive every page. Set them to your approved process limits.")
        if is_admin:
            with st.form("limits_form"):
                new_limits = {}
                for p in PARAMS:
                    c1, c2 = st.columns(2)
                    lo = c1.number_input(f"{LABELS[p][0]} minimum ({LABELS[p][1]})", value=limits[p][0], key=f"lo_{p}")
                    hi = c2.number_input(f"{LABELS[p][0]} maximum ({LABELS[p][1]})", value=limits[p][1], key=f"hi_{p}")
                    new_limits[p] = [lo, hi]
                if st.form_submit_button("Save ranges"):
                    if any(v[0] >= v[1] for v in new_limits.values()):
                        st.error("Each minimum must be lower than its maximum.")
                    else:
                        save_limits(new_limits)
                        st.success("Ranges saved.")
                        st.rerun()
        else:
            show_table(pd.DataFrame([{"Parameter": f"{LABELS[p][0]} ({LABELS[p][1]})", "Minimum": limits[p][0], "Maximum": limits[p][1]} for p in PARAMS]))
            st.caption("Only administrators can change the ranges.")

    cols = st.columns(3)
    readings = {}
    default_row = hist.iloc[-1] if not hist.empty else None
    for col, p in zip(cols, PARAMS):
        start = float(default_row[p]) if default_row is not None else float(sum(limits[p]) / 2)
        readings[p] = col.number_input(f"{LABELS[p][0]} ({LABELS[p][1]})", value=start, key=f"read_{p}")
    if default_row is not None:
        st.caption("Readings start from the latest logged batch.")

    if st.button("Find likely causes", type="primary"):
        rows = []
        enough = len(hist) >= 5
        for p in PARAMS:
            lo, hi = limits[p]
            sev = breach_severity(readings[p], lo, hi)
            points, why = 0, []
            if sev > 0:
                points += 3 if sev > 0.1 else 2
                side = "below" if readings[p] < lo else "above"
                why.append(f"{side} the allowed range by {sev * 100:.0f}% of its width")
            corr = z = float("nan")
            rate_in = rate_out = float("nan")
            n_out = 0
            if enough:
                col = hist[p]
                if col.nunique() > 1 and hist["rate"].nunique() > 1:
                    corr = float(col.corr(hist["rate"]))
                if col.std() and col.std() > 0:
                    z = float((readings[p] - col.mean()) / col.std())
                om = out_mask(col, lo, hi)
                n_out = int(om.sum())
                if om.any():
                    rate_out = hist.loc[om, "defects_found"].sum() / hist.loc[om, "units_inspected"].sum() * 100
                if (~om).any():
                    rate_in = hist.loc[~om, "defects_found"].sum() / hist.loc[~om, "units_inspected"].sum() * 100
                if not math.isnan(corr) and abs(corr) >= 0.3:
                    points += 2 if abs(corr) >= 0.5 else 1
                    why.append(f"{'higher' if corr > 0 else 'lower'} values went with more defects (r = {corr:+.2f})")
                if not math.isnan(z) and abs(z) >= 2:
                    points += 1
                    why.append(f"unusual compared with past batches (z = {z:+.1f})")
                if not math.isnan(rate_out) and not math.isnan(rate_in) and rate_out > rate_in * 1.5 and n_out >= 2:
                    points += 1
                    why.append(f"out-of-range batches had {rate_out:.1f}% defects vs {rate_in:.1f}% in range")
            priority = "High" if points >= 4 else "Medium" if points >= 2 else "Low"
            check = {"temperature": "Verify the temperature sensor and approved setpoint",
                     "pressure": "Check the pressure sensor and supply",
                     "speed": "Compare machine speed with the process specification"}[p]
            rows.append({"Parameter": LABELS[p][0], "Reading": readings[p], "Allowed": f"{lo:g} to {hi:g}",
                         "Priority": priority, "Why": "; ".join(why) if why else "Inside the range and nothing unusual in the history",
                         "Suggested check": check if points >= 2 else "No action needed", "_points": points,
                         "Correlation (r)": None if math.isnan(corr) else round(corr, 2)})
        result = pd.DataFrame(rows).sort_values("_points", ascending=False)
        st.session_state["diagnosis"] = (result, enough)

    if "diagnosis" in st.session_state:
        result, enough = st.session_state["diagnosis"]
        if (result["Priority"] == "Low").all():
            st.success("Nothing stands out. All readings are in range and look normal.")
        top = result.iloc[0]
        if top["Priority"] != "Low":
            st.markdown(f"Most likely contributor: **{top['Parameter']}** {pill(top['Priority'] + ' priority', 'bad' if top['Priority'] == 'High' else 'warn')}",
                        unsafe_allow_html=True)
        show_table(result.drop(columns=["_points"]))
        if enough and result["Correlation (r)"].notna().any():
            cdf = result.dropna(subset=["Correlation (r)"])
            fig = px.bar(cdf, x="Parameter", y="Correlation (r)", range_y=[-1, 1], title="How each reading moves with defect rate in past batches")
            fig.update_traces(marker_color="#0F766E")
            show_chart(fig)
        if not enough:
            st.info("Log at least 5 batches in Process log to add history-based evidence (correlation, unusual readings, past defect rates).")
        st.caption("Correlation shows what moved together, not what caused the defect. Confirm with a physical check.")

# ---------------- Batch risk
elif page == "Batch Risk":
    page_head("Batch risk", "Estimate the defect rate of an upcoming batch from past batches with similar conditions.")
    st.caption("This uses your batch history and statistics. It is not a trained ML model.")
    c1, c2 = st.columns(2)
    batch = c1.text_input("Batch ID", "BATCH-001")
    size = c2.number_input("Units in batch", min_value=1, value=100)
    cols = st.columns(3)
    planned = {}
    for col, p in zip(cols, PARAMS):
        planned[p] = col.number_input(f"Planned {LABELS[p][0].lower()} ({LABELS[p][1]})", value=float(sum(limits[p]) / 2), key=f"plan_{p}")
    out_now = [LABELS[p][0] for p in PARAMS if breach_severity(planned[p], *limits[p]) > 0]
    n_out_now = len(out_now)

    if len(hist) >= 5:
        h = hist.copy()
        h["breaches"] = sum(out_mask(h[p], *limits[p]).astype(int) for p in PARAMS)
        h["bucket"] = h["breaches"].clip(upper=2)
        bucket = min(n_out_now, 2)
        overall_rate = h["defects_found"].sum() / h["units_inspected"].sum()
        grp = h[h["bucket"] == bucket]
        k, n = int(grp["defects_found"].sum()), int(grp["units_inspected"].sum())
        prior = 20  # pseudo-units pulling thin history toward the overall rate
        est = (k + prior * overall_rate) / (n + prior)
        lo_ci, hi_ci = wilson(round(est * (n + prior)), n + prior)
        label, kind = risk_level(est * 100)
        a, b, c, d = st.columns(4)
        a.metric("Estimated defect rate", f"{est * 100:.1f}%")
        b.metric("Likely range", f"{lo_ci * 100:.1f}% to {hi_ci * 100:.1f}%")
        c.metric("Expected flagged units", f"{size * est:.1f}")
        d.metric("Risk level", label)
        if n_out_now:
            st.markdown(f"{pill(f'{n_out_now} reading(s) out of range', 'warn')} {', '.join(out_now)}", unsafe_allow_html=True)
        else:
            st.markdown(pill("All planned readings in range", "ok"), unsafe_allow_html=True)
        basis = {0: "batches run fully in range", 1: "batches with one reading out of range", 2: "batches with two or more readings out of range"}[bucket]
        st.caption(f"Based on {len(grp)} past {basis} ({n:,} units). Thin history is blended with the overall rate of {overall_rate * 100:.1f}%.")
        summary = h.groupby("bucket").agg(batches=("batch_id", "count"), units=("units_inspected", "sum"), defects=("defects_found", "sum")).reset_index()
        summary["Defect rate (%)"] = (summary["defects"] / summary["units"] * 100).round(2)
        summary["Conditions"] = summary["bucket"].map({0: "All in range", 1: "One out of range", 2: "Two or more out of range"})
        fig = px.bar(summary, x="Conditions", y="Defect rate (%)", text="Defect rate (%)")
        fig.update_traces(marker_color="#0F766E")
        show_chart(fig)
    else:
        st.info("Log at least 5 batches in Process log to get a history-based estimate. Until then, set a risk by hand.")
        probability = st.slider("Assumed risk (%)", 0.0, 100.0, 5.0)
        a, b, c = st.columns(3)
        a.metric("Batch", batch)
        b.metric("Assumed risk", f"{probability:.1f}%")
        c.metric("Expected flagged units", f"{size * probability / 100:.1f}")

# ---------------- Recommendations
elif page == "Recommendations":
    page_head("Recommendations", "Record which checks the operator approved. No machine settings are changed.")
    suggestions = []
    if "diagnosis" in st.session_state:
        res, _ = st.session_state["diagnosis"]
        suggestions = [f"{r['Suggested check']}" for _, r in res.iterrows() if r["Priority"] != "Low"]
    if suggestions:
        st.info("From the latest cause check: " + "; ".join(suggestions))
    options = ["Verify temperature sensor and setpoint", "Inspect pressure sensor and supply",
               "Check machine speed", "Inspect tool condition and alignment",
               "Check camera lighting and focus", "Perform confirmatory manual inspection"]
    c1, c2 = st.columns(2, gap="large")
    recommendation = c1.selectbox("Recommended check", options)
    batch = c2.text_input("Batch ID", key="action_batch")
    decision = c1.selectbox("Operator decision", ["Pending review", "Approved", "Rejected"])
    notes = c2.text_area("Notes", height=100)
    if st.button("Save decision", type="primary"):
        append_row(ACTIONS, {"timestamp": now_iso(), "batch_id": batch, "operator_email": st.session_state["email"],
                             "recommendation": recommendation, "decision": decision, "notes": notes})
        st.success("Decision saved. Approved actions can be verified on the Did it work? page.")
        action_data = read_csv(ACTIONS)
    if not action_data.empty:
        st.subheader("Past decisions")
        show_table(action_data.sort_values("timestamp", ascending=False))

# ---------------- Closed-loop verification
elif page == "Closed-loop Verification":
    page_head("Did it work?", "Compare the defect rate before and after an approved action, with a significance check.")
    mode = st.radio("Data source", ["From logged batches", "Enter numbers manually"], horizontal=True)
    bn = bd = an = ad = None
    label = "Manual entry"

    if mode == "From logged batches":
        approved = action_data[action_data["decision"] == "Approved"].copy() if not action_data.empty and "decision" in action_data else pd.DataFrame()
        if approved.empty:
            st.info("No approved actions yet. Approve one on the Recommendations page, or enter numbers manually.")
        elif hist.empty:
            st.info("No batches logged yet. Add batches in Process log, or enter numbers manually.")
        else:
            approved["when"] = pd.to_datetime(approved["timestamp"], errors="coerce")
            approved = approved.dropna(subset=["when"]).sort_values("when", ascending=False)
            idx = st.selectbox("Approved action", approved.index,
                               format_func=lambda i: f"{approved.loc[i, 'when']:%d %b %Y %H:%M} — {approved.loc[i, 'recommendation']}")
            n_side = st.slider("Batches to compare on each side", 3, 20, 5)
            t0 = approved.loc[idx, "when"]
            before_df = hist[hist["timestamp"] < t0].tail(n_side)
            after_df = hist[hist["timestamp"] >= t0].head(n_side)
            if before_df.empty or after_df.empty:
                st.warning(f"Found {len(before_df)} batches before and {len(after_df)} after this action. You need at least one on each side.")
            else:
                bn, bd = int(before_df["units_inspected"].sum()), int(before_df["defects_found"].sum())
                an, ad = int(after_df["units_inspected"].sum()), int(after_df["defects_found"].sum())
                label = str(approved.loc[idx, "recommendation"])
                st.caption(f"Before: {len(before_df)} batches, {bn:,} units. After: {len(after_df)} batches, {an:,} units.")
    else:
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("#### Before the change")
            bn = st.number_input("Units inspected", min_value=1, value=100, key="bn")
            bd = st.number_input("Confirmed defects", min_value=0, value=8, key="bd")
        with b:
            st.markdown("#### After the change")
            an = st.number_input("Units inspected", min_value=1, value=100, key="an")
            ad = st.number_input("Confirmed defects", min_value=0, value=4, key="ad")

    if bn is not None:
        if bd > bn or ad > an:
            st.error("Defects can't be higher than the units inspected. Check the numbers.")
        elif st.button("Compare", type="primary"):
            st.session_state["verify_result"] = (compare_rates(bn, bd, an, ad), bn, bd, an, ad, label)

    if "verify_result" in st.session_state and bn is not None:
        r, bn_, bd_, an_, ad_, lab = st.session_state["verify_result"]
        x, y, z = st.columns(3)
        x.metric("Before", f"{r['before']:.2f}%")
        y.metric("After", f"{r['after']:.2f}%")
        z.metric("Change (points)", f"{-r['diff']:+.2f}", delta_color="inverse")
        if r["p"] < 0.05 and r["diff"] > 0:
            verdict, kind = "Improved. The drop is statistically supported.", "ok"
        elif r["p"] < 0.05 and r["diff"] < 0:
            verdict, kind = "Worse. The increase is statistically supported.", "bad"
        else:
            verdict, kind = "No clear change yet. The difference could be chance, so log more batches.", "warn"
        st.markdown(pill(verdict, kind), unsafe_allow_html=True)
        direction = "fell" if r["diff"] >= 0 else "rose"
        st.write(f"Defect rate {direction} by **{abs(r['diff']):.2f} points** ({abs(r['relative']):.0f}% relative). "
                 f"95% range for the drop: {r['ci_low']:.2f} to {r['ci_high']:.2f} points, p = {r['p']:.3f}.")
        fig = px.bar(pd.DataFrame({"Period": ["Before", "After"], "Defect rate (%)": [r["before"], r["after"]]}),
                     x="Period", y="Defect rate (%)", text_auto=".2f", color="Period",
                     color_discrete_map={"Before": "#8CA0AC", "After": "#0F766E"})
        fig.update_layout(showlegend=False)
        show_chart(fig)
        st.caption("A before/after comparison alone does not prove the action caused the difference. Other conditions may have changed.")
        if st.button("Save verification"):
            append_row(VERIFY, {"timestamp": now_iso(), "operator_email": st.session_state["email"], "action": lab,
                                "units_before": bn_, "defects_before": bd_, "units_after": an_, "defects_after": ad_,
                                "rate_before_percent": round(r["before"], 3), "rate_after_percent": round(r["after"], 3),
                                "p_value": round(r["p"], 4), "verdict": verdict})
            st.success("Verification saved.")
            verify_data = read_csv(VERIFY)

    if not verify_data.empty:
        st.subheader("Saved verifications")
        show_table(verify_data.sort_values("timestamp", ascending=False))

# ---------------- Reports & email
elif page == "Reports & Email":
    page_head("Reports", "Download results or email them to someone authorized.")
    datasets = {"Inspection results": REPORT, "Process log": PROCESS, "Operator decisions": ACTIONS, "Verification results": VERIFY}
    which = st.selectbox("Report", list(datasets))
    df = read_csv(datasets[which])
    if which == "Inspection results" and df.empty:
        df = pd.DataFrame([{"image": p.name, "result": "NOT ANALYZED", "notes": "Image saved; no inspection result recorded."}
                           for p in reversed(saved_images)])
    if df.empty:
        st.info("Nothing in this report yet.")
    else:
        show_table(df)
    data = df.to_csv(index=False).encode("utf-8-sig")
    slug = which.lower().replace(" ", "_")
    filename = f"mqai_{slug}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    st.download_button("Download report (CSV)", data, filename, "text/csv", type="primary", disabled=df.empty)
    st.divider()
    st.subheader("Email this report")
    if not is_admin:
        st.info("Only administrators can email reports. You can still download the CSV above.")
    else:
        if smtp_configured():
            st.success("Email is set up.")
        else:
            st.warning("Email isn't set up yet. Set MQAI_SMTP_USER and MQAI_SMTP_PASSWORD in the terminal, then restart the app. "
                       "For Gmail, use an app password. Optional: MQAI_SMTP_HOST and MQAI_SMTP_PORT.")
        recipient = st.text_input("Recipient email", placeholder="name@company.com")
        authorized = st.checkbox("This person is authorized to receive the report.")
        if st.button("Send email", type="primary", disabled=df.empty):
            if not valid_email(recipient):
                st.error("Enter a valid recipient email.")
            elif not authorized:
                st.error("Tick the box to confirm the recipient is authorized.")
            else:
                try:
                    send_report(recipient.strip(), data, filename, f"ManufacturingQualityAI: {which.lower()}")
                    st.success(f"Report sent to {recipient.strip()}.")
                except Exception as exc:
                    st.error(f"Email not sent: {exc}")

# ---------------- History
elif page == "History":
    page_head("History", "Every saved inspection and image.")
    df = read_csv(REPORT)
    if not df.empty:
        if "result" in df:
            pick = st.selectbox("Show", ["All results", DEFECT, CLEAR])
            if pick != "All results":
                df = df[df["result"] == pick]
        show_table(df.sort_values("timestamp", ascending=False) if "timestamp" in df else df)
        st.download_button("Download history (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "inspection_history.csv", "text/csv")
    else:
        st.info("No inspections saved yet.")
    shown = {p.name for p in saved_images}
    names = set(df["image"].astype(str)) & shown if not df.empty and "image" in df else shown
    for p in saved_images:
        if p.name in names or df.empty:
            with st.expander(p.name):
                show_image(str(p))

# ---------------- Manage users (admin only)
elif page == "Manage users" and is_admin:
    page_head("Users", "Add people who can sign in to this workspace.")
    udf = users()
    if not udf.empty:
        show_table(udf[[c for c in ["email", "role", "created_at"] if c in udf.columns]])
    st.subheader("Add a user")
    with st.form("add_user", clear_on_submit=True):
        c1, c2 = st.columns(2)
        new_email = c1.text_input("Email", placeholder="name@company.com")
        new_role = c2.selectbox("Role", ["user", "admin"])
        new_pw = st.text_input("Temporary password", type="password", help="At least 12 characters.")
        add = st.form_submit_button("Add user")
    if add:
        existing = set(udf["email"].astype(str).str.lower()) if not udf.empty else set()
        if not valid_email(new_email):
            st.error("Enter a valid email address.")
        elif new_email.strip().lower() in existing:
            st.error("A user with this email already exists.")
        elif len(new_pw) < 12:
            st.error("Password must be at least 12 characters.")
        else:
            create_user(new_email, new_pw, new_role)
            st.success(f"Added {new_email.strip().lower()} as {new_role}.")
