import streamlit as st
import numpy as np
import math
import json
import datetime
from collections import Counter

# ─── MEDIAPIPE / CV2 SAFE IMPORT ────────────────────────────────────────────────
try:
    import cv2
    import mediapipe as mp
    import av

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, RTCConfiguration

    MEDIAPIPE_OK = True
except Exception as e:
    MEDIAPIPE_OK = False
    MEDIAPIPE_ERROR = str(e)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DUNK LAB",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --orange: #FF4B2B;
    --orange-light: #FF6B4B;
    --black: #080808;
    --dark: #0F0F0F;
    --card: #161616;
    --card2: #1C1C1C;
    --border: #252525;
    --white: #F0EBE0;
    --muted: #555555;
    --green: #00FF87;
    --red: #FF3B3B;
    --gold: #FFD700;
    --blue: #00BFFF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--black);
    color: var(--white);
}

.stApp {
    background-color: var(--black);
    background-image:
        radial-gradient(ellipse at 80% 0%, rgba(255,75,43,0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 0% 100%, rgba(255,75,43,0.05) 0%, transparent 50%);
}

[data-testid="stSidebar"] {
    background: var(--dark);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Bebas Neue', sans-serif;
    color: var(--orange);
    letter-spacing: 2px;
}

/* TITLE */
.dunk-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5.5rem;
    line-height: 0.88;
    letter-spacing: 5px;
    color: var(--white);
    text-shadow: 5px 5px 0px var(--orange);
    margin: 0;
}
.dunk-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 5px;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* METRICS */
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3rem !important;
    color: var(--orange) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted) !important;
    font-size: 0.65rem !important;
}

/* CARDS */
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--orange);
}
.stat-card h4 {
    font-family: 'Bebas Neue', sans-serif;
    color: var(--orange);
    letter-spacing: 2px;
    font-size: 1rem;
    margin: 0 0 0.3rem 0;
}
.stat-card p { color: var(--white); font-size: 0.9rem; margin: 0; }

/* RANK CARD */
.rank-card {
    background: linear-gradient(135deg, var(--card) 0%, #1a1008 100%);
    border: 1px solid #3A2A10;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    text-align: center;
}
.rank-card .rank-icon {
    font-size: 2.5rem;
    display: block;
    margin-bottom: 0.3rem;
}
.rank-card .rank-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 3px;
    color: var(--gold);
    display: block;
}
.rank-card .rank-range {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* WORKOUT CARDS */
.workout-card {
    background: var(--card);
    border-left: 3px solid var(--orange);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.4rem;
    margin-bottom: 0.6rem;
    transition: background 0.15s;
}
.workout-card:hover { background: var(--card2); }
.workout-card .exercise-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 1px;
    color: var(--white);
}
.workout-card .exercise-detail {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* HISTORY CARDS */
.history-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: border-color 0.2s;
}
.history-card:hover { border-color: var(--orange); }
.history-card .hc-left { flex: 1; }
.history-card .hc-jump {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--orange);
    line-height: 1;
}
.history-card .hc-meta {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 1px;
    margin-top: 0.2rem;
}
.history-card .hc-rank {
    font-size: 1.8rem;
}

/* CORRECTION BOX */
.correction-box {
    background: linear-gradient(135deg, rgba(255,75,43,0.1), rgba(255,75,43,0.03));
    border: 1px solid rgba(255,75,43,0.4);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-top: 0.6rem;
}
.correction-box .corr-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 3px;
    color: var(--orange);
    margin-bottom: 0.4rem;
    display: block;
}
.correction-box .corr-text {
    font-size: 0.9rem;
    color: var(--white);
    line-height: 1.5;
}

/* BADGES */
.badge {
    display: inline-block;
    background: var(--orange);
    color: var(--black);
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
}
.badge-green { background: var(--green); color: var(--black); }
.badge-red { background: var(--red); color: var(--white); }
.badge-gold { background: var(--gold); color: var(--black); }

/* FORM ALERTS */
.form-error {
    background: rgba(255,59,59,0.1);
    border: 1px solid rgba(255,59,59,0.4);
    border-radius: 6px;
    padding: 0.55rem 1rem;
    color: var(--red);
    font-size: 0.83rem;
    margin-top: 0.35rem;
}
.form-ok {
    background: rgba(0,255,135,0.08);
    border: 1px solid rgba(0,255,135,0.3);
    border-radius: 6px;
    padding: 0.55rem 1rem;
    color: var(--green);
    font-size: 0.83rem;
    margin-top: 0.35rem;
}

/* WELCOME BANNER */
.welcome-banner {
    background: linear-gradient(90deg, rgba(255,75,43,0.15), transparent);
    border: 1px solid rgba(255,75,43,0.25);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}
.welcome-banner strong { color: var(--orange); }

/* DIVIDER */
.orange-line {
    height: 2px;
    background: linear-gradient(90deg, var(--orange), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
    font-size: 1rem;
    color: var(--muted);
    border: none;
    background: transparent;
    padding: 0.6rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    color: var(--orange) !important;
    border-bottom: 2px solid var(--orange) !important;
    background: transparent !important;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #FF4B2B, #FF6B4B);
    color: var(--black);
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 3px;
    font-size: 1rem;
    border: none;
    border-radius: 6px;
    padding: 0.65rem 2rem;
    transition: all 0.15s;
    box-shadow: 0 4px 15px rgba(255,75,43,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #FF6B4B, #FF4B2B);
    color: var(--black);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255,75,43,0.4);
}

/* INPUTS */
.stNumberInput input, .stTextInput input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
    border-radius: 6px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--orange) !important;
}
.stSelectbox > div, .stMultiSelect > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
}

/* PROGRESS BAR */
.progress-bar-container {
    background: var(--card);
    border-radius: 8px;
    height: 8px;
    margin: 0.5rem 0 0.3rem;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, var(--orange), var(--gold));
    transition: width 0.5s ease;
}

/* SESSION HEADER */
.session-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 3px;
    color: var(--muted);
    margin: 1rem 0 0.4rem;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─── RANK SYSTEM ────────────────────────────────────────────────────────────────

RANKS = [
    {"name": "Benchwarmer", "icon": "🧊", "min": 0,   "max": 40,  "color": "#4FC3F7"},
    {"name": "Sixth Man",   "icon": "🏀", "min": 40,  "max": 60,  "color": "#81C784"},
    {"name": "Starter",     "icon": "🔥", "min": 60,  "max": 80,  "color": "#FFB74D"},
    {"name": "All-Star",    "icon": "⭐", "min": 80,  "max": 100, "color": "#CE93D8"},
    {"name": "Dunk Master", "icon": "👑", "min": 100, "max": 999, "color": "#FFD700"},
]

REAL_TIME_CORRECTIONS = {
    "Left knee caving in": "Drive your left knee OUT — push it toward your pinky toe on every landing.",
    "Right knee caving in": "Focus on right hip abductor activation. Squeeze glutes on takeoff.",
    "Squat too shallow — go deeper": "Hit parallel! Sit INTO the squat — your crease should break knee level.",
    "Keep your back straight": "Brace your core like you're about to take a punch. Keep that chest up!",
    "Knee angle too acute in lunge": "Don't let the knee collapse forward — shin stays vertical, push the hip down.",
}

GENERAL_CORRECTIONS = [
    "Increase your penultimate step speed — it generates elastic energy for the jump.",
    "Deepen your squat depth on the approach — tap more power from your hips.",
    "Drive your arms UP explosively — arm swing adds 5–10 cm to your jump.",
    "Land softly with bent knees — protect joints and reset faster.",
    "Keep your gaze forward during takeoff — chin up helps full hip extension.",
]


def get_rank(jump_cm: float) -> dict:
    for r in RANKS:
        if r["min"] <= jump_cm < r["max"]:
            return r
    return RANKS[-1]


def get_rank_progress(jump_cm: float) -> float:
    rank = get_rank(jump_cm)
    if rank["max"] == 999:
        return 1.0
    span = rank["max"] - rank["min"]
    pos = jump_cm - rank["min"]
    return min(pos / span, 1.0)


def get_realtime_correction(form_errors: list) -> str:
    for err in form_errors:
        if err in REAL_TIME_CORRECTIONS:
            return REAL_TIME_CORRECTIONS[err]
    import random
    return random.choice(GENERAL_CORRECTIONS)


# ─── SESSION STATE INIT ──────────────────────────────────────────────────────────
defaults = {
    "clean_reps": 0,
    "total_reps": 0,
    "max_jump_cm": 0.0,
    "form_errors": [],
    "last_exercise": "jump",
    "workout_plan": None,
    "prev_clean_reps": 0,
    "username": "",
    "jump_history": [],       # list of {cm, ts, rank, exercise}
    "all_time_history": {},   # {username: [{cm, ts, rank}]}  — in-memory DB
    "height_cm": 180,
    "weight_kg": 80,
    "inventory": ["No Equipment"],
    "realtime_correction": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))


def record_jump(cm: float, exercise: str):
    rank = get_rank(cm)
    entry = {
        "cm": round(cm, 1),
        "ts": datetime.datetime.now().strftime("%H:%M:%S"),
        "rank": rank["name"],
        "rank_icon": rank["icon"],
        "exercise": exercise,
    }
    st.session_state["jump_history"].append(entry)

    # update in-memory "DB"
    username = st.session_state.get("username", "").strip() or "anonymous"
    db = st.session_state["all_time_history"]
    if username not in db:
        db[username] = []
    db[username].append(entry)


def generate_workout_plan(client, height_cm, weight_kg, inventory, form_errors):
    error_summary = ""
    if form_errors:
        unique_errors = list(set(form_errors[-20:]))
        error_summary = f"\n\nForm issues detected: {', '.join(unique_errors)}."

    inventory_str = ", ".join(inventory) if inventory else "No equipment"

    prompt = f"""You are an elite basketball strength & conditioning coach specializing in vertical jump development.

Athlete profile:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Available equipment: {inventory_str}
- Session form notes: {error_summary if error_summary else 'No major errors detected.'}

Create a highly personalized 4-week vertical jump program. Return ONLY valid JSON in this exact structure:
{{
  "program_name": "...",
  "goal": "...",
  "weeks": [
    {{
      "week": 1,
      "focus": "...",
      "days": [
        {{
          "day": "Monday",
          "exercises": [
            {{
              "name": "...",
              "sets": 3,
              "reps": "10",
              "rest_sec": 60,
              "cue": "...",
              "equipment": "..."
            }}
          ]
        }}
      ]
    }}
  ],
  "tips": ["...", "..."]
}}

Tailor exercise selection strictly to available equipment. Address detected form errors with corrective drills."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2500,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─── VIDEO TRANSFORMER ────────────────────────────────────────────────────────────

if MEDIAPIPE_OK:
    class JumpAnalyzer(VideoTransformerBase):
        def __init__(self):
            self.pose = mp_pose.Pose(
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                model_complexity=1,
            )
            self.rest_hip_y = None
            self.calibrated = False
            self.calibration_frames = 0
            self.hip_y_buffer = []
            self.rep_phase = "down"
            self._last_recorded_cm = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            overlay = img.copy()

            form_errors_this_frame = []
            jump_cm = 0.0
            left_knee_angle = None
            right_knee_angle = None

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                mp_drawing.draw_landmarks(
                    overlay,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 75, 43), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(60, 60, 60), thickness=1
                    ),
                )

                def lp(idx):
                    l = lm[idx]
                    return (int(l.x * w), int(l.y * h)), l.y

                (lhip_px, lhip_y_n) = lp(mp_pose.PoseLandmark.LEFT_HIP)
                (rhip_px, rhip_y_n) = lp(mp_pose.PoseLandmark.RIGHT_HIP)
                (lknee_px, _) = lp(mp_pose.PoseLandmark.LEFT_KNEE)
                (rknee_px, _) = lp(mp_pose.PoseLandmark.RIGHT_KNEE)
                (lank_px, _) = lp(mp_pose.PoseLandmark.LEFT_ANKLE)
                (rank_px, _) = lp(mp_pose.PoseLandmark.RIGHT_ANKLE)
                (lsho_px, _) = lp(mp_pose.PoseLandmark.LEFT_SHOULDER)
                (rsho_px, _) = lp(mp_pose.PoseLandmark.RIGHT_SHOULDER)

                mid_hip_y_norm = (lhip_y_n + rhip_y_n) / 2

                if not self.calibrated:
                    self.hip_y_buffer.append(mid_hip_y_norm)
                    self.calibration_frames += 1
                    if self.calibration_frames >= 30:
                        self.rest_hip_y = np.mean(self.hip_y_buffer)
                        self.calibrated = True
                    progress = int((self.calibration_frames / 30) * 100)
                    cv2.putText(overlay, f"CALIBRATING... {progress}%",
                                (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 200, 0), 2)
                else:
                    height_cm_val = st.session_state.get("height_cm", 180)
                    delta_norm = self.rest_hip_y - mid_hip_y_norm
                    cm_per_norm = height_cm_val / 0.85
                    jump_cm = max(0.0, delta_norm * cm_per_norm)

                    if jump_cm > st.session_state["max_jump_cm"]:
                        st.session_state["max_jump_cm"] = round(jump_cm, 1)

                    left_knee_angle = calculate_angle(lhip_px, lknee_px, lank_px)
                    right_knee_angle = calculate_angle(rhip_px, rknee_px, rank_px)

                    exercise = st.session_state.get("last_exercise", "jump")

                    if exercise in ("squat", "jump"):
                        if lknee_px[0] < lank_px[0] - 30:
                            form_errors_this_frame.append("Left knee caving in")
                        if rknee_px[0] > rank_px[0] + 30:
                            form_errors_this_frame.append("Right knee caving in")
                        if exercise == "squat":
                            if left_knee_angle > 110 or right_knee_angle > 110:
                                form_errors_this_frame.append("Squat too shallow — go deeper")

                    if exercise == "lunge":
                        if left_knee_angle < 70 or right_knee_angle < 70:
                            form_errors_this_frame.append("Knee angle too acute in lunge")

                    sho_mid_x = (lsho_px[0] + rsho_px[0]) / 2
                    hip_mid_x = (lhip_px[0] + rhip_px[0]) / 2
                    if abs(sho_mid_x - hip_mid_x) > w * 0.12:
                        form_errors_this_frame.append("Keep your back straight")

                    if exercise == "jump":
                        if jump_cm > 5 and self.rep_phase == "down":
                            self.rep_phase = "up"
                        elif jump_cm < 2 and self.rep_phase == "up":
                            self.rep_phase = "down"
                            st.session_state["total_reps"] += 1
                            if not form_errors_this_frame:
                                st.session_state["clean_reps"] += 1
                            # record jump
                            record_jump(self._last_recorded_cm or jump_cm, exercise)
                            if form_errors_this_frame:
                                st.session_state["realtime_correction"] = get_realtime_correction(form_errors_this_frame)

                    if exercise == "squat":
                        avg_knee = (left_knee_angle + right_knee_angle) / 2
                        if avg_knee < 95 and self.rep_phase == "down":
                            self.rep_phase = "up"
                        elif avg_knee > 160 and self.rep_phase == "up":
                            self.rep_phase = "down"
                            st.session_state["total_reps"] += 1
                            if not form_errors_this_frame:
                                st.session_state["clean_reps"] += 1

                    # track peak cm during "up" phase
                    if self.rep_phase == "up":
                        self._last_recorded_cm = max(self._last_recorded_cm, jump_cm)
                    else:
                        self._last_recorded_cm = 0.0

                    if form_errors_this_frame:
                        st.session_state["form_errors"].extend(form_errors_this_frame)

                    # jump bar
                    bar_max_px = int(h * 0.4)
                    bar_h = int(min(jump_cm / 60.0, 1.0) * bar_max_px)
                    cv2.rectangle(overlay, (w - 40, h - 60), (w - 20, h - 60 - bar_max_px),
                                  (30, 30, 30), -1)
                    cv2.rectangle(overlay, (w - 40, h - 60), (w - 20, h - 60 - bar_h),
                                  (255, 75, 43), -1)
                    cv2.putText(overlay, f"{jump_cm:.0f}cm",
                                (w - 72, h - 65 - bar_max_px),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 75, 43), 1)

                    if left_knee_angle:
                        color = (0, 255, 135) if not form_errors_this_frame else (255, 59, 59)
                        cv2.putText(overlay, f"L:{left_knee_angle:.0f}",
                                    (lknee_px[0] - 40, lknee_px[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(overlay, f"R:{right_knee_angle:.0f}",
                                    (rknee_px[0] + 10, rknee_px[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    for i, err in enumerate(form_errors_this_frame[:3]):
                        cv2.putText(overlay, f"! {err}",
                                    (15, 80 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 59, 59), 2)

                    if not form_errors_this_frame and jump_cm > 3:
                        cv2.putText(overlay, "CLEAN FORM ✓",
                                    (15, h - 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 135), 2)

            cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (255, 75, 43), 2)
            result = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
            return av.VideoFrame.from_ndarray(result, format="bgr24")


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    # USERNAME
    st.markdown("## 👤 PLAYER")
    st.markdown('<div class="orange-line"></div>', unsafe_allow_html=True)

    username_input = st.text_input("Username", value=st.session_state["username"],
                                    placeholder="Enter your name...")
    if username_input != st.session_state["username"]:
        st.session_state["username"] = username_input

    # Welcome back message
    username = st.session_state["username"].strip()
    db = st.session_state["all_time_history"]
    if username and username in db and db[username]:
        last_jump = db[username][-1]["cm"]
        st.markdown(f"""
        <div class="welcome-banner">
            Welcome back, <strong>{username}</strong>!<br>
            Your last jump was <strong>{last_jump} cm</strong>. Let's beat it! 🔥
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ATHLETE PROFILE
    st.markdown("## 🏋️ ATHLETE PROFILE")
    height_cm = st.number_input("Height (cm)", min_value=140, max_value=230, value=180, step=1)
    weight_kg = st.number_input("Weight (kg)", min_value=40, max_value=200, value=80, step=1)
    st.session_state["height_cm"] = height_cm
    st.session_state["weight_kg"] = weight_kg

    st.markdown("---")

    # SESSION STATS
    st.markdown("## 📊 SESSION STATS")
    max_j = st.session_state["max_jump_cm"]
    rank = get_rank(max_j)
    progress_pct = get_rank_progress(max_j)
    next_rank_idx = min(RANKS.index(rank) + 1, len(RANKS) - 1)
    next_rank = RANKS[next_rank_idx]

    st.markdown(f"""
    <div class="rank-card">
        <span class="rank-icon">{rank['icon']}</span>
        <span class="rank-name">{rank['name']}</span>
        <span class="rank-range">{rank['min']}–{rank['max'] if rank['max'] < 999 else '∞'} cm</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.72rem;color:var(--muted);letter-spacing:1px;margin-bottom:4px">
        PROGRESS TO {next_rank['icon']} {next_rank['name'].upper()}
    </div>
    <div class="progress-bar-container">
        <div class="progress-bar-fill" style="width:{int(progress_pct*100)}%"></div>
    </div>
    <div style="font-size:0.75rem;color:var(--muted)">{int(progress_pct*100)}% — Best: {max_j:.1f} cm</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 🎒 EQUIPMENT")
    inventory = st.multiselect(
        "Available gear",
        ["Dumbbells", "Kettlebell", "Barbell", "Jump Rope", "No Equipment"],
        default=st.session_state["inventory"],
    )
    st.session_state["inventory"] = inventory

    st.markdown("---")
    st.markdown("## 🎯 EXERCISE MODE")
    exercise_mode = st.selectbox(
        "Track",
        ["jump", "squat", "lunge"],
        format_func=lambda x: x.upper(),
    )
    st.session_state["last_exercise"] = exercise_mode

    st.markdown("---")
    if st.button("🔄 RESET SESSION"):
        for k, v in defaults.items():
            if k not in ("all_time_history", "username"):  # preserve DB and username
                st.session_state[k] = v
        st.rerun()


# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────────

col_title, col_metrics = st.columns([3, 2])

with col_title:
    st.markdown('<p class="dunk-title">DUNK<br>LAB</p>', unsafe_allow_html=True)
    st.markdown('<p class="dunk-subtitle">Vertical Jump AI Trainer</p>', unsafe_allow_html=True)

with col_metrics:
    mc1, mc2, mc3 = st.columns(3)
    delta_reps = st.session_state["clean_reps"] - st.session_state["prev_clean_reps"]
    st.session_state["prev_clean_reps"] = st.session_state["clean_reps"]
    with mc1:
        st.metric("CLEAN REPS", st.session_state["clean_reps"],
                  delta=delta_reps if delta_reps else None)
    with mc2:
        st.metric("MAX JUMP", f"{st.session_state['max_jump_cm']:.0f} cm")
    with mc3:
        total = st.session_state["total_reps"]
        clean = st.session_state["clean_reps"]
        pct = int(clean / total * 100) if total > 0 else 0
        st.metric("CLEAN %", f"{pct}%")

st.markdown('<div class="orange-line"></div>', unsafe_allow_html=True)

tab_cam, tab_workout, tab_history = st.tabs(["📷 LIVE ANALYSIS", "💪 WORKOUT PLAN", "📊 SESSION LOG"])

# ─── TAB 1: CAMERA ───────────────────────────────────────────────────────────────
with tab_cam:
    if not MEDIAPIPE_OK:
        st.error(f"⚠️ MediaPipe / OpenCV не загрузились. Ошибка: `{MEDIAPIPE_ERROR}`")
        st.info("Убедитесь, что в репозитории есть файл `packages.txt` со строкой `libgl1`.")
    else:
        col_vid, col_info = st.columns([3, 1])

        with col_vid:
            st.markdown("### REAL-TIME POSE ANALYSIS")
            st.caption("Stand 2–3 metres from camera. Keep full body in frame.")

            RTC_CONFIG = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            webrtc_streamer(
                key="jump_analyzer",
                video_transformer_factory=JumpAnalyzer,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_transform=True,
            )

            # ── REAL-TIME CORRECTION BLOCK ──────────────────────────────────
            st.markdown("### ⚡ REAL-TIME CORRECTION")
            recent_errors = list(set(st.session_state["form_errors"][-10:]))
            correction = get_realtime_correction(recent_errors) if recent_errors else None

            if correction:
                st.markdown(f"""
                <div class="correction-box">
                    <span class="corr-label">COACH SAYS</span>
                    <div class="corr-text">💬 {correction}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="correction-box" style="border-color:rgba(0,255,135,0.3)">
                    <span class="corr-label" style="color:var(--green)">ALL CLEAR</span>
                    <div class="corr-text">✅ No corrections needed — technique looks solid!</div>
                </div>
                """, unsafe_allow_html=True)

            # ── JUMP HISTORY (last 10) under camera ────────────────────────
            history = st.session_state["jump_history"]
            if history:
                st.markdown("### 📈 JUMP HISTORY (THIS SESSION)")
                # line chart
                recent_10 = [e["cm"] for e in history[-10:]]
                st.line_chart(
                    {"Jump Height (cm)": recent_10},
                    height=180,
                    use_container_width=True,
                )

                st.markdown("### ATTEMPTS")
                for entry in reversed(history[-8:]):
                    st.markdown(f"""
                    <div class="history-card">
                        <div class="hc-left">
                            <div class="hc-jump">{entry['cm']} <span style="font-size:1rem;color:var(--muted)">cm</span></div>
                            <div class="hc-meta">{entry['exercise'].upper()} &nbsp;·&nbsp; {entry['ts']}</div>
                        </div>
                        <div class="hc-rank">{entry['rank_icon']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_info:
            st.markdown("### LIVE STATS")

            st.markdown(f"""
            <div class="stat-card">
                <h4>REPS</h4>
                <p style="font-size:2rem;font-family:'Bebas Neue',sans-serif;color:#FF4B2B">
                    {st.session_state['clean_reps']}
                    <span style="font-size:1rem;color:#555"> / {st.session_state['total_reps']}</span>
                </p>
            </div>
            <div class="stat-card">
                <h4>MAX HEIGHT</h4>
                <p style="font-size:2rem;font-family:'Bebas Neue',sans-serif;color:#FF4B2B">
                    {st.session_state['max_jump_cm']:.0f}
                    <span style="font-size:1rem;color:#555">cm</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            rank = get_rank(st.session_state["max_jump_cm"])
            st.markdown(f"""
            <div class="rank-card" style="text-align:center;padding:0.8rem">
                <span style="font-size:2rem">{rank['icon']}</span><br>
                <span style="font-family:'Bebas Neue',sans-serif;font-size:1.1rem;letter-spacing:2px;color:#FFD700">{rank['name']}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**INSTRUCTIONS**")
            exercise = st.session_state.get("last_exercise", "jump")
            instructions = {
                "jump": ["Stand upright — calibration ~2s", "Jump explosively", "Land softly, reset stance", "Watch knee angles on screen"],
                "squat": ["Feet shoulder width", "Descend until ~90°", "Drive through heels", "Keep back straight"],
                "lunge": ["Step forward firmly", "Rear knee near ground", "Push back to start", "Alternate legs"],
            }
            for tip in instructions.get(exercise, []):
                st.markdown(f"• {tip}")

            if recent_errors:
                st.markdown("**FORM ALERTS**")
                for err in recent_errors[:4]:
                    st.markdown(f'<div class="form-error">⚠️ {err}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="form-ok">✓ Form looks good!</div>', unsafe_allow_html=True)


# ─── TAB 2: WORKOUT PLAN ─────────────────────────────────────────────────────────
with tab_workout:
    st.markdown("### AI-GENERATED TRAINING PLAN")

    col_gen, col_info2 = st.columns([2, 1])

    with col_gen:
        openai_api_key = None
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass

        if not openai_api_key:
            st.warning("🔑 OpenAI API Key не найден. Добавьте `OPENAI_API_KEY` в **Settings → Secrets** вашего приложения на Streamlit Cloud.")
        else:
            if st.button("⚡ GENERATE MY PLAN", use_container_width=True):
                with st.spinner("GPT-4o is designing your program..."):
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=openai_api_key)
                        plan = generate_workout_plan(
                            client, height_cm, weight_kg, inventory,
                            st.session_state["form_errors"],
                        )
                        st.session_state["workout_plan"] = plan
                        st.success("Plan generated!")
                    except json.JSONDecodeError:
                        st.error("Could not parse the AI response. Please try again.")
                    except Exception as e:
                        st.error(f"API error: {e}")

        if st.session_state["workout_plan"]:
            plan = st.session_state["workout_plan"]
            st.markdown(f"""
            <div class="stat-card">
                <h4>{plan.get('program_name', 'YOUR PROGRAM')}</h4>
                <p>{plan.get('goal', '')}</p>
            </div>
            """, unsafe_allow_html=True)

            for week_data in plan.get("weeks", []):
                st.markdown(f"#### Week {week_data['week']} — {week_data.get('focus','')}")
                for day_data in week_data.get("days", []):
                    st.markdown(f"**{day_data['day']}**")
                    for ex in day_data.get("exercises", []):
                        equip = ex.get('equipment', '')
                        badge_html = f'<span class="badge">{equip}</span>' if equip else ''
                        st.markdown(f"""
                        <div class="workout-card">
                            <div class="exercise-name">{ex.get('name','')}</div>
                            <div class="exercise-detail">
                                {badge_html}
                                {ex.get('sets',3)} sets × {ex.get('reps','10')} &nbsp;|&nbsp; Rest: {ex.get('rest_sec',60)}s
                            </div>
                            <div class="exercise-detail" style="margin-top:0.3rem;font-style:italic">
                                💡 {ex.get('cue','')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("---")

            tips = plan.get("tips", [])
            if tips:
                st.markdown("#### COACH TIPS")
                for tip in tips:
                    st.markdown(f"• {tip}")

    with col_info2:
        st.markdown("### YOUR PROFILE")
        st.markdown(f"""
        <div class="stat-card"><h4>HEIGHT</h4><p>{height_cm} cm</p></div>
        <div class="stat-card"><h4>WEIGHT</h4><p>{weight_kg} kg</p></div>
        <div class="stat-card"><h4>EQUIPMENT</h4><p>{', '.join(inventory) if inventory else 'None'}</p></div>
        """, unsafe_allow_html=True)

        errors = list(set(st.session_state["form_errors"]))
        if errors:
            st.markdown("### FORM ISSUES LOGGED")
            st.caption("These will be included in your AI plan:")
            for e in errors[:6]:
                st.markdown(f'<div class="form-error">⚠️ {e}</div>', unsafe_allow_html=True)

        # Correction hint in workout tab too
        if errors:
            correction = get_realtime_correction(errors)
            st.markdown(f"""
            <div class="correction-box" style="margin-top:1rem">
                <span class="corr-label">TOP CORRECTION DRILL</span>
                <div class="corr-text">💬 {correction}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── TAB 3: SESSION LOG ──────────────────────────────────────────────────────────
with tab_history:
    st.markdown("### SESSION SUMMARY")

    s1, s2, s3, s4 = st.columns(4)
    total = st.session_state["total_reps"]
    clean = st.session_state["clean_reps"]
    pct = round(clean / total * 100, 1) if total > 0 else 0
    max_j = st.session_state["max_jump_cm"]

    with s1: st.metric("Total Reps", total)
    with s2: st.metric("Clean Reps", clean)
    with s3: st.metric("Form Score", f"{pct}%")
    with s4: st.metric("Peak Jump", f"{max_j:.1f} cm")

    st.markdown('<div class="orange-line"></div>', unsafe_allow_html=True)

    # All-time stats for user
    username = st.session_state["username"].strip()
    db = st.session_state["all_time_history"]
    user_history = db.get(username or "anonymous", [])

    if user_history:
        st.markdown(f"### 📈 {username.upper() if username else 'SESSION'} — JUMP CHART")
        all_cms = [e["cm"] for e in user_history]
        st.line_chart({"Height (cm)": all_cms}, height=200, use_container_width=True)

        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            st.markdown(f"**Best:** {max(all_cms):.1f} cm  {get_rank(max(all_cms))['icon']}")
            st.markdown(f"**Average:** {sum(all_cms)/len(all_cms):.1f} cm")
            st.markdown(f"**Attempts:** {len(all_cms)}")

        with col_h2:
            rank = get_rank(max(all_cms))
            st.markdown(f"""
            <div class="rank-card">
                <span class="rank-icon">{rank['icon']}</span>
                <span class="rank-name">{rank['name']}</span>
                <span class="rank-range">Best: {max(all_cms):.1f} cm</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### ATTEMPT LOG")
        for entry in reversed(user_history[-15:]):
            st.markdown(f"""
            <div class="history-card">
                <div class="hc-left">
                    <div class="hc-jump">{entry['cm']} <span style="font-size:1rem;color:var(--muted)">cm</span></div>
                    <div class="hc-meta">{entry['exercise'].upper()} &nbsp;·&nbsp; {entry['ts']} &nbsp;·&nbsp; {entry['rank']}</div>
                </div>
                <div class="hc-rank">{entry['rank_icon']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="orange-line"></div>', unsafe_allow_html=True)

    # Form error log
    errors = st.session_state["form_errors"]
    if errors:
        st.markdown("### FORM ERROR LOG")
        error_counts = Counter(errors)
        for err, count in error_counts.most_common():
            pct_e = round(count / len(errors) * 100)
            severity = "badge-red" if pct_e > 40 else ""
            correction_hint = REAL_TIME_CORRECTIONS.get(err, "")
            st.markdown(f"""
            <div class="workout-card">
                <span class="badge {severity}">{count}×</span>
                <span style="font-size:0.9rem">{err}</span>
                <span style="color:var(--muted);font-size:0.8rem;margin-left:8px">({pct_e}% of errors)</span>
                {f'<div class="exercise-detail" style="margin-top:0.4rem;font-style:italic;color:#AAA">💡 Fix: {correction_hint}</div>' if correction_hint else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="form-ok" style="font-size:1rem">🎉 No form errors recorded this session!</div>',
                    unsafe_allow_html=True)

    # Coach verdict
    st.markdown("### COACH VERDICT")
    if pct >= 80:
        msg = "Excellent session! Your form is consistent. Focus on progressive overload."
        color = "var(--green)"
    elif pct >= 50:
        msg = "Good effort. Review the form errors above and focus on technique before adding load."
        color = "var(--orange)"
    else:
        msg = "Form needs work. Run the AI plan and prioritize the corrective drills."
        color = "var(--red)"
    st.markdown(f'<p style="color:{color};font-size:1rem;font-weight:500">{msg}</p>', unsafe_allow_html=True)

    # Google Sheets integration stub
    with st.expander("📡 Google Sheets Integration (Cloud Persistence)"):
        st.markdown("""
        To persist data across sessions, connect Google Sheets:

        ```python
        # requirements.txt
        st-gsheets-connection

        # secrets.toml
        [connections.gsheets]
        spreadsheet = "https://docs.google.com/spreadsheets/d/YOUR_ID"
        type = "service_account"
        project_id = "..."
        private_key = "..."
        client_email = "..."

        # In your app:
        from streamlit_gsheets import GSheetsConnection
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="jumps")
        conn.update(worksheet="jumps", data=df)
        ```

        Each jump entry saves: `username, timestamp, height_cm, rank, exercise, form_errors`
        """)
