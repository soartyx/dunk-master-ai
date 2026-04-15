import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
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
    --orange: #FF5C00;
    --black: #0A0A0A;
    --dark: #111111;
    --card: #181818;
    --border: #2A2A2A;
    --white: #F5F0E8;
    --muted: #666666;
    --green: #00FF87;
    --red: #FF3B3B;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--black);
    color: var(--white);
}

.stApp { background-color: var(--black); }

/* Sidebar */
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

/* Title */
.dunk-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    line-height: 0.9;
    letter-spacing: 4px;
    color: var(--white);
    text-shadow: 4px 4px 0px var(--orange);
    margin: 0;
}
.dunk-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* Metric overrides */
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.5rem !important;
    color: var(--orange) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted) !important;
    font-size: 0.7rem !important;
}
[data-testid="stMetricDelta"] svg { fill: var(--green); }

/* Cards */
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.stat-card h4 {
    font-family: 'Bebas Neue', sans-serif;
    color: var(--orange);
    letter-spacing: 2px;
    font-size: 1.1rem;
    margin: 0 0 0.3rem 0;
}
.stat-card p { color: var(--white); font-size: 0.9rem; margin: 0; }

/* Workout table */
.workout-card {
    background: var(--card);
    border-left: 3px solid var(--orange);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.4rem;
    margin-bottom: 0.6rem;
}
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
.badge {
    display: inline-block;
    background: var(--orange);
    color: var(--black);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
}
.badge-green { background: var(--green); }
.badge-red { background: var(--red); }

/* Form errors */
.form-error {
    background: rgba(255,59,59,0.12);
    border: 1px solid var(--red);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    color: var(--red);
    font-size: 0.85rem;
    margin-top: 0.4rem;
}
.form-ok {
    background: rgba(0,255,135,0.1);
    border: 1px solid var(--green);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    color: var(--green);
    font-size: 0.85rem;
    margin-top: 0.4rem;
}

/* Divider */
.orange-line {
    height: 2px;
    background: linear-gradient(90deg, var(--orange), transparent);
    margin: 1.5rem 0;
}

/* Tabs */
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

/* Buttons */
.stButton > button {
    background: var(--orange);
    color: var(--black);
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
    font-size: 1rem;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 2rem;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: var(--white);
    color: var(--black);
    transform: translateY(-1px);
}

/* Number inputs */
.stNumberInput input, .stTextInput input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
    border-radius: 4px !important;
}
.stSelectbox > div, .stMultiSelect > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE INIT ──────────────────────────────────────────────────────────
defaults = {
    "clean_reps": 0,
    "total_reps": 0,
    "max_jump_cm": 0.0,
    "form_errors": [],
    "last_exercise": "jump",
    "workout_plan": None,
    "prev_clean_reps": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── MEDIAPIPE SETUP ─────────────────────────────────────────────────────────────
# --- МЕДИАПАЙП СЕТАП ---
import mediapipe.python.solutions.pose as pose_solution
import mediapipe.python.solutions.drawing_utils as drawing_utils_solution
import mediapipe.python.solutions.drawing_styles as drawing_styles_solution

mp_pose = pose_solution
mp_drawing = drawing_utils_solution
mp_drawing_styles = drawing_styles_solution
# mp_pose = mp_pose_module
# mp_drawing = mp_drawing_module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Calculate angle at point B formed by A-B-C."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))


def px_to_cm(px_diff, height_cm, frame_height, hip_y_frac):
    """Convert pixel difference to cm using user height for calibration.
    Assumption: full body occupies ~85% of frame height when standing."""
    body_px = frame_height * 0.85
    px_per_cm = body_px / height_cm
    return px_diff / (px_per_cm + 1e-9)


def generate_workout_plan(client, height_cm, weight_kg, inventory, form_errors):
    """Call GPT-4o to get a personalized jump training plan."""
    error_summary = ""
    if form_errors:
        unique_errors = list(set(form_errors[-20:]))
        error_summary = f"\n\nForm issues detected during session: {', '.join(unique_errors)}."

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
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─── VIDEO TRANSFORMER ────────────────────────────────────────────────────────────

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
        self.in_air = False
        self.rep_phase = "down"  # down / up
        self.knee_angle_buffer = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Dark overlay for readability
        overlay = img.copy()

        form_errors_this_frame = []
        jump_cm = 0.0
        left_knee_angle = None
        right_knee_angle = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Draw skeleton
            mp_drawing.draw_landmarks(
                overlay,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 92, 0), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(80, 80, 80), thickness=1
                ),
            )

            # Key landmarks (normalized coords → pixel)
            def lp(idx):
                l = lm[idx]
                return (int(l.x * w), int(l.y * h)), l.y  # (px, px), norm_y

            (lhip_px, lhip_y_n) = lp(mp_pose.PoseLandmark.LEFT_HIP)
            (rhip_px, rhip_y_n) = lp(mp_pose.PoseLandmark.RIGHT_HIP)
            (lknee_px, _) = lp(mp_pose.PoseLandmark.LEFT_KNEE)
            (rknee_px, _) = lp(mp_pose.PoseLandmark.RIGHT_KNEE)
            (lank_px, _) = lp(mp_pose.PoseLandmark.LEFT_ANKLE)
            (rank_px, _) = lp(mp_pose.PoseLandmark.RIGHT_ANKLE)
            (lsho_px, _) = lp(mp_pose.PoseLandmark.LEFT_SHOULDER)
            (rsho_px, _) = lp(mp_pose.PoseLandmark.RIGHT_SHOULDER)

            mid_hip_y_norm = (lhip_y_n + rhip_y_n) / 2
            mid_hip_y_px = int(mid_hip_y_norm * h)

            # ── Calibration (first 30 frames) ──
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
                # ── Jump height calculation ──
                height_cm_val = st.session_state.get("height_cm", 180)
                delta_norm = self.rest_hip_y - mid_hip_y_norm  # positive = jumped up
                body_norm_height = 0.85  # fraction of frame
                cm_per_norm = height_cm_val / body_norm_height
                jump_cm = max(0.0, delta_norm * cm_per_norm)

                if jump_cm > st.session_state["max_jump_cm"]:
                    st.session_state["max_jump_cm"] = round(jump_cm, 1)

                # ── Knee angle calculation ──
                left_knee_angle = calculate_angle(lhip_px, lknee_px, lank_px)
                right_knee_angle = calculate_angle(rhip_px, rknee_px, rank_px)

                # ── Form checks ──
                exercise = st.session_state.get("last_exercise", "jump")

                if exercise in ("squat", "jump"):
                    # Knee cave check: knee x should be between hip and ankle x (approx)
                    if lknee_px[0] < lank_px[0] - 30:
                        form_errors_this_frame.append("Left knee caving in")
                    if rknee_px[0] > rank_px[0] + 30:
                        form_errors_this_frame.append("Right knee caving in")

                    # Depth check for squats
                    if exercise == "squat":
                        if left_knee_angle > 110 or right_knee_angle > 110:
                            form_errors_this_frame.append("Squat too shallow — go deeper")

                if exercise == "lunge":
                    if left_knee_angle < 70 or right_knee_angle < 70:
                        form_errors_this_frame.append("Knee angle too acute in lunge")

                # Back straightness: shoulder–hip vertical alignment
                sho_mid_x = (lsho_px[0] + rsho_px[0]) / 2
                hip_mid_x = (lhip_px[0] + rhip_px[0]) / 2
                if abs(sho_mid_x - hip_mid_x) > w * 0.12:
                    form_errors_this_frame.append("Keep your back straight")

                # ── Rep counting (jump) ──
                if exercise == "jump":
                    if jump_cm > 5 and self.rep_phase == "down":
                        self.rep_phase = "up"
                    elif jump_cm < 2 and self.rep_phase == "up":
                        self.rep_phase = "down"
                        st.session_state["total_reps"] += 1
                        if not form_errors_this_frame:
                            st.session_state["clean_reps"] += 1

                if exercise == "squat":
                    avg_knee = (left_knee_angle + right_knee_angle) / 2
                    if avg_knee < 95 and self.rep_phase == "down":
                        self.rep_phase = "up"
                    elif avg_knee > 160 and self.rep_phase == "up":
                        self.rep_phase = "down"
                        st.session_state["total_reps"] += 1
                        if not form_errors_this_frame:
                            st.session_state["clean_reps"] += 1

                # ── Store errors ──
                if form_errors_this_frame:
                    st.session_state["form_errors"].extend(form_errors_this_frame)

                # ── HUD Drawing ──
                # Jump height bar
                bar_max_px = int(h * 0.4)
                bar_h = int(min(jump_cm / 60.0, 1.0) * bar_max_px)
                cv2.rectangle(overlay, (w - 40, h - 60), (w - 20, h - 60 - bar_max_px),
                              (40, 40, 40), -1)
                cv2.rectangle(overlay, (w - 40, h - 60), (w - 20, h - 60 - bar_h),
                              (255, 92, 0), -1)
                cv2.putText(overlay, f"{jump_cm:.0f}cm",
                            (w - 70, h - 65 - bar_max_px),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 92, 0), 1)

                # Knee angles
                if left_knee_angle:
                    color = (0, 255, 135) if not form_errors_this_frame else (255, 59, 59)
                    cv2.putText(overlay, f"L:{left_knee_angle:.0f}",
                                (lknee_px[0] - 40, lknee_px[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.putText(overlay, f"R:{right_knee_angle:.0f}",
                                (rknee_px[0] + 10, rknee_px[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Form errors on screen
                for i, err in enumerate(form_errors_this_frame[:3]):
                    cv2.putText(overlay, f"⚠ {err}",
                                (15, 80 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 59, 59), 2)

                # Clean rep indicator
                if not form_errors_this_frame and jump_cm > 3:
                    cv2.putText(overlay, "CLEAN FORM ✓",
                                (15, h - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 135), 2)

        # Orange frame border
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (255, 92, 0), 2)

        # Blend overlay
        result = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
        return av.VideoFrame.from_ndarray(result, format="bgr24")


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 ATHLETE PROFILE")
    st.markdown('<div class="orange-line"></div>', unsafe_allow_html=True)

    height_cm = st.number_input("Height (cm)", min_value=140, max_value=230,
                                  value=180, step=1)
    weight_kg = st.number_input("Weight (kg)", min_value=40, max_value=200,
                                  value=80, step=1)
    st.session_state["height_cm"] = height_cm
    st.session_state["weight_kg"] = weight_kg

    st.markdown("---")
    st.markdown("## 🎒 EQUIPMENT")
    inventory = st.multiselect(
        "Available gear",
        ["Dumbbells", "Kettlebell", "Barbell", "Jump Rope", "No Equipment"],
        default=["No Equipment"],
    )
    st.session_state["inventory"] = inventory

    st.markdown("---")
    st.markdown("## 🤖 AI ENGINE")
    api_key = st.text_input("OpenAI API Key", type="password",
                              placeholder="sk-...")

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
            st.session_state[k] = v
        st.rerun()

# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────────

# Header
col_title, col_metrics = st.columns([3, 2])

with col_title:
    st.markdown('<p class="dunk-title">DUNK<br>LAB</p>', unsafe_allow_html=True)
    st.markdown('<p class="dunk-subtitle">Vertical Jump AI Trainer</p>',
                unsafe_allow_html=True)

with col_metrics:
    mc1, mc2, mc3 = st.columns(3)
    delta_reps = st.session_state["clean_reps"] - st.session_state["prev_clean_reps"]
    st.session_state["prev_clean_reps"] = st.session_state["clean_reps"]
    with mc1:
        st.metric("PROGRESS", st.session_state["clean_reps"],
                  delta=delta_reps if delta_reps else None,
                  help="Clean reps this session")
    with mc2:
        st.metric("MAX JUMP", f"{st.session_state['max_jump_cm']:.0f} cm",
                  help="Peak jump height detected")
    with mc3:
        total = st.session_state["total_reps"]
        clean = st.session_state["clean_reps"]
        pct = int(clean / total * 100) if total > 0 else 0
        st.metric("CLEAN %", f"{pct}%", help="Clean form percentage")

st.markdown('<div class="orange-line"></div>', unsafe_allow_html=True)

# Tabs
tab_cam, tab_workout, tab_history = st.tabs(["📷 LIVE ANALYSIS", "💪 WORKOUT PLAN", "📊 SESSION LOG"])

# ─── TAB 1: CAMERA ───────────────────────────────────────────────────────────────
with tab_cam:
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

    with col_info:
        st.markdown("### LIVE STATS")

        st.markdown(f"""
        <div class="stat-card">
            <h4>REPS</h4>
            <p style="font-size:2rem;font-family:'Bebas Neue',sans-serif;color:#FF5C00">
                {st.session_state['clean_reps']} <span style="font-size:1rem;color:#666"> / {st.session_state['total_reps']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-card">
            <h4>MAX HEIGHT</h4>
            <p style="font-size:2rem;font-family:'Bebas Neue',sans-serif;color:#FF5C00">
                {st.session_state['max_jump_cm']:.0f} <span style="font-size:1rem;color:#666">cm</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**INSTRUCTIONS**")
        exercise = st.session_state.get("last_exercise", "jump")
        instructions = {
            "jump": ["Stand upright — calibration takes ~2s", "Jump explosively", "Land softly, reset stance", "Watch knee angles on screen"],
            "squat": ["Feet shoulder width", "Descend until ~90°", "Drive through heels", "Keep back straight"],
            "lunge": ["Step forward firmly", "Rear knee near ground", "Push back to start", "Alternate legs"],
        }
        for tip in instructions.get(exercise, []):
            st.markdown(f"• {tip}")

        # Recent form errors
        recent_errors = list(set(st.session_state["form_errors"][-10:]))
        if recent_errors:
            st.markdown("**FORM ALERTS**")
            for err in recent_errors[:5]:
                st.markdown(f'<div class="form-error">⚠️ {err}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="form-ok">✓ Form looks good!</div>',
                        unsafe_allow_html=True)

# ─── TAB 2: WORKOUT PLAN ─────────────────────────────────────────────────────────
with tab_workout:
    st.markdown("### AI-GENERATED TRAINING PLAN")

    col_gen, col_info2 = st.columns([2, 1])

    with col_gen:
        if not api_key:
            st.info("🔑 Enter your OpenAI API Key in the sidebar to generate a personalized plan.")
        else:
            if st.button("⚡ GENERATE MY PLAN", use_container_width=True):
                with st.spinner("GPT-4o is designing your program..."):
                    try:
                        client = OpenAI(api_key=api_key)
                        plan = generate_workout_plan(
                            client,
                            height_cm,
                            weight_kg,
                            inventory,
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
                                {ex.get('sets',3)} sets × {ex.get('reps','10')} &nbsp;|&nbsp;
                                Rest: {ex.get('rest_sec',60)}s
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
        <div class="stat-card">
            <h4>HEIGHT</h4><p>{height_cm} cm</p>
        </div>
        <div class="stat-card">
            <h4>WEIGHT</h4><p>{weight_kg} kg</p>
        </div>
        <div class="stat-card">
            <h4>EQUIPMENT</h4><p>{', '.join(inventory) if inventory else 'None'}</p>
        </div>
        """, unsafe_allow_html=True)

        errors = list(set(st.session_state["form_errors"]))
        if errors:
            st.markdown("### FORM ISSUES LOGGED")
            st.caption("These will be included in your AI plan:")
            for e in errors[:6]:
                st.markdown(f'<div class="form-error">⚠️ {e}</div>',
                            unsafe_allow_html=True)

# ─── TAB 3: SESSION LOG ──────────────────────────────────────────────────────────
with tab_history:
    st.markdown("### SESSION SUMMARY")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Reps", st.session_state["total_reps"])
    with s2:
        st.metric("Clean Reps", st.session_state["clean_reps"])
    with s3:
        total = st.session_state["total_reps"]
        clean = st.session_state["clean_reps"]
        pct = round(clean / total * 100, 1) if total > 0 else 0
        st.metric("Form Score", f"{pct}%")
    with s4:
        st.metric("Peak Jump", f"{st.session_state['max_jump_cm']:.1f} cm")

    st.markdown('<div class="orange-line"></div>', unsafe_allow_html=True)

    errors = st.session_state["form_errors"]
    if errors:
        st.markdown("### FORM ERROR LOG")
        from collections import Counter
        error_counts = Counter(errors)
        for err, count in error_counts.most_common():
            pct_e = round(count / len(errors) * 100)
            severity = "badge-red" if pct_e > 40 else ""
            st.markdown(f"""
            <div class="workout-card">
                <span class="badge {severity}">{count}×</span>
                <span style="font-size:0.9rem">{err}</span>
                <span style="color:var(--muted);font-size:0.8rem;margin-left:8px">({pct_e}% of errors)</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="form-ok" style="font-size:1rem">🎉 No form errors recorded this session!</div>',
                    unsafe_allow_html=True)

    st.markdown("### WHAT THIS MEANS")
    if pct >= 80:
        msg = "Excellent session! Your form is consistent. Focus on progressive overload."
        color = "var(--green)"
    elif pct >= 50:
        msg = "Good effort. Review the form errors above and focus on technique before adding load."
        color = "var(--orange)"
    else:
        msg = "Form needs work. Run the AI plan and prioritize the corrective drills."
        color = "var(--red)"
    st.markdown(f'<p style="color:{color};font-size:1rem">{msg}</p>',
                unsafe_allow_html=True)
