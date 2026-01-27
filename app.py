import gradio as gr
import cv2
import numpy as np
import os
import random
import time
import csv
import uuid
import shutil
from datetime import datetime
from functools import partial

# --- Configuration ---
AI_FOLDER = "./AI"
HUMAN_FOLDER = "./Human"
PART1_CSV_FILE = "emotion_responses_part1.csv"
PART2_CSV_FILE = "emotion_responses_part2.csv"
METADATA_FILE = "stimuli_metadata.csv"
DEBLUR_DURATION_S = 5  # Seconds to go from Blur -> Clear

# --- Advanced Features Config ---
URL_PARAM_PARTICIPANT_ID = "pid"
# Keep emotion order fixed across all participants.
RANDOMIZE_EMOTION_ORDER_DEFAULT = False
RANDOMIZE_EMOTION_ORDER_PARAM = "randomize"
CHOICE_PLACEHOLDER = "Select an emotion..."

# --- Sampling Config ---
BALANCE_SUBSET_DEFAULT = True
MAX_PER_STRATUM = None  # Optionally set to an int to cap trials per (type, emotion)
ALLOWED_ANGLES = {"forward"}  # Restrict to front-facing stimuli.

# --- CSS STYLES ---
APP_CSS = f"""
#emotion_choice, #emotion_choice .wrap {{ max-height: 260px; overflow-y: auto; }}
#start_btn > button, 
#next_btn > button {{
  font-size: 20px !important;
  padding: 12px 22px !important;
  min-height: 48px !important;
}}
#emotion_choice label,
#emotion_choice .wrap label,
#emotion_choice .wrap span {{
  font-size: 20px !important;
}}
#emotion_choice .wrap {{
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  justify-content: center !important;
  align-items: center !important;
  gap: 8px 12px;
}}
#emotion_choice .wrap label {{
  justify-content: center;
  width: auto;
  margin: 4px 0;
}}
#emotion_choice input[type="radio"] {{
  transform: scale(1.2);
  margin-right: 8px;
}}
#emotion_choice .wrap label {{
  padding: 8px 12px !important;
}}

@media (max-width: 640px) {{
  #img_anim img {{ max-height: 280px; object-fit: contain; }}
}}

#img_anim img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
}}

/* --- ANIMATED IMAGE (The Test) --- */
/* 1. Start HEAVILY BLURRED by default */
#img_anim img {{
    filter: blur(50px); 
    display: block;
    transform: scale(1.0);
}}

/* 2. The JS adds this class to animate it to clear */
.image-clear {{
    transition: filter {DEBLUR_DURATION_S}s linear !important;
    filter: blur(0px) !important;
}}

/* Snap instantly to clear when the participant selects an answer. */
.image-snap {{
    transition: none !important;
    filter: blur(0px) !important;
}}

/* Force a blurred state immediately (used before loading the next image). */
.image-preblur {{
    transition: none !important;
    filter: blur(50px) !important;
}}

#progress_text {{
  font-size: 36px;
  text-align: center;
  line-height: 1.2;
}}

#part2_section {{
  padding: 16px 20px;
  box-sizing: border-box;
}}

#part2_section h1,
#part2_section h2 {{
  font-size: 44px !important;
}}

#part2_instructions_section {{
  text-align: center;
}}

#part2_instructions_section h1 {{
  font-size: 64px !important;
  margin-bottom: 8px !important;
}}

#part2_instructions_section h2 {{
  font-size: 28px !important;
}}

#part2_start_btn > button {{
  font-size: 22px !important;
  padding: 12px 26px !important;
}}

#part2_completion_text {{
  text-align: center;
}}

#part2_completion_text h1 {{
  font-size: 140px !important;
  margin: 0 !important;
  line-height: 1 !important;
}}

#part2_completion_text h2 {{
  font-size: 48px !important;
  margin-top: 8px !important;
}}

#part2_section input[type="number"] {{
  font-size: 30px !important;
  font-weight: 700 !important;
}}

#part2_section label,
#part2_section .wrap label,
#part2_section .wrap span {{
  font-size: 20px !important;
}}

#part2_section .wrap {{
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  gap: 8px 12px;
}}

#part2_section input[type="radio"] {{
  transform: scale(1.35);
  margin-right: 8px;
}}

#part2_section .wrap label {{
  padding: 6px 10px !important;
}}

#app_title {{
  text-align: center;
  margin-bottom: 16px;
}}

#app_title h1,
#app_title p {{
  font-weight: 700;
  font-size: 56px;
  margin: 0;
}}

"""

# --- Constants & Mappings ---
UNKNOWN_LABEL = "unknown"
FILENAME_FIELD_ORDER = ["emotion"]

# Fixed emotion set and order for all trials.
EMOTION_CHOICES_ORDER = [
    "neutral",
    "happy",
    "angry",
    "afraid",
    "disgusted",
    "sad",
    "surprised",
]
ALLOWED_EMOTIONS = set(EMOTION_CHOICES_ORDER)
EMOTION_ALIASES = {
    "fearful": "afraid",
    "fear": "afraid",
}

# --- Stimulus Types ---
STIMULUS_TYPE_REAL = "real_kdef"
STIMULUS_TYPE_AI = "ai_kdef_like"

# --- Ratings Config (Part 2) ---
RATING_SCALE_MIN = 1
RATING_SCALE_MAX = 7
SCALE_CHOICES = list(range(RATING_SCALE_MIN, RATING_SCALE_MAX + 1))

# --- Part 2 Rating Keys ---
PART2_KEYS = ["age", "masc", "attr", "quality", "artifact"]

# Part-specific outputs: one row per image per part, with minimal metadata.
PART1_HEADERS = [
    "participant_id",
    "session_id",
    "stimulus_id",
    "stimulus_type",
    "target_emotion",
    "emotion_trial_index",
    "emotion_rt_ms",
    "selected_emotion",
    "accuracy",
    "emotion_timestamp",
]

PART2_HEADERS = [
    "participant_id",
    "session_id",
    "stimulus_id",
    "stimulus_type",
    "target_emotion",
    "matching_trial_index",
    "match_age_rating",
    "match_masc_rating",
    "match_attr_rating",
    "match_quality_rating",
    "match_artifact_rating",
    "matching_timestamp",
]

# --- Data Structure ---
class ImageData:
    def __init__(
        self,
        path,
        source,
        emotion,
        sex=UNKNOWN_LABEL,
        ethnicity=UNKNOWN_LABEL,
        angle=UNKNOWN_LABEL,
        face_type=UNKNOWN_LABEL,
        stimulus_type=UNKNOWN_LABEL,
        stimulus_id="",
    ):
        self.path = path
        self.source = source
        self.emotion = emotion
        self.sex = sex
        self.ethnicity = ethnicity
        self.angle = angle
        self.face_type = face_type
        self.name = os.path.basename(path)
        self.stimulus_id = stimulus_id or os.path.splitext(self.name)[0].strip().lower()
        self.stimulus_type = stimulus_type or UNKNOWN_LABEL

# --- Helper Functions ---
def normalize_label(value):
    if value is None: return ""
    return str(value).strip().lower().replace(" ", "-")

def canonicalize_emotion(label):
    norm = normalize_label(label)
    if not norm:
        return ""
    return EMOTION_ALIASES.get(norm, norm)

def load_metadata(metadata_path):
    if not os.path.exists(metadata_path): return {}
    metadata = {}
    with open(metadata_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("image_name") or row.get("filename") or row.get("image")
            if not name: continue
            key = name.strip().lower()
            entry = {
                "emotion": normalize_label(row.get("emotion")),
                "sex": normalize_label(row.get("sex")),
                "ethnicity": normalize_label(row.get("ethnicity")),
                "angle": normalize_label(row.get("angle")),
                "face_type": normalize_label(row.get("face_type") or row.get("type") or row.get("source")),
            }
            metadata[key] = entry
            stem = os.path.splitext(key)[0]
            metadata.setdefault(stem, entry)
    return metadata

def parse_filename_fields(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    parts = base_name.split('_')
    if len(parts) < 2: return {}
    fields = {}
    for field in FILENAME_FIELD_ORDER:
        if not parts: break
        fields[field] = normalize_label(parts.pop())
    return fields

def resolve_field(metadata, filename_fields, key, default=UNKNOWN_LABEL):
    value = ""
    if metadata: value = normalize_label(metadata.get(key))
    if not value: value = filename_fields.get(key, "")
    return value or default

def resolve_face_type(metadata, source):
    if metadata and metadata.get("face_type"): return normalize_label(metadata.get("face_type"))
    return normalize_label(source)

def resolve_stimulus_type(face_type, source):
    ft = normalize_label(face_type) or normalize_label(source)
    if ft in {"human", "real", "real-kdef", "real_kdef"}:
        return STIMULUS_TYPE_REAL
    if ft in {"ai", "synthetic", "ai-kdef-like", "ai_kdef_like"}:
        return STIMULUS_TYPE_AI
    # Fall back to folder/source label.
    return STIMULUS_TYPE_REAL if normalize_label(source) == "human" else STIMULUS_TYPE_AI

def make_stimulus_id(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.strip().lower()

def select_balanced_subset(images, max_per_stratum=None):
    if not images:
        return []
    strata = {}
    for img in images:
        key = (img.stimulus_type, img.emotion)
        strata.setdefault(key, []).append(img)
    counts = {k: len(v) for k, v in strata.items()}
    if not counts:
        return images
    min_count = min(counts.values())
    if max_per_stratum is not None:
        min_count = min(min_count, int(max_per_stratum))
    if min_count <= 0:
        return images

    sampled = []
    for key, items in strata.items():
        if len(items) <= min_count:
            sampled.extend(items)
        else:
            sampled.extend(random.sample(items, k=min_count))
    random.shuffle(sampled)
    print(f"[DEBUG] Balanced subset: {len(sampled)} trials across {len(strata)} strata (per-stratum={min_count}).")
    return sampled

def build_row_template(state, image_data):
    # Minimal row template: only fields that are written to the part CSVs.
    return {
        "participant_id": state.get("participant_id", ""),
        "session_id": state.get("session_id", ""),
        "stimulus_id": image_data.stimulus_id,
        "stimulus_type": image_data.stimulus_type,
        "target_emotion": image_data.emotion,
        # Part 1 fields (filled in later).
        "emotion_trial_index": "",
        "emotion_rt_ms": "",
        "selected_emotion": "",
        "accuracy": "",
        "emotion_timestamp": "",
        # Part 2 fields (filled in later).
        "matching_trial_index": "",
        "match_age_rating": "",
        "match_masc_rating": "",
        "match_attr_rating": "",
        "match_quality_rating": "",
        "match_artifact_rating": "",
        "matching_timestamp": "",
    }

def ensure_csv_file_for(file_path, headers):
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        return file_path, ""

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)
    if existing_header != headers:
        base, ext = os.path.splitext(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{base}_{timestamp}{ext or '.csv'}"
        try:
            shutil.copy2(file_path, backup_file)
            backup_msg = f"Copied existing results to: {backup_file}"
        except Exception as e:
            backup_msg = f"Could not copy existing results ({e})."

        # Reinitialize the base file with the expected header.
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        return file_path, f"{backup_msg}\nReinitialized results file: {file_path}"
    return file_path, ""

def ensure_csv_files():
    part1_file, part1_status = ensure_csv_file_for(PART1_CSV_FILE, PART1_HEADERS)
    part2_file, part2_status = ensure_csv_file_for(PART2_CSV_FILE, PART2_HEADERS)
    statuses = [s for s in [part1_status, part2_status] if s]
    status_lines = [
        f"Part 1 file: {part1_file}",
        f"Part 2 file: {part2_file}",
    ]
    status_lines.extend(statuses)
    status_msg = "\n".join(status_lines)
    return part1_file, part2_file, status_msg

def get_participant_id(request):
    if request is None: return ""
    pid = request.query_params.get(URL_PARAM_PARTICIPANT_ID)
    return str(pid).strip() if pid else ""

def scan_images():
    images = []
    emotions = set()
    metadata = load_metadata(METADATA_FILE)
    skipped_missing_emotion = []
    skipped_angle = []
    skipped_emotion = []

    for folder, source in [(AI_FOLDER, "AI"), (HUMAN_FOLDER, "Human")]:
        if not os.path.exists(folder): continue
        for filename in os.listdir(folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            path = os.path.join(folder, filename)
            meta_key = filename.lower()
            meta = metadata.get(meta_key) or metadata.get(os.path.splitext(meta_key)[0]) or {}
            filename_fields = parse_filename_fields(path)

            emotion_raw = resolve_field(meta, filename_fields, "emotion", "")
            emotion = canonicalize_emotion(emotion_raw)
            if not emotion or emotion == UNKNOWN_LABEL:
                skipped_missing_emotion.append(filename)
                continue
            if emotion not in ALLOWED_EMOTIONS:
                skipped_emotion.append((filename, emotion_raw))
                continue

            sex = resolve_field(meta, filename_fields, "sex", UNKNOWN_LABEL)
            ethnicity = resolve_field(meta, filename_fields, "ethnicity", UNKNOWN_LABEL)
            angle = resolve_field(meta, filename_fields, "angle", UNKNOWN_LABEL)
            if ALLOWED_ANGLES and angle not in ALLOWED_ANGLES:
                skipped_angle.append((filename, angle))
                continue
            face_type = resolve_face_type(meta, source) or UNKNOWN_LABEL
            stimulus_type = resolve_stimulus_type(face_type, source)
            stimulus_id = make_stimulus_id(filename)

            emotions.add(emotion)
            images.append(
                ImageData(
                    path,
                    source,
                    emotion,
                    sex=sex,
                    ethnicity=ethnicity,
                    angle=angle,
                    face_type=face_type,
                    stimulus_type=stimulus_type,
                    stimulus_id=stimulus_id,
                )
            )
    
    if skipped_missing_emotion:
        print(f"[DEBUG] Skipped {len(skipped_missing_emotion)} images without emotion label.")
    if skipped_angle:
        print(
            f"[DEBUG] Filtered out {len(skipped_angle)} images due to angle "
            f"(allowed={sorted(ALLOWED_ANGLES)})."
        )
    if skipped_emotion:
        print(
            f"[DEBUG] Filtered out {len(skipped_emotion)} images due to emotion "
            f"(allowed={EMOTION_CHOICES_ORDER})."
        )
    return images, emotions

def crop_face(image_path, target_size=512):
    if not os.path.exists(image_path): return None
    img = cv2.imread(image_path)
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cropped = img
    
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            padding = int(0.3 * w)
            x, y = max(0, x - padding), max(0, y - padding)
            w, h = min(img.shape[1] - x, w + 2 * padding), min(img.shape[0] - y, h + 2 * padding)
            cropped = img[y:y+h, x:x+w]

    h, w, _ = cropped.shape
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    resized_img = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# --- Backend Logic ---

def initialize_experiment(request: gr.Request):
    os.makedirs(AI_FOLDER, exist_ok=True)
    os.makedirs(HUMAN_FOLDER, exist_ok=True)
    images, emotions = scan_images()
    
    if not images:
        return None, "Error: No images found.", gr.update(interactive=False)

    session_id = str(uuid.uuid4())
    participant_id = get_participant_id(request)
    if not participant_id:
        participant_id = f"anon-{session_id}"
        msg = f"Participant ID: {participant_id}"
    else:
        msg = f"Participant ID: {participant_id}"

    csv_file_part1, csv_file_part2, csv_status = ensure_csv_files()

    # Optionally select a balanced subset across (stimulus_type, emotion).
    selected_images = images
    if BALANCE_SUBSET_DEFAULT:
        balanced = select_balanced_subset(images, MAX_PER_STRATUM)
        if balanced:
            selected_images = balanced

    available_emotions = {img.emotion for img in selected_images}
    missing_emotions = [e for e in EMOTION_CHOICES_ORDER if e not in available_emotions]
    if missing_emotions:
        print(f"[DEBUG] No stimuli found for emotions: {missing_emotions}")
    random.shuffle(selected_images)
    initial_state = {
        "participant_id": participant_id,
        "session_id": session_id,
        "csv_file": csv_file_part1,
        "csv_file_part1": csv_file_part1,
        "csv_file_part2": csv_file_part2,
        "all_images": selected_images,
        "part2_images": [],
        # Fixed order across participants.
        "emotions": list(EMOTION_CHOICES_ORDER),
        "current_index": -1,
        "current_choices": [],
        "randomize_emotions": RANDOMIZE_EMOTION_ORDER_DEFAULT,
        "start_time": None,
        "phase": "emotion",
        "part2_started": False,
        "part2_index": -1,
        "part2_start_time": None,
        "part2_touched": {k: False for k in PART2_KEYS},
    }
    
    if request:
        val = request.query_params.get(RANDOMIZE_EMOTION_ORDER_PARAM)
        if val and val.lower() in ['0','false','no']:
            initial_state["randomize_emotions"] = False

    part2_images = list(selected_images)
    random.shuffle(part2_images)
    initial_state["part2_images"] = part2_images

    return initial_state, f"{msg}\n{csv_status}", gr.update(interactive=True)

def start_interface(state):
    if not state: 
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )

def show_next_image(state):
    # Returns: [state, img_anim_update, progress_text, choices_update, next_btn_update]
    if not state:
        return (
            state,
            gr.update(visible=False, interactive=False),
            "Error",
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False),
        )

    state["current_index"] += 1
    index = state["current_index"]

    if index >= len(state["all_images"]):
        state["part2_started"] = False
        state["part2_index"] = -1
        state["part2_start_time"] = None
        state["part2_touched"] = {k: False for k in PART2_KEYS}
        state["phase"] = "part2_instructions"
        return (
            state,
            gr.update(visible=False),
            "",
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False),
        )

    image_data = state["all_images"][index]
    cropped_image = crop_face(image_data.path)
    
    if cropped_image is None:
        # Recursive skip if image fails to load
        return show_next_image(state)

    state["start_time"] = time.monotonic()
    
    # Keep emotion order fixed across all trials and participants.
    choices = list(state["emotions"])
    state["current_choices"] = choices
    choices_with_placeholder = [CHOICE_PLACEHOLDER] + choices

    return (
        state,
        gr.update(value=cropped_image, visible=True, interactive=False),
        f"Image {index + 1} of {len(state['all_images'])}",
        gr.update(choices=choices_with_placeholder, value=CHOICE_PLACEHOLDER, visible=True, interactive=True),
        gr.update(interactive=False, visible=True),
    )

def update_sections_for_phase(state):
    if not state:
        return gr.update(), gr.update(), gr.update()
    phase = state.get("phase")
    if phase == "emotion":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    if phase == "part2_instructions":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    if phase in {"part2", "complete"}:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    return gr.update(), gr.update(), gr.update()

def start_part2(state):
    if not state or state.get("phase") != "part2_instructions":
        return state
    state["phase"] = "part2"
    state["part2_started"] = False
    state["part2_index"] = -1
    state["part2_start_time"] = None
    state["part2_touched"] = {k: False for k in PART2_KEYS}
    return state

def on_emotion_select(state, selected_emotion):
    # Returns: [state, image_update, choices_interactive, next_btn_interactive]
    if not state or not selected_emotion or normalize_label(selected_emotion) == normalize_label(CHOICE_PLACEHOLDER):
        # Do nothing if placeholder selected
        return state, gr.update(), gr.update(), gr.update()
    
    try:
        start_time = state.get("start_time") or time.monotonic()
        response_time_ms = int(round((time.monotonic() - start_time) * 1000))
        image_data = state["all_images"][state["current_index"]]
        normalized_sel = canonicalize_emotion(selected_emotion)
        accuracy = "correct" if normalized_sel == image_data.emotion else "incorrect"
        trial_index = state.get("current_index", -1) + 1

        row = build_row_template(
            state,
            image_data,
        )
        row["selected_emotion"] = normalized_sel
        row["accuracy"] = accuracy

        row["emotion_trial_index"] = trial_index
        row["emotion_rt_ms"] = response_time_ms
        row["emotion_timestamp"] = datetime.now().isoformat()

        part1_file = state.get("csv_file_part1") or state.get("csv_file")
        with open(part1_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row[h] for h in PART1_HEADERS])
        print(f"[DEBUG] Saved Part 1 rating ({normalized_sel}, {response_time_ms}ms) -> {part1_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Hide Animated, Show Static (Snap), Disable Dropdown, Enable Next
    return (
        state,
        gr.update(visible=True, interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
    )

# --- Part 2 Helpers ---

def _to_int(value):
    if value is None or value == "":
        return ""
    try:
        return int(value)
    except Exception:
        return ""

# --- Part 2: Face Rating Logic ---

def start_part2_phase(state):
    # Returns: [state, main_section, part2_section]
    if not state or state.get("phase") != "part2" or state.get("part2_started"):
        return state, gr.update(), gr.update()
    state["part2_started"] = True
    state["part2_index"] = -1
    state["part2_start_time"] = None
    state["part2_touched"] = {k: False for k in PART2_KEYS}
    return state, gr.update(visible=False), gr.update(visible=True)

def _no_part2_updates(state):
    # Returns: [state, part2_image, part2_progress_text, part2_status_text, part2_completion_text,
    #           part2_age_radio, part2_masc_radio, part2_attr_radio, part2_quality_radio, part2_artifact_radio,
    #           part2_next_btn]
    return (
        state,
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )

def _part2_reset_updates():
    return (
        gr.update(value=None, interactive=True),  # part2_age_radio
        gr.update(value=None, interactive=True),  # part2_masc_radio
        gr.update(value=None, interactive=True),  # part2_attr_radio
        gr.update(value=None, interactive=True),  # part2_quality_radio
        gr.update(value=None, interactive=True),  # part2_artifact_radio
    )

def show_next_part2_image(state):
    if not state or state.get("phase") != "part2" or not state.get("part2_started"):
        return _no_part2_updates(state)

    images = state.get("part2_images") or state.get("all_images") or []
    state["part2_index"] = state.get("part2_index", -1) + 1
    index = state["part2_index"]

    if index >= len(images):
        state["phase"] = "complete"
        completion_md = "# ✅\n## Complete!"
        return (
            state,
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value=completion_md, visible=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    image_data = images[index]
    cropped_image = crop_face(image_data.path)
    if cropped_image is None:
        return show_next_part2_image(state)

    state["part2_start_time"] = time.monotonic()
    state["part2_touched"] = {k: False for k in PART2_KEYS}
    reset_updates = _part2_reset_updates()

    return (
        state,
        gr.update(value=cropped_image, visible=True),
        gr.update(value=f"Image {index + 1} of {len(images)}", visible=True),
        gr.update(value="Rate all five items to enable Next.", visible=True),
        gr.update(value="", visible=False),
        reset_updates[0],
        reset_updates[1],
        reset_updates[2],
        reset_updates[3],
        reset_updates[4],
        gr.update(interactive=False),
    )

def _mark_part2_touched(state, _value, key):
    if not state or state.get("phase") != "part2" or not state.get("part2_started"):
        return state, gr.update(), gr.update(), gr.update()
    touched = dict(state.get("part2_touched") or {})
    touched[key] = _value not in (None, "")
    state["part2_touched"] = touched
    ready = all(touched.get(k, False) for k in PART2_KEYS)
    message = "All items answered. Click Next." if ready else "Rate all five items to continue."
    return state, gr.update(interactive=ready), gr.update(message), gr.update()

def advance_part2(state, age_rating, masc_rating, attr_rating, quality_rating, artifact_rating):
    if not state or state.get("phase") != "part2" or not state.get("part2_started"):
        return _no_part2_updates(state)

    values = {
        "age": age_rating,
        "masc": masc_rating,
        "attr": attr_rating,
        "quality": quality_rating,
        "artifact": artifact_rating,
    }
    missing = [k for k, v in values.items() if v in (None, "")]
    if missing:
        state["part2_touched"] = {k: (values[k] not in (None, "")) for k in PART2_KEYS}
        return (
            state,
            gr.update(),
            gr.update(),
            gr.update("Please answer all five items before continuing."),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
        )

    images = state.get("part2_images") or state.get("all_images") or []
    index = state.get("part2_index", -1)
    if index < 0 or index >= len(images):
        return (
            state,
            gr.update(),
            gr.update(),
            gr.update("No rating target available."),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
        )

    start_time = state.get("part2_start_time") or time.monotonic()
    response_time_ms = int(round((time.monotonic() - start_time) * 1000))
    image_data = images[index]
    trial_index = index + 1

    try:
        row = build_row_template(
            state,
            image_data,
        )
        row["match_age_rating"] = _to_int(age_rating)
        row["match_masc_rating"] = _to_int(masc_rating)
        row["match_attr_rating"] = _to_int(attr_rating)
        row["match_quality_rating"] = _to_int(quality_rating)
        row["match_artifact_rating"] = _to_int(artifact_rating)

        row["matching_trial_index"] = trial_index
        row["matching_timestamp"] = datetime.now().isoformat()

        part2_file = state.get("csv_file_part2") or state.get("csv_file")
        with open(part2_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row[h] for h in PART2_HEADERS])
        print(f"[DEBUG] Saved Part 2 ratings ({response_time_ms}ms) -> {part2_file}")
    except Exception as e:
        print(f"Error saving Part 2 CSV: {e}")
        return (
            state,
            gr.update(),
            gr.update(),
            gr.update(f"Error saving ratings: {e}"),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(interactive=True),
        )

    return show_next_part2_image(state)

# --- JAVASCRIPT ---
# Logic: Find the animated image element, reset its class to remove 'image-clear',
# force a reflow, then add 'image-clear' to start the transition.
js_functions = """
() => {
    window.preBlur = function() {
        const el = document.querySelector("#img_anim img");
        if (!el) return;
        // Immediately remove any clear/snap state and force a blurred render.
        el.classList.remove("image-clear");
        el.classList.remove("image-snap");
        el.classList.add("image-preblur");
        el.style.transition = "none";
        el.style.filter = "blur(50px)";
        void el.offsetWidth;
    };

    window.triggerDeblur = function() {
        const el = document.querySelector("#img_anim img");
        if (el) {
            // Ensure we start from a blurred state, then animate to clear.
            window.preBlur();
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    el.style.transition = "";
                    el.style.filter = "";
                    el.classList.remove("image-preblur");
                    el.classList.add("image-clear");
                });
            });
        }
    };

    window.snapClear = function() {
        const el = document.querySelector("#img_anim img");
        if (el) {
            el.classList.remove('image-clear');
            el.classList.add('image-snap');
        }
    };
}
"""

# --- Gradio App ---
with gr.Blocks(theme=gr.themes.Soft(), css=APP_CSS) as app:
    state = gr.State()
    gr.Markdown("Face Emotion Recognition Study", elem_id="app_title")
    
    # 1. Landing Page
    with gr.Column(visible=True) as instructions_section:
        gr.Markdown(f"# Instructions\n ## Identify the emotion as the image becomes clear ({DEBLUR_DURATION_S}s).")
        start_btn = gr.Button("START STUDY", variant="primary", elem_id="start_btn")
        status_text = gr.Markdown("")

    # 2. Main Experiment Interface
    with gr.Column(visible=False) as main_section:
        # Image Stack: Two images occupy the same conceptual space
        with gr.Group():
            # Animated Image: Visible initially, performs blur->clear
            image_anim = gr.Image(label="", elem_id="img_anim", height=400, width=400, interactive=False, show_label=False, visible=True)
        
        progress_text = gr.Markdown("", elem_id="progress_text")
        
        # Controls
        emotion_choice = gr.Radio(choices=[], label="Select the emotion", visible=False, interactive=True, elem_id="emotion_choice")
        next_image_btn = gr.Button("Next Image ▶", variant="secondary", visible=True, interactive=False, elem_id="next_btn")

    # 3. Part 2 Instructions
    with gr.Column(visible=False, elem_id="part2_instructions_section") as part2_instructions_section:
        gr.Markdown(
            "# Part 2\n"
            "## You will now rate each face on several dimensions.\n"
            "## Use the 1–7 scale for each item, then click Next Face ▶."
        )
        part2_start_btn = gr.Button("Start Part 2 ▶", variant="primary", elem_id="part2_start_btn")

    # 4. Part 2: Rate The Images
    with gr.Column(visible=False, elem_id="part2_section") as part2_section:
        gr.Markdown(
            "# Rate The Images\n"
            "## Use the 1–7 scale for each item."
        )
        with gr.Row():
            with gr.Column(scale=1):
                part2_image = gr.Image(
                    label="",
                    height=400,
                    width=400,
                    interactive=False,
                    show_label=False,
                    visible=False,
                )
                part2_progress_text = gr.Markdown("", visible=False)
                part2_status_text = gr.Markdown("", visible=False)
                part2_completion_text = gr.Markdown("", elem_id="part2_completion_text", visible=False)
            with gr.Column(scale=1):
                part2_age_radio = gr.Radio(
                    choices=SCALE_CHOICES,
                    value=None,
                    label="Perceived age (1 = very young, 7 = very old)",
                )
                part2_masc_radio = gr.Radio(
                    choices=SCALE_CHOICES,
                    value=None,
                    label="Femininity–masculinity (1 = very feminine, 7 = very masculine)",
                )
                part2_attr_radio = gr.Radio(
                    choices=SCALE_CHOICES,
                    value=None,
                    label="Attractiveness (1 = not at all, 7 = very attractive)",
                )
                part2_quality_radio = gr.Radio(
                    choices=SCALE_CHOICES,
                    value=None,
                    label="Image quality (1 = very poor, 7 = excellent)",
                )
                part2_artifact_radio = gr.Radio(
                    choices=SCALE_CHOICES,
                    value=None,
                    label="Artifacts / oddness (1 = none, 7 = a lot)",
                )

                part2_next_btn = gr.Button("Next Face ▶", variant="primary", interactive=False)

    # --- Event Wiring ---

    # App Load
    app.load(fn=initialize_experiment, outputs=[state, status_text, start_btn]).then(fn=None, js=js_functions)

    # Start Button -> Show Interface -> Load First Image -> Trigger Animation
    start_btn.click(
        fn=start_interface,
        inputs=[state],
        outputs=[instructions_section, start_btn, main_section, part2_instructions_section, part2_section],
        show_progress="hidden",
        js="() => window.preBlur && window.preBlur()",
    ).then(
        fn=show_next_image,
        inputs=[state],
        outputs=[state, image_anim, progress_text, emotion_choice, next_image_btn],
    ).then(
        fn=update_sections_for_phase,
        inputs=[state],
        outputs=[
            main_section,
            part2_instructions_section,
            part2_section,
        ],
    ).then(
        fn=None, js="() => window.triggerDeblur()"
    )

    # Emotion Selected -> Swap Images (Snap to Clear) -> Save Data
    emotion_choice.change(
        fn=on_emotion_select, 
        inputs=[state, emotion_choice], 
        outputs=[state, image_anim, emotion_choice, next_image_btn],
        show_progress="hidden",
    ).then(
        fn=None, js="() => window.snapClear()"
    )

    # Next Button -> Load New Image -> Reset Layout -> Trigger Animation
    next_image_btn.click(
        fn=show_next_image,
        inputs=[state],
        outputs=[state, image_anim, progress_text, emotion_choice, next_image_btn],
        show_progress="hidden",
        js="() => window.preBlur && window.preBlur()",
    ).then(
        fn=update_sections_for_phase,
        inputs=[state],
        outputs=[
            main_section,
            part2_instructions_section,
            part2_section,
        ],
    ).then(
        fn=None, js="() => window.triggerDeblur()"
    )

    # Part 2 Start -> Show ratings block -> Load first rating image
    part2_start_btn.click(
        fn=start_part2,
        inputs=[state],
        outputs=[state],
        show_progress="hidden",
    ).then(
        fn=update_sections_for_phase,
        inputs=[state],
        outputs=[
            main_section,
            part2_instructions_section,
            part2_section,
        ],
    ).then(
        fn=start_part2_phase,
        inputs=[state],
        outputs=[
            state,
            main_section,
            part2_section,
        ],
    ).then(
        fn=show_next_part2_image,
        inputs=[state],
        outputs=[
            state,
            part2_image,
            part2_progress_text,
            part2_status_text,
            part2_completion_text,
            part2_age_radio,
            part2_masc_radio,
            part2_attr_radio,
            part2_quality_radio,
            part2_artifact_radio,
            part2_next_btn,
        ],
    )

    # Part 2 gating: require interaction with all five ratings
    part2_age_radio.change(
        fn=partial(_mark_part2_touched, key="age"),
        inputs=[state, part2_age_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
    )
    part2_masc_radio.change(
        fn=partial(_mark_part2_touched, key="masc"),
        inputs=[state, part2_masc_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
    )
    part2_attr_radio.change(
        fn=partial(_mark_part2_touched, key="attr"),
        inputs=[state, part2_attr_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
    )
    part2_quality_radio.change(
        fn=partial(_mark_part2_touched, key="quality"),
        inputs=[state, part2_quality_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
    )
    part2_artifact_radio.change(
        fn=partial(_mark_part2_touched, key="artifact"),
        inputs=[state, part2_artifact_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
    )

    # Part 2 Next -> Save and advance
    part2_next_btn.click(
        fn=advance_part2,
        inputs=[state, part2_age_radio, part2_masc_radio, part2_attr_radio, part2_quality_radio, part2_artifact_radio],
        outputs=[
            state,
            part2_image,
            part2_progress_text,
            part2_status_text,
            part2_completion_text,
            part2_age_radio,
            part2_masc_radio,
            part2_attr_radio,
            part2_quality_radio,
            part2_artifact_radio,
            part2_next_btn,
        ],
        show_progress="hidden",
    )

if __name__ == "__main__":
    app.launch()
