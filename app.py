import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import cv2
import numpy as np
import random
import time
import csv
import json
import uuid
import shutil
from datetime import datetime
from functools import partial
from urllib import parse, request as urlrequest

# --- Configuration ---
AI_FOLDER = "./AI"
HUMAN_FOLDER = "./Human"
PART1_CSV_FILE = "emotion_responses_part1.csv"
PART2_CSV_FILE = "emotion_responses_part2.csv"
METADATA_FILE = "stimuli_metadata.csv"

# --- Advanced Features Config ---
URL_PARAM_PARTICIPANT_ID = "pid"
# Keep emotion order fixed across all participants.
RANDOMIZE_EMOTION_ORDER_DEFAULT = False
RANDOMIZE_EMOTION_ORDER_PARAM = "randomize"
TURNSTILE_SITE_KEY_ENV = "TURNSTILE_SITE_KEY"
TURNSTILE_SECRET_KEY_ENV = "TURNSTILE_SECRET_KEY"
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
DOWNLOAD_PASSWORD_ENV = "CSV_DOWNLOAD_PASSWORD"

# --- Sampling Config ---
BALANCE_SUBSET_DEFAULT = True
MAX_PER_STRATUM = None  # Optionally set to an int to cap trials per (type, emotion)
ALLOWED_ANGLES = {"forward"}  # Restrict to front-facing stimuli.

# --- CSS STYLES ---
APP_CSS = f"""
#start_btn > button, 
#next_btn > button {{
  font-size: 20px !important;
  padding: 12px 22px !important;
  min-height: 48px !important;
}}
#emotion_choice {{
  max-width: 760px;
  margin: 0 auto;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}
#emotion_choice > div,
#emotion_choice fieldset,
#emotion_choice .form {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}}
#emotion_choice .wrap {{
  display: grid !important;
  grid-template-columns: repeat(2, minmax(220px, 1fr));
  gap: 18px !important;
  width: 100%;
  background: transparent !important;
}}
#emotion_choice input[type="radio"] {{
  position: absolute;
  opacity: 0;
  pointer-events: none;
}}
#emotion_choice .wrap label {{
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 100% !important;
  min-height: 120px !important;
  margin: 0 !important;
  padding: 20px !important;
  border: 2px solid #4b5563 !important;
  border-radius: 22px !important;
  background: #6b7280 !important;
  color: #ffffff !important;
  box-sizing: border-box;
  cursor: pointer;
  transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
}}
#emotion_choice .wrap label:hover {{
  transform: translateY(-1px);
  border-color: #374151 !important;
  background: #5b6472 !important;
  box-shadow: 0 8px 20px rgba(31, 41, 55, 0.22);
}}
#emotion_choice .wrap label:has(input[type="radio"]:checked) {{
  background: #374151 !important;
  border-color: #111827 !important;
  box-shadow: 0 10px 24px rgba(17, 24, 39, 0.28);
}}
#emotion_choice .wrap span {{
  display: block;
  width: 100%;
  text-align: center;
  font-size: 34px !important;
  font-weight: 700 !important;
  line-height: 1.1;
  color: #ffffff !important;
}}

@media (max-width: 640px) {{
  #img_anim img {{ max-height: 280px; object-fit: contain; }}
  #emotion_choice .wrap {{
    grid-template-columns: repeat(2, minmax(140px, 1fr));
    gap: 12px !important;
  }}
  #emotion_choice .wrap label {{
    min-height: 96px !important;
    padding: 16px !important;
  }}
  #emotion_choice .wrap span {{
    font-size: 26px !important;
  }}
  #app_title h1 {{
    font-size: 24px !important;
  }}
  #instructions_heading h1 {{
    font-size: 28px !important;
  }}
  #instructions_heading h2 {{
    font-size: 18px !important;
  }}
}}

#img_anim img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
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
  margin: 0 0 6px 0 !important;
  padding: 0 !important;
  text-align: center;
  overflow: visible !important;
}}

#app_title > div {{
  overflow: visible !important;
}}

#app_title h1 {{
  margin: 0 !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  line-height: 1.1 !important;
}}

#instructions_section {{
  max-width: 760px;
  margin: 0 auto;
}}

#instructions_heading {{
  text-align: center;
}}

#instructions_heading h1 {{
  font-size: 40px !important;
  margin: 0 0 8px !important;
}}

#instructions_heading h2 {{
  font-size: 24px !important;
  margin: 0 !important;
}}

#human_check_wrap {{
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}}

#human_check_wrap .cf-turnstile {{
  margin: 0 auto;
}}

#download_sidebar {{
  border-left: 1px solid #d1d5db;
}}

#download_sidebar h2 {{
  margin-top: 0 !important;
}}

#unlock_downloads_btn > button {{
  width: 100%;
}}

#download_status {{
  min-height: 24px;
}}

"""

# --- Constants & Mappings ---
UNKNOWN_LABEL = "unknown"
FILENAME_FIELD_ORDER = ["emotion"]

# Fixed emotion set and order for all trials.
EMOTION_CHOICES = [
    ("Happy", "happy"),
    ("Sad", "sad"),
    ("Angry", "angry"),
    ("Fear", "fear"),
]
EMOTION_CHOICES_ORDER = [value for _, value in EMOTION_CHOICES]
ALLOWED_EMOTIONS = set(EMOTION_CHOICES_ORDER)
EMOTION_ALIASES = {
    "afraid": "fear",
    "fearful": "fear",
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

VERIFIED_SESSION_IDS = set()

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

def turnstile_site_key():
    return os.getenv(TURNSTILE_SITE_KEY_ENV, "").strip()

def turnstile_secret_key():
    return os.getenv(TURNSTILE_SECRET_KEY_ENV, "").strip()

def turnstile_is_enabled():
    return bool(turnstile_site_key() and turnstile_secret_key())

def turnstile_is_partially_configured():
    return bool(turnstile_site_key()) ^ bool(turnstile_secret_key())

def render_turnstile_widget():
    if not turnstile_is_enabled():
        return ""
    return """
<div id="human_check_wrap">
  <div id="turnstile_widget"></div>
</div>
"""

TURNSTILE_HEAD = """
<script>
(() => {
  const siteKey = %s;
  let turnstileWidgetId = null;
  let renderedNode = null;
  let observer = null;
  let renderScheduled = false;

  function getTurnstileInput() {
    return document.querySelector("#turnstile_token textarea, #turnstile_token input");
  }

  function getTurnstileMount() {
    return document.querySelector("#turnstile_widget");
  }

  function dispatchTurnstileValue(value) {
    const input = getTurnstileInput();
    if (!input) return;
    input.value = value || "";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  window.onTurnstileSuccess = function(token) {
    dispatchTurnstileValue(token);
  };

  window.onTurnstileExpired = function() {
    dispatchTurnstileValue("");
    if (window.turnstile && turnstileWidgetId !== null) {
      try {
        window.turnstile.reset(turnstileWidgetId);
        return;
      } catch (_error) {
      }
    }
    window.renderTurnstileWidget();
  };

  window.onTurnstileError = function(errorCode) {
    console.error("Turnstile error:", errorCode);
    dispatchTurnstileValue("");
    return false;
  };

  function clearWidgetReference() {
    turnstileWidgetId = null;
    renderedNode = null;
  }

  function renderTurnstileWidget() {
    const mount = getTurnstileMount();
    if (!mount || !siteKey) return false;
    if (!window.turnstile || typeof window.turnstile.render !== "function") return false;

    if (renderedNode && renderedNode !== mount) {
      try {
        if (turnstileWidgetId !== null) {
          window.turnstile.remove(turnstileWidgetId);
        }
      } catch (_error) {
      }
      clearWidgetReference();
    }

    if (renderedNode === mount && turnstileWidgetId !== null) {
      return true;
    }

    dispatchTurnstileValue("");
    mount.replaceChildren();

    try {
      turnstileWidgetId = window.turnstile.render(mount, {
        sitekey: siteKey,
        callback: window.onTurnstileSuccess,
        "expired-callback": window.onTurnstileExpired,
        "error-callback": window.onTurnstileError,
      });
      renderedNode = mount;
      return true;
    } catch (error) {
      console.error("Turnstile render failed:", error);
      clearWidgetReference();
      return false;
    }
  }

  function scheduleTurnstileRender() {
    if (renderScheduled) return;
    renderScheduled = true;
    window.requestAnimationFrame(() => {
      renderScheduled = false;
      renderTurnstileWidget();
    });
  }

  window.onTurnstileApiLoad = function() {
    scheduleTurnstileRender();
  };

  window.renderTurnstileWidget = scheduleTurnstileRender;

  window.resetTurnstileWidget = function() {
    dispatchTurnstileValue("");
    if (window.turnstile && turnstileWidgetId !== null) {
      try {
        window.turnstile.reset(turnstileWidgetId);
        return;
      } catch (_error) {
      }
    }
    clearWidgetReference();
    scheduleTurnstileRender();
  };

  function startTurnstileObserver() {
    if (observer) return;
    const root = document.documentElement || document.body;
    if (!root) return;
    observer = new MutationObserver(() => {
      scheduleTurnstileRender();
    });
    observer.observe(root, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      startTurnstileObserver();
      scheduleTurnstileRender();
    }, { once: true });
  } else {
    startTurnstileObserver();
    scheduleTurnstileRender();
  }

  window.addEventListener("load", scheduleTurnstileRender);
})();
</script>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit&onload=onTurnstileApiLoad" defer></script>
""" % json.dumps(turnstile_site_key()) if turnstile_site_key() else None

APP_THEME = gr.themes.Soft()

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

def get_downloadable_csv_files():
    return [path for path in [PART1_CSV_FILE, PART2_CSV_FILE, METADATA_FILE] if os.path.exists(path)]

def unlock_downloads(password):
    expected = str(os.getenv(DOWNLOAD_PASSWORD_ENV) or "").strip()
    if not expected:
        return (
            gr.update(value=f"Download access is not configured. Set `{DOWNLOAD_PASSWORD_ENV}` on the server.", visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=""),
        )

    if str(password or "") != expected:
        return (
            gr.update(value="Incorrect password.", visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=""),
        )

    files = get_downloadable_csv_files()
    if not files:
        return (
            gr.update(value="No CSV files are available yet.", visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=""),
        )

    return (
        gr.update(value=f"Downloads unlocked. {len(files)} file(s) available.", visible=True),
        gr.update(value=files, visible=True),
        gr.update(value=""),
    )

def get_participant_id(request):
    if request is None: return ""
    pid = request.query_params.get(URL_PARAM_PARTICIPANT_ID)
    return str(pid).strip() if pid else ""

def get_request_ip(request):
    if request is None:
        return ""
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return str(host).strip() if host else ""

def turnstile_status_text():
    if turnstile_is_enabled():
        return "Complete the human check to enable Start."
    if turnstile_is_partially_configured():
        return (
            f"Turnstile is misconfigured. Set both `{TURNSTILE_SITE_KEY_ENV}` "
            f"and `{TURNSTILE_SECRET_KEY_ENV}` to enable bot protection."
        )
    return (
        f"Bot protection is currently off. Add `{TURNSTILE_SITE_KEY_ENV}` "
        f"and `{TURNSTILE_SECRET_KEY_ENV}` to enable Cloudflare Turnstile."
    )

def verify_turnstile_token(token, request):
    if not turnstile_is_enabled():
        return True, ""

    token = str(token or "").strip()
    if not token:
        return False, "Please complete the human check before starting."

    payload = {
        "secret": turnstile_secret_key(),
        "response": token,
    }
    remote_ip = get_request_ip(request)
    if remote_ip:
        payload["remoteip"] = remote_ip

    verify_request = urlrequest.Request(
        TURNSTILE_VERIFY_URL,
        data=parse.urlencode(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urlrequest.urlopen(verify_request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"Turnstile verification request failed: {exc}")
        return False, "Could not verify the human check. Please try again."

    if body.get("success"):
        return True, ""

    error_codes = body.get("error-codes") or []
    print(f"Turnstile verification failed: {error_codes}")
    return False, "Human check expired or failed. Please try again."

def is_verified_session(state):
    if not turnstile_is_enabled():
        return True
    if not state:
        return False
    session_id = str(state.get("session_id") or "").strip()
    return bool(session_id) and session_id in VERIFIED_SESSION_IDS and bool(state.get("human_verified"))

def _blocked_start_response(state, message, start_interactive=False):
    return (
        state,
        gr.update(visible=True),
        gr.update(visible=True, interactive=start_interactive),
        gr.update(visible=False),
        gr.update(),
        gr.update(value=message),
        gr.update(visible=False, interactive=False),
        gr.update(value=""),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False, interactive=False),
    )

def _blocked_main_response(state, message="Complete the human check to continue."):
    return (
        state,
        gr.update(visible=False, interactive=False),
        message,
        gr.update(visible=False, interactive=False),
        gr.update(visible=False, interactive=False),
    )

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
        return (
            None,
            "Error: No images found.",
            gr.update(interactive=False),
            gr.update(value=turnstile_status_text()),
        )

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
        "human_verified": not turnstile_is_enabled(),
        "csv_file": csv_file_part1,
        "csv_file_part1": csv_file_part1,
        "csv_file_part2": csv_file_part2,
        "all_images": selected_images,
        "part2_images": [],
        # Fixed order across participants.
        "emotions": list(EMOTION_CHOICES),
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

    if not turnstile_is_enabled():
        VERIFIED_SESSION_IDS.add(session_id)

    start_enabled = bool(images) and not turnstile_is_enabled() and not turnstile_is_partially_configured()
    return (
        initial_state,
        f"{msg}\n{csv_status}",
        gr.update(interactive=start_enabled),
        gr.update(value=turnstile_status_text()),
    )

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

def begin_study(state, turnstile_token, request: gr.Request):
    if not state:
        return _blocked_start_response(state, "Error: study state is missing.", start_interactive=False)

    if turnstile_is_partially_configured():
        return _blocked_start_response(
            state,
            f"Turnstile is misconfigured. Set both `{TURNSTILE_SITE_KEY_ENV}` and `{TURNSTILE_SECRET_KEY_ENV}`.",
            start_interactive=False,
        )

    verified, message = verify_turnstile_token(turnstile_token, request)
    if not verified:
        return _blocked_start_response(
            state,
            message,
            start_interactive=bool(str(turnstile_token or "").strip()),
        )

    session_id = str(state.get("session_id") or "").strip()
    if session_id:
        VERIFIED_SESSION_IDS.add(session_id)
    state["human_verified"] = True

    next_state, image_update, progress_value, choice_update, next_btn_update = show_next_image(state)
    return (
        next_state,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(),
        gr.update(value="Human check passed."),
        image_update,
        gr.update(value=progress_value),
        choice_update,
        next_btn_update,
    )

def on_turnstile_token_change(token):
    if turnstile_is_partially_configured():
        return (
            gr.update(interactive=False),
            gr.update(
                value=(
                    f"Turnstile is misconfigured. Set both `{TURNSTILE_SITE_KEY_ENV}` "
                    f"and `{TURNSTILE_SECRET_KEY_ENV}`."
                )
            ),
        )
    if not turnstile_is_enabled():
        return (
            gr.update(interactive=True),
            gr.update(value=turnstile_status_text()),
        )

    has_token = bool(str(token or "").strip())
    return (
        gr.update(interactive=has_token),
        gr.update(
            value="Human check complete. Click Start." if has_token else "Complete the human check to enable Start."
        ),
    )

def show_next_image(state):
    # Returns: [state, img_anim_update, progress_text, choices_update, next_btn_update]
    if not state:
        return _blocked_main_response(state, "Error")
    if not is_verified_session(state):
        return _blocked_main_response(state)

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

    return (
        state,
        gr.update(value=cropped_image, visible=True, interactive=False),
        f"Image {index + 1} of {len(state['all_images'])}",
        gr.update(choices=choices, value=None, visible=True, interactive=True),
        gr.update(interactive=False, visible=False),
    )

def advance_main_phase(state):
    next_state, image_update, progress_update, choice_update, next_btn_update = show_next_image(state)
    phase = next_state.get("phase") if next_state else None
    show_intro = phase == "part2_instructions"
    if show_intro:
        print("[DEBUG] Transitioned from Part 1 to Part 2 instructions.")
    return (
        next_state,
        image_update,
        gr.update(value=progress_update, visible=not show_intro),
        choice_update,
        next_btn_update,
        gr.update(visible=show_intro),
        gr.update(visible=show_intro),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )

def on_emotion_select(state, selected_emotion):
    # Returns: [state, image_update, choices_interactive, next_btn_interactive]
    if not is_verified_session(state):
        return state, gr.update(), gr.update(interactive=False), gr.update(interactive=False)
    if not state or not selected_emotion:
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

    # Freeze the current response while the follow-up event advances automatically.
    return (
        state,
        gr.update(visible=True, interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False, visible=False),
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

def start_part2(state):
    if not is_verified_session(state):
        return state
    if not state or state.get("phase") != "part2_instructions":
        return state

    state["phase"] = "part2"
    state["part2_started"] = True
    state["part2_index"] = -1
    state["part2_start_time"] = None
    state["part2_touched"] = {k: False for k in PART2_KEYS}
    print("[DEBUG] Starting Part 2.")
    return state

def begin_part2(state):
    next_state = start_part2(state)
    part2_outputs = show_next_part2_image(next_state)
    phase = part2_outputs[0].get("phase") if part2_outputs[0] else None
    show_part2 = phase in {"part2", "complete"}
    return (
        part2_outputs[0],
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=show_part2),
        *part2_outputs[1:],
    )

def _no_part2_updates(state):
    # Returns: [state, part2_image, part2_progress_text, part2_status_text, part2_completion_text,
    #           part2_age_radio, part2_masc_radio, part2_attr_radio, part2_quality_radio, part2_artifact_radio,
    #           part2_next_btn]
    return (
        state,
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False, interactive=False),
    )

def _part2_reset_updates():
    return (
        gr.update(value=None, interactive=True, visible=True),  # part2_age_radio
        gr.update(value=None, interactive=True, visible=True),  # part2_masc_radio
        gr.update(value=None, interactive=True, visible=True),  # part2_attr_radio
        gr.update(value=None, interactive=True, visible=True),  # part2_quality_radio
        gr.update(value=None, interactive=True, visible=True),  # part2_artifact_radio
    )

def show_next_part2_image(state):
    if not is_verified_session(state):
        return _no_part2_updates(state)
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
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
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
        gr.update(interactive=False, visible=True),
    )

def _mark_part2_touched(state, _value, key):
    if not is_verified_session(state):
        return state, gr.update(interactive=False), gr.update("Complete the human check to continue."), gr.update()
    if not state or state.get("phase") != "part2" or not state.get("part2_started"):
        return state, gr.update(), gr.update(), gr.update()
    touched = dict(state.get("part2_touched") or {})
    touched[key] = _value not in (None, "")
    state["part2_touched"] = touched
    ready = all(touched.get(k, False) for k in PART2_KEYS)
    message = "All items answered. Click Next." if ready else "Rate all five items to continue."
    return state, gr.update(interactive=ready), gr.update(message), gr.update()

def advance_part2(state, age_rating, masc_rating, attr_rating, quality_rating, artifact_rating):
    if not is_verified_session(state):
        return _no_part2_updates(state)
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

# --- Gradio App ---
with gr.Blocks() as app:
    state = gr.State()

    with gr.Sidebar(label="Data Downloads", open=False, position="right", width=340, elem_id="download_sidebar"):
        gr.Markdown(
            "## Data Downloads\n"
            "Enter the password to unlock the CSV files."
        )
        download_password = gr.Textbox(
            label="Password",
            type="password",
            placeholder="Enter download password",
            elem_id="download_password",
        )
        unlock_downloads_btn = gr.Button("Unlock Downloads", variant="secondary", elem_id="unlock_downloads_btn")
        download_status = gr.Markdown("", visible=False, elem_id="download_status")
        download_files = gr.File(
            label="Available CSV files",
            file_count="multiple",
            interactive=False,
            visible=False,
        )
    
    # 1. Landing Page
    with gr.Column(visible=True, elem_id="instructions_section") as instructions_section:
        gr.HTML("<h1>Face Emotion Recognition Study</h1>", elem_id="app_title")
        gr.Markdown("# Instructions\n ## Identify the emotion shown in each face.", elem_id="instructions_heading")
        turnstile_token = gr.Textbox(value="", visible="hidden", elem_id="turnstile_token", render=True)
        human_check_widget = gr.HTML(render_turnstile_widget(), visible=turnstile_is_enabled())
        human_check_status = gr.Markdown("")
        start_btn = gr.Button("START STUDY", variant="primary", elem_id="start_btn", interactive=False)
        status_text = gr.Markdown("")

    # 2. Main Experiment Interface
    with gr.Column(visible=False) as main_section:
        with gr.Group():
            image_anim = gr.Image(label="", elem_id="img_anim", height=400, width=400, interactive=False, show_label=False, visible=True)

        progress_text = gr.Markdown("", elem_id="progress_text")

        # Controls
        emotion_choice = gr.Radio(choices=[], label="Select the emotion", visible=False, interactive=True, elem_id="emotion_choice")
        next_image_btn = gr.Button("Next Image ▶", variant="secondary", visible=False, interactive=False, elem_id="next_btn")

        with gr.Column(elem_id="part2_instructions_section"):
            part2_intro_text = gr.Markdown(
                "# Part 2\n"
                "## You will now rate each face on several dimensions.\n"
                "## Use the 1–7 scale for each item, then click Next Face ▶.",
                visible="hidden",
            )
            part2_start_btn = gr.Button("Start Part 2 ▶", variant="primary", elem_id="part2_start_btn", visible="hidden")

        with gr.Column(elem_id="part2_section"):
            part2_title = gr.Markdown(
                "# Rate The Images\n"
                "## Use the 1–7 scale for each item.",
                visible="hidden",
            )
            with gr.Row():
                with gr.Column(scale=1):
                    part2_image = gr.Image(
                        label="",
                        height=400,
                        width=400,
                        interactive=False,
                        show_label=False,
                        visible="hidden",
                    )
                    part2_progress_text = gr.Markdown("", visible="hidden")
                    part2_status_text = gr.Markdown("", visible="hidden")
                    part2_completion_text = gr.Markdown("", elem_id="part2_completion_text", visible="hidden")
                with gr.Column(scale=1):
                    part2_age_radio = gr.Radio(
                        choices=SCALE_CHOICES,
                        value=None,
                        label="Perceived age (1 = very young, 7 = very old)",
                        visible="hidden",
                    )
                    part2_masc_radio = gr.Radio(
                        choices=SCALE_CHOICES,
                        value=None,
                        label="Femininity–masculinity (1 = very feminine, 7 = very masculine)",
                        visible="hidden",
                    )
                    part2_attr_radio = gr.Radio(
                        choices=SCALE_CHOICES,
                        value=None,
                        label="Attractiveness (1 = not at all, 7 = very attractive)",
                        visible="hidden",
                    )
                    part2_quality_radio = gr.Radio(
                        choices=SCALE_CHOICES,
                        value=None,
                        label="Image quality (1 = very poor, 7 = excellent)",
                        visible="hidden",
                    )
                    part2_artifact_radio = gr.Radio(
                        choices=SCALE_CHOICES,
                        value=None,
                        label="This image contains visual glitches or unnatural details. (1 = strongly disagree, 7 = strongly agree)",
                        visible="hidden",
                    )

                    part2_next_btn = gr.Button("Next Face ▶", variant="primary", interactive=False, visible="hidden")

    # --- Event Wiring ---

    # App Load
    app.load(
        fn=initialize_experiment,
        outputs=[state, status_text, start_btn, human_check_status],
        api_visibility="private",
    )

    turnstile_token.change(
        fn=on_turnstile_token_change,
        inputs=[turnstile_token],
        outputs=[start_btn, human_check_status],
        show_progress="hidden",
        api_visibility="private",
    )

    unlock_downloads_btn.click(
        fn=unlock_downloads,
        inputs=[download_password],
        outputs=[download_status, download_files, download_password],
        show_progress="hidden",
        api_visibility="private",
    )

    download_password.submit(
        fn=unlock_downloads,
        inputs=[download_password],
        outputs=[download_status, download_files, download_password],
        show_progress="hidden",
        api_visibility="private",
    )

    # Start Button -> Show Interface -> Load First Image -> Trigger Animation
    start_btn.click(
        fn=begin_study,
        inputs=[state, turnstile_token],
        outputs=[
            state,
            instructions_section,
            start_btn,
            main_section,
            status_text,
            human_check_status,
            image_anim,
            progress_text,
            emotion_choice,
            next_image_btn,
        ],
        show_progress="hidden",
        api_visibility="private",
    ).then(
        fn=None,
        js="() => { if (window.resetTurnstileWidget) window.resetTurnstileWidget(); }",
    )

    # Emotion Selected -> Save Data -> Advance automatically
    emotion_choice.input(
        fn=on_emotion_select, 
        inputs=[state, emotion_choice], 
        outputs=[state, image_anim, emotion_choice, next_image_btn],
        show_progress="hidden",
        api_visibility="private",
    ).then(
        fn=advance_main_phase,
        inputs=[state],
        outputs=[
            state,
            image_anim,
            progress_text,
            emotion_choice,
            next_image_btn,
            part2_intro_text,
            part2_start_btn,
            part2_title,
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
        api_visibility="private",
    )

    # Next Button -> Load New Image -> Reset Layout -> Trigger Animation
    next_image_btn.click(
        fn=advance_main_phase,
        inputs=[state],
        outputs=[
            state,
            image_anim,
            progress_text,
            emotion_choice,
            next_image_btn,
            part2_intro_text,
            part2_start_btn,
            part2_title,
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
        api_visibility="private",
    )

    # Part 2 Start -> Show ratings block -> Load first rating image
    part2_start_btn.click(
        fn=begin_part2,
        inputs=[state],
        outputs=[
            state,
            image_anim,
            progress_text,
            emotion_choice,
            next_image_btn,
            part2_intro_text,
            part2_start_btn,
            part2_title,
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
        api_visibility="private",
    )

    # Part 2 gating: require interaction with all five ratings
    part2_age_radio.change(
        fn=partial(_mark_part2_touched, key="age"),
        inputs=[state, part2_age_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
        api_visibility="private",
    )
    part2_masc_radio.change(
        fn=partial(_mark_part2_touched, key="masc"),
        inputs=[state, part2_masc_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
        api_visibility="private",
    )
    part2_attr_radio.change(
        fn=partial(_mark_part2_touched, key="attr"),
        inputs=[state, part2_attr_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
        api_visibility="private",
    )
    part2_quality_radio.change(
        fn=partial(_mark_part2_touched, key="quality"),
        inputs=[state, part2_quality_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
        api_visibility="private",
    )
    part2_artifact_radio.change(
        fn=partial(_mark_part2_touched, key="artifact"),
        inputs=[state, part2_artifact_radio],
        outputs=[state, part2_next_btn, part2_status_text, part2_completion_text],
        show_progress="hidden",
        api_visibility="private",
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
        api_visibility="private",
    )

if __name__ == "__main__":
    # Gradio 6 performs a HEAD request against the local root URL during launch.
    # On Hugging Face Spaces this check can fail even after startup-events succeed,
    # which aborts launch before the app is served. share=True skips that check,
    # and Gradio immediately disables actual share tunnels on Spaces.
    launch_share = bool(os.getenv("SPACE_ID"))
    # Support reverse proxies that publish the app under a subpath such as /facelab.
    launch_root_path = os.getenv("GRADIO_ROOT_PATH") or None
    app.launch(
        share=launch_share,
        root_path=launch_root_path,
        theme=APP_THEME,
        css=APP_CSS,
        head=TURNSTILE_HEAD,
        footer_links=["gradio"],
    )
