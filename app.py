import gradio as gr
import cv2
import numpy as np
import os
import random
import time
import csv
import uuid
from datetime import datetime

# --- Configuration ---
AI_FOLDER = "./AI"
HUMAN_FOLDER = "./Human"
CSV_FILE = "emotion_responses.csv"
METADATA_FILE = "stimuli_metadata.csv"
DEBLUR_DURATION_S = 5  # Seconds to go from Blur -> Clear

# --- Advanced Features Config ---
URL_PARAM_PARTICIPANT_ID = "pid"
RANDOMIZE_EMOTION_ORDER_DEFAULT = True
RANDOMIZE_EMOTION_ORDER_PARAM = "randomize"
CHOICE_PLACEHOLDER = "Select an emotion..."

# --- CSS STYLES ---
APP_CSS = f"""
#emotion_choice, #emotion_choice .wrap {{ max-height: 260px; overflow-y: auto; }}
#next_btn {{ margin: 8px 0 12px 0; }}

@media (max-width: 640px) {{
  #img_anim img, #img_static img {{ max-height: 280px; object-fit: contain; }}
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

/* --- STATIC IMAGE (The Result) --- */
/* No special CSS needed. It will just be a normal, clear image. 
   We ensure it aligns perfectly with the animated one. */
#img_static img {{
    display: block;
    filter: blur(0px);
}}
"""

# --- Constants & Mappings ---
UNKNOWN_LABEL = "unknown"
UNKNOWN_CODE = 0
FILENAME_FIELD_ORDER = ["emotion"]

EMOTION_CODE_MAP = {"happy": 1, "sad": 2, "angry": 3, "surprised": 4, "disgusted": 5, "fearful": 6, "neutral": 7, "unknown": 0}
SEX_CODE_MAP = {"male": 1, "female": 2, "other": 3, "unknown": 0}
ETHNICITY_CODE_MAP = {"caucasian": 1, "black": 2, "asian": 3, "latino": 4, "middle-eastern": 5, "indigenous": 6, "other": 7, "unknown": 0}
ANGLE_CODE_MAP = {"forward": 1, "front-left": 2, "front-right": 3, "left": 4, "right": 5, "up": 6, "down": 7, "unknown": 0}
TYPE_CODE_MAP = {"human": 1, "ai": 2, "unknown": 0}

CSV_HEADERS = [
    "participant_id", "session_id", "image_name", "image_source", "face_type", "face_type_code",
    "correct_emotion", "correct_emotion_code", "face_sex", "face_sex_code", "face_ethnicity", "face_ethnicity_code",
    "face_angle", "face_angle_code", "selected_emotion", "selected_emotion_code", "accuracy",
    "response_time_ms", "button_order", "timestamp",
]

# --- Data Structure ---
class ImageData:
    def __init__(self, path, source, emotion, sex=UNKNOWN_LABEL, ethnicity=UNKNOWN_LABEL, angle=UNKNOWN_LABEL, face_type=UNKNOWN_LABEL):
        self.path = path
        self.source = source
        self.emotion = emotion
        self.sex = sex
        self.ethnicity = ethnicity
        self.angle = angle
        self.face_type = face_type
        self.name = os.path.basename(path)

# --- Helper Functions ---
def normalize_label(value):
    if value is None: return ""
    return str(value).strip().lower().replace(" ", "-")

def get_code(code_map, label):
    return code_map.get(normalize_label(label), UNKNOWN_CODE)

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

def ensure_csv_file():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        return CSV_FILE, ""
    
    with open(CSV_FILE, newline='') as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)
    if existing_header != CSV_HEADERS:
        base, ext = os.path.splitext(CSV_FILE)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file = f"{base}_{timestamp}{ext or '.csv'}"
        with open(new_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        return new_file, f"Using new results file: {new_file}"
    return CSV_FILE, ""

def get_participant_id(request):
    if request is None: return ""
    pid = request.query_params.get(URL_PARAM_PARTICIPANT_ID)
    return str(pid).strip() if pid else ""

def scan_images():
    images = []
    emotions = set()
    metadata = load_metadata(METADATA_FILE)
    skipped = []

    for folder, source in [(AI_FOLDER, "AI"), (HUMAN_FOLDER, "Human")]:
        if not os.path.exists(folder): continue
        for filename in os.listdir(folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            path = os.path.join(folder, filename)
            meta_key = filename.lower()
            meta = metadata.get(meta_key) or metadata.get(os.path.splitext(meta_key)[0]) or {}
            filename_fields = parse_filename_fields(path)

            emotion = resolve_field(meta, filename_fields, "emotion", "")
            if not emotion or emotion == UNKNOWN_LABEL:
                skipped.append(filename)
                continue

            sex = resolve_field(meta, filename_fields, "sex", UNKNOWN_LABEL)
            ethnicity = resolve_field(meta, filename_fields, "ethnicity", UNKNOWN_LABEL)
            angle = resolve_field(meta, filename_fields, "angle", UNKNOWN_LABEL)
            face_type = resolve_face_type(meta, source) or UNKNOWN_LABEL

            emotions.add(emotion)
            images.append(ImageData(path, source, emotion, sex=sex, ethnicity=ethnicity, angle=angle, face_type=face_type))
    
    if skipped: print(f"[DEBUG] Skipped {len(skipped)} images without emotion label.")
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

    csv_file, csv_status = ensure_csv_file()
    
    random.shuffle(images)
    initial_state = {
        "participant_id": participant_id,
        "session_id": session_id,
        "csv_file": csv_file,
        "all_images": images,
        "emotions": sorted(list(emotions)),
        "current_index": -1,
        "current_choices": [],
        "randomize_emotions": RANDOMIZE_EMOTION_ORDER_DEFAULT,
        "start_time": None,
    }
    
    if request:
        val = request.query_params.get(RANDOMIZE_EMOTION_ORDER_PARAM)
        if val and val.lower() in ['0','false','no']:
            initial_state["randomize_emotions"] = False

    return initial_state, f"{msg}\n{csv_status}", gr.update(interactive=True)

def start_interface(state):
    if not state: 
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def show_next_image(state):
    # Returns: [state, img_anim, img_static, progress_text, anim_visible, static_visible, choices_update]
    if not state: 
        return state, None, None, "Error", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    state["current_index"] += 1
    index = state["current_index"]

    if index >= len(state["all_images"]):
        return state, None, None, "Experiment complete!", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    image_data = state["all_images"][index]
    cropped_image = crop_face(image_data.path)
    
    if cropped_image is None:
        # Recursive skip if image fails to load
        return show_next_image(state)

    state["start_time"] = time.monotonic()
    
    choices = list(state["emotions"])
    if state.get("randomize_emotions"):
        choices = random.sample(choices, k=len(choices))
    state["current_choices"] = choices
    choices_with_placeholder = [CHOICE_PLACEHOLDER] + choices

    return (
        state, 
        cropped_image, # For Animated Component
        cropped_image, # For Static Component
        f"Image {index + 1} of {len(state['all_images'])}", 
        gr.update(visible=True, interactive=False), # Show Animated
        gr.update(visible=False),                   # Hide Static
        gr.update(choices=choices_with_placeholder, value=CHOICE_PLACEHOLDER, visible=True, interactive=True),
    )

def on_emotion_select(state, selected_emotion):
    # Returns: [anim_visible, static_visible, choices_interactive, next_btn_interactive]
    if not state or not selected_emotion or normalize_label(selected_emotion) == normalize_label(CHOICE_PLACEHOLDER):
        # Do nothing if placeholder selected
        return gr.update(), gr.update(), gr.update(), gr.update()
    
    try:
        start_time = state.get("start_time") or time.monotonic()
        response_time_ms = int(round((time.monotonic() - start_time) * 1000))
        image_data = state["all_images"][state["current_index"]]
        normalized_sel = normalize_label(selected_emotion)
        accuracy = "correct" if normalized_sel == image_data.emotion else "incorrect"
        
        with open(state["csv_file"], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                state["participant_id"], state["session_id"], image_data.name, image_data.source,
                image_data.face_type, get_code(TYPE_CODE_MAP, image_data.face_type),
                image_data.emotion, get_code(EMOTION_CODE_MAP, image_data.emotion),
                image_data.sex, get_code(SEX_CODE_MAP, image_data.sex),
                image_data.ethnicity, get_code(ETHNICITY_CODE_MAP, image_data.ethnicity),
                image_data.angle, get_code(ANGLE_CODE_MAP, image_data.angle),
                normalized_sel, get_code(EMOTION_CODE_MAP, normalized_sel),
                accuracy, response_time_ms, "|".join(state.get("current_choices", [])),
                datetime.now().isoformat(),
            ])
        print(f"[DEBUG] Saved {normalized_sel} ({response_time_ms}ms)")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Hide Animated, Show Static (Snap), Disable Dropdown, Enable Next
    return gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False), gr.update(interactive=True)

# --- JAVASCRIPT ---
# Logic: Find the animated image element, reset its class to remove 'image-clear',
# force a reflow, then add 'image-clear' to start the transition.
js_functions = """
() => {
    window.triggerDeblur = function() {
        const el = document.querySelector("#img_anim img");
        if (el) {
            // 1. Reset to start state (Blurred)
            el.classList.remove('image-clear');
            
            // 2. Force Browser Reflow (Crucial for restarting CSS animations)
            void el.offsetWidth; 
            
            // 3. Start Animation
            setTimeout(() => {
                el.classList.add('image-clear');
            }, 100); 
        }
    };
}
"""

# --- Gradio App ---
with gr.Blocks(theme=gr.themes.Soft(), css=APP_CSS) as app:
    state = gr.State()
    gr.Markdown("# Face Emotion Recognition Study")
    
    # 1. Landing Page
    with gr.Column(visible=True) as instructions_section:
        gr.Markdown(f"## Instructions\nIdentify the emotion as the image becomes clear ({DEBLUR_DURATION_S}s).")
        start_btn = gr.Button("START STUDY", variant="primary")
        status_text = gr.Markdown("")

    # 2. Main Experiment Interface
    with gr.Column(visible=False) as main_section:
        # Image Stack: Two images occupy the same conceptual space
        with gr.Group():
            # Animated Image: Visible initially, performs blur->clear
            image_anim = gr.Image(label="", elem_id="img_anim", height=400, width=400, interactive=False, show_label=False, visible=True)
            # Static Image: Hidden initially, shows instantly when user selects answer
            image_static = gr.Image(label="", elem_id="img_static", height=400, width=400, interactive=False, show_label=False, visible=False)
        
        progress_text = gr.Markdown("")
        
        # Controls
        emotion_choice = gr.Radio(choices=[], label="Select the emotion", visible=False, interactive=True, elem_id="emotion_choice")
        next_image_btn = gr.Button("Next Image ▶", variant="secondary", visible=True, interactive=False, elem_id="next_btn")

    # --- Event Wiring ---

    # App Load
    app.load(fn=initialize_experiment, outputs=[state, status_text, start_btn]).then(fn=None, js=js_functions)

    # Start Button -> Show Interface -> Load First Image -> Trigger Animation
    start_btn.click(
        fn=start_interface, inputs=[state], outputs=[instructions_section, start_btn, main_section]
    ).then(
        fn=show_next_image, 
        inputs=[state], 
        outputs=[state, image_anim, image_static, progress_text, image_anim, image_static, emotion_choice]
    ).then(
        fn=None, js="() => window.triggerDeblur()"
    )

    # Emotion Selected -> Swap Images (Snap to Clear) -> Save Data
    emotion_choice.change(
        fn=on_emotion_select, 
        inputs=[state, emotion_choice], 
        outputs=[image_anim, image_static, emotion_choice, next_image_btn]
    )

    # Next Button -> Load New Image -> Reset Layout -> Trigger Animation
    next_image_btn.click(
        fn=show_next_image, 
        inputs=[state], 
        outputs=[state, image_anim, image_static, progress_text, image_anim, image_static, emotion_choice]
    ).then(
        fn=None, js="() => window.triggerDeblur()"
    )

if __name__ == "__main__":
    app.launch()