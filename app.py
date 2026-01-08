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
DEBLUR_DURATION_S = 10

# Query param used in URLs like: https://.../app?pid=12345
URL_PARAM_PARTICIPANT_ID = "pid"
# Randomize emotion choice order per trial (can be overridden by URL param).
RANDOMIZE_EMOTION_ORDER_DEFAULT = True
RANDOMIZE_EMOTION_ORDER_PARAM = "randomize"

# Label normalization defaults.
UNKNOWN_LABEL = "unknown"
UNKNOWN_CODE = 0

# Filename parsing order from the RIGHT side. Extend if you encode more fields in filenames.
# Example filename if you extend: "subject_happy_female_asian_front-left.png"
FILENAME_FIELD_ORDER = ["emotion"]

# Code mappings (edit here when your coding scheme changes).
EMOTION_CODE_MAP = {
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "surprised": 4,
    "disgusted": 5,
    "fearful": 6,
    "neutral": 7,
    "unknown": 0,
}
SEX_CODE_MAP = {
    "male": 1,
    "female": 2,
    "other": 3,
    "unknown": 0,
}
ETHNICITY_CODE_MAP = {
    "caucasian": 1,
    "black": 2,
    "asian": 3,
    "latino": 4,
    "middle-eastern": 5,
    "indigenous": 6,
    "other": 7,
    "unknown": 0,
}
ANGLE_CODE_MAP = {
    "forward": 1,
    "front-left": 2,
    "front-right": 3,
    "left": 4,
    "right": 5,
    "up": 6,
    "down": 7,
    "unknown": 0,
}
TYPE_CODE_MAP = {
    "human": 1,
    "ai": 2,
    "unknown": 0,
}

CSV_HEADERS = [
    "participant_id",
    "session_id",
    "image_name",
    "image_source",
    "face_type",
    "face_type_code",
    "correct_emotion",
    "correct_emotion_code",
    "face_sex",
    "face_sex_code",
    "face_ethnicity",
    "face_ethnicity_code",
    "face_angle",
    "face_angle_code",
    "selected_emotion",
    "selected_emotion_code",
    "accuracy",
    "response_time_ms",
    "button_order",
    "timestamp",
]

# --- Data Structure ---
class ImageData:
    """A simple class to hold information about each image."""
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
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = value.replace(" ", "-")
    return value

def get_code(code_map, label):
    label = normalize_label(label)
    if not label:
        return UNKNOWN_CODE
    return code_map.get(label, UNKNOWN_CODE)

def load_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        return {}
    metadata = {}
    with open(metadata_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("image_name") or row.get("filename") or row.get("image")
            if not name:
                continue
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
    if len(parts) < 2:
        return {}
    fields = {}
    for field in FILENAME_FIELD_ORDER:
        if not parts:
            break
        fields[field] = normalize_label(parts.pop())
    return fields

def resolve_field(metadata, filename_fields, key, default=UNKNOWN_LABEL):
    value = ""
    if metadata:
        value = normalize_label(metadata.get(key))
    if not value:
        value = filename_fields.get(key, "")
    return value or default

def resolve_face_type(metadata, source):
    if metadata:
        face_type = metadata.get("face_type")
        if face_type:
            return normalize_label(face_type)
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

def parse_randomize_param(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ("0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    return None

def get_participant_id(request):
    if request is None:
        return ""
    participant_id = request.query_params.get(URL_PARAM_PARTICIPANT_ID)
    if participant_id is None:
        return ""
    return str(participant_id).strip()

def scan_images():
    images = []
    emotions = set()
    metadata = load_metadata(METADATA_FILE)
    skipped = []

    for folder, source in [(AI_FOLDER, "AI"), (HUMAN_FOLDER, "Human")]:
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
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

    if skipped:
        print(f"[DEBUG] Skipped {len(skipped)} images without an emotion label.")

    return images, emotions

# --- Backend Functions ---

def crop_face(image_path, target_size=512):
    """
    Crops the image to the largest detected face, then resizes and pads it 
    to a fixed square size. Returns original if no face is found.
    """
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print(f"ERROR: Haar Cascade file not found at {cascade_path}")
        # Still try to process the original image
        cropped = img
    else:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            # If no face is detected, use the whole image
            cropped = img
        else:
            # Get the largest face and add padding
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            padding = int(0.3 * w)
            x, y = max(0, x - padding), max(0, y - padding)
            w, h = min(img.shape[1] - x, w + 2 * padding), min(img.shape[0] - y, h + 2 * padding)
            cropped = img[y:y+h, x:x+w]

    # --- NEW RESIZING AND PADDING LOGIC ---
    # 1. Resize the image to fit within the target size while maintaining aspect ratio
    h, w, _ = cropped.shape
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    resized_img = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Create a black square canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # 3. Paste the resized image onto the center of the canvas
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    # 4. Convert to RGB for Gradio display
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def initialize_experiment(request: gr.Request):
    """Scans folders for images and prepares the experiment state."""
    os.makedirs(AI_FOLDER, exist_ok=True)
    os.makedirs(HUMAN_FOLDER, exist_ok=True)

    images, emotions = scan_images()
    if not images:
        return None, "Error: No images found. Please add images to 'AI' and 'Human' folders.", gr.update(interactive=False)

    sorted_emotions = sorted(list(emotions))
    if not sorted_emotions:
        return None, "Error: No valid emotion labels found in image names or metadata.", gr.update(interactive=False)

    session_id = str(uuid.uuid4())
    participant_id = get_participant_id(request)
    if not participant_id:
        participant_id = f"anon-{session_id}"
        participant_msg = f"Participant ID: {participant_id} (auto-generated; add ?{URL_PARAM_PARTICIPANT_ID}=... to URL)"
    else:
        participant_msg = f"Participant ID: {participant_id}"

    randomize_emotions = RANDOMIZE_EMOTION_ORDER_DEFAULT
    if request is not None:
        override = parse_randomize_param(request.query_params.get(RANDOMIZE_EMOTION_ORDER_PARAM))
        if override is not None:
            randomize_emotions = override

    csv_file, csv_status = ensure_csv_file()
    status_lines = [participant_msg]
    if csv_status:
        status_lines.append(csv_status)

    random.shuffle(images)
    initial_state = {
        "participant_id": participant_id,
        "session_id": session_id,
        "csv_file": csv_file,
        "all_images": images,
        "emotions": sorted_emotions,
        "current_index": -1,
        "current_choices": [],
        "randomize_emotions": randomize_emotions,
        "start_time": None,
    }

    return initial_state, "\n\n".join(status_lines), gr.update(interactive=True)

def start_interface(state):
    """Hides instructions and shows the main experiment UI."""
    if not state:
        return (
            gr.update(visible=True),  # instructions_section
            gr.update(visible=True),  # start_btn
            gr.update(visible=False), # main_section
        )
    return (
        gr.update(visible=False), # instructions_section
        gr.update(visible=False), # start_btn
        gr.update(visible=True),  # main_section
    )

def show_next_image(state):
    """Loads the next image and updates the state."""
    if not state:
        return (
            state,
            None,
            "No experiment state available.",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    state["current_index"] += 1
    index = state["current_index"]

    if index >= len(state["all_images"]):
        return (
            state, 
            None, 
            "Experiment complete! Thank you for participating.", 
            gr.update(visible=False),  # next_image_btn
            gr.update(visible=False),  # emotion_choice
        )

    image_data = state["all_images"][index]
    cropped_image = crop_face(image_data.path)
    
    if cropped_image is None:
        return (
            state, 
            None, 
            f"Error loading image: {image_data.name}", 
            gr.update(visible=True),   # show Next so user can skip the broken one
            gr.update(visible=False),  # emotion_choice
        )

    state["start_time"] = time.monotonic()
    print(f"[DEBUG] Showing image {index+1}/{len(state['all_images'])}: {image_data.name}")

    choices = list(state["emotions"])
    if state.get("randomize_emotions"):
        choices = random.sample(choices, k=len(choices))
    state["current_choices"] = choices

    return (
        state, 
        cropped_image, 
        f"Image {index + 1} of {len(state['all_images'])}", 
        gr.update(visible=False), # hide Next until a choice is made
        gr.update(choices=choices, value=None, visible=True, interactive=True),
    )

def on_emotion_select(state, selected_emotion):
    """Handles emotion selection and records data, then shows Next."""
    if not state or not selected_emotion:
        return gr.update(), gr.update()

    selected_emotion = normalize_label(selected_emotion)
    # Try to save; don't let errors block UI updates
    try:
        start_time = state.get("start_time") or time.monotonic()
        response_time_ms = int(round((time.monotonic() - start_time) * 1000))
        image_data = state["all_images"][state["current_index"]]
        accuracy = "correct" if selected_emotion == image_data.emotion else "incorrect"
        with open(state["csv_file"], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                state["participant_id"],
                state["session_id"],
                image_data.name,
                image_data.source,
                image_data.face_type,
                get_code(TYPE_CODE_MAP, image_data.face_type),
                image_data.emotion,
                get_code(EMOTION_CODE_MAP, image_data.emotion),
                image_data.sex,
                get_code(SEX_CODE_MAP, image_data.sex),
                image_data.ethnicity,
                get_code(ETHNICITY_CODE_MAP, image_data.ethnicity),
                image_data.angle,
                get_code(ANGLE_CODE_MAP, image_data.angle),
                selected_emotion,
                get_code(EMOTION_CODE_MAP, selected_emotion),
                accuracy,
                response_time_ms,
                "|".join(state.get("current_choices", [])),
                datetime.now().isoformat(),
            ])
        print(f"[DEBUG] Selected '{selected_emotion}' for {image_data.name} in {response_time_ms}ms")
    except Exception as e:
        print("-----------!! ERROR: Could not save data to CSV. !!-----------")
        print(e)
        print("----------------------------------------------------------------")

    return (
        gr.update(visible=False, interactive=False), # emotion_choice
        gr.update(visible=True),  # next_image_btn
    )

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    state = gr.State()

    gr.Markdown("# Face Emotion Recognition Study")
    
    with gr.Column(visible=True) as instructions_section:
        gr.Markdown(
            """
            ## Instructions
            1.  An image of a face will appear. It will start very blurry.
            2.  The image will gradually become clear over 10 seconds.
            3.  As soon as you recognize the emotion, select the corresponding option below.
            4.  The image will become fully clear, and a "Next Image" button will appear.
            5.  Click "Next Image" to continue the study.
            
            **Please respond as quickly and accurately as you can. Your response time is being measured.**
            """
        )
        start_btn = gr.Button("START STUDY", variant="primary")
        status_text = gr.Markdown("")

    with gr.Column(visible=False) as main_section:
        image_display = gr.Image(label="", elem_id="image_display", height=400, width=400, interactive=False)
        progress_text = gr.Markdown("")
        emotion_choice = gr.Radio(choices=[], label="Select the emotion", visible=False, interactive=True)

        next_image_btn = gr.Button("Next Image ▶", variant="secondary", visible=False)

    # --- Event Handlers ---
    app.load(
        fn=initialize_experiment,
        outputs=[state, status_text, start_btn]
    ).then(
        fn=None,
        js=f"""() => {{
            // define animation helpers once per session
            window.animationFrameId = null;
            window.deblurImage = function() {{
                const img = document.querySelector("#image_display img");
                if (!img) return;
                const duration = {DEBLUR_DURATION_S * 1000};
                const initialBlur = 20;
                let startTime = null;
                function animate(currentTime) {{
                    if (!startTime) startTime = currentTime;
                    const elapsedTime = currentTime - startTime;
                    const progress = Math.min(elapsedTime / duration, 1);
                    const currentBlur = initialBlur * (1 - progress);
                    img.style.filter = 'blur(' + currentBlur + 'px)';
                    if (progress < 1) {{
                        window.animationFrameId = requestAnimationFrame(animate);
                    }}
                }}
                cancelAnimationFrame(window.animationFrameId);
                const img2 = document.querySelector("#image_display img");
                if (img2) img2.style.filter = 'blur(' + initialBlur + 'px)';
                window.animationFrameId = requestAnimationFrame(animate);
            }};
            window.unblurImmediately = function() {{
                cancelAnimationFrame(window.animationFrameId);
                const img = document.querySelector("#image_display img");
                if (img) img.style.filter = 'blur(0px)';
            }};
        }}"""
    )

    start_btn.click(
        fn=start_interface,
        inputs=[state],
        outputs=[instructions_section, start_btn, main_section]
    ).then(
        fn=show_next_image,
        inputs=[state],
        outputs=[state, image_display, progress_text, next_image_btn, emotion_choice]
    ).then(
        fn=None,
        js="() => window.deblurImage()"
    )

    # IMPORTANT: bind JS + Python in the SAME change call (no .then)
    emotion_choice.change(
        fn=on_emotion_select,
        inputs=[state, emotion_choice],
        outputs=[emotion_choice, next_image_btn],
        js="() => window.unblurImmediately()"
    )

    next_image_btn.click(
        fn=show_next_image,
        inputs=[state],
        outputs=[state, image_display, progress_text, next_image_btn, emotion_choice]
    ).then(
        fn=None,
        js="() => window.deblurImage()"
    )

if __name__ == "__main__":
    print("Starting Gradio app...")
    print("Please create two folders: './AI' and './Human'")
    print("Place images in them named like 'any_name_happy.jpg', 'some_face_sad.png', etc.")
    print(f"Optional metadata file: '{METADATA_FILE}' with columns image_name, emotion, sex, ethnicity, angle, face_type.")
    print(f"Participant ID via URL param '?{URL_PARAM_PARTICIPANT_ID}=...'")
    app.launch()
