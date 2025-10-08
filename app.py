import gradio as gr
import cv2
import numpy as np
import os
import random
import time
import csv
import uuid
from datetime import datetime
from PIL import Image

# --- Configuration ---
AI_FOLDER = "./AI"
HUMAN_FOLDER = "./Human"
CSV_FILE = "emotion_responses.csv"
DEBLUR_DURATION_S = 10

# --- Data Structure ---
class ImageData:
    """A simple class to hold information about each image."""
    def __init__(self, path, source, emotion):
        self.path = path
        self.source = source
        self.emotion = emotion
        self.name = os.path.basename(path)

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

def initialize_experiment():
    """Scans folders for images, creates dummy files if needed, and prepares the experiment state."""
    # Create demo folders/images if missing
    os.makedirs(AI_FOLDER, exist_ok=True)
    os.makedirs(HUMAN_FOLDER, exist_ok=True)
    
    images = []
    emotions = set()

    for folder, source in [(AI_FOLDER, "AI"), (HUMAN_FOLDER, "Human")]:
        if not os.path.exists(folder): 
            continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = os.path.splitext(filename)[0].split('_')
                if len(parts) < 2:
                    continue
                emotion = parts[-1].lower()
                emotions.add(emotion)
                path = os.path.join(folder, filename)
                images.append(ImageData(path, source, emotion))

    if not images:
        return None, "Error: No images found. Please add images to 'AI' and 'Human' folders with names like 'name_emotion.jpg'"

    random.shuffle(images)
    sorted_emotions = sorted(list(emotions))
    # we only have 4 buttons; trim if more
    sorted_emotions = sorted_emotions[:4] if sorted_emotions else ["happy", "sad", "angry", "surprised"]
    
    initial_state = {
        "user_id": str(uuid.uuid4()),
        "all_images": images,
        "emotions": sorted_emotions,
        "current_index": -1,
        "start_time": None
    }
    
    # Create the CSV file with headers if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'user_id', 'image_name', 'image_source', 'correct_emotion', 
                'selected_emotion', 'response_time_s', 'timestamp'
            ])
    
    return initial_state, ""

def start_interface(state):
    """Hides instructions and shows the main experiment UI."""
    num_emotions = len(state["emotions"])
    button_updates = [gr.update(visible=True, value=state["emotions"][i]) for i in range(num_emotions)]
    button_updates += [gr.update(visible=False)] * (4 - num_emotions) # Hide unused buttons

    return (
        gr.update(visible=False), # instructions_section
        gr.update(visible=False), # start_btn
        gr.update(visible=True),  # main_section
        gr.update(visible=True),  # emotion_buttons_row
        *button_updates
    )

def show_next_image(state):
    """Loads the next image and updates the state."""
    state["current_index"] += 1
    index = state["current_index"]

    num_emotions = len(state["emotions"])

    if index >= len(state["all_images"]):
        btn_updates = [gr.update(visible=False, interactive=False)] * 4
        return (
            state, 
            None, 
            "Experiment complete! Thank you for participating.", 
            gr.update(visible=False),  # next_image_btn
            gr.update(visible=False),  # emotion_buttons_row
            *btn_updates
        )

    image_data = state["all_images"][index]
    cropped_image = crop_face(image_data.path)
    
    if cropped_image is None:
        btn_updates = [gr.update(visible=False, interactive=False)] * 4
        return (
            state, 
            None, 
            f"Error loading image: {image_data.name}", 
            gr.update(visible=True),   # show Next so user can skip the broken one
            gr.update(visible=False), 
            *btn_updates
        )

    state["start_time"] = time.time()
    print(f"[DEBUG] Showing image {index+1}/{len(state['all_images'])}: {image_data.name}")

    # Enable only the number of active emotion buttons
    button_interactivity = [gr.update(visible=True, interactive=True)] * num_emotions
    button_interactivity += [gr.update(visible=False, interactive=False)] * (4 - num_emotions)

    return (
        state, 
        cropped_image, 
        f"Image {index + 1} of {len(state['all_images'])}", 
        gr.update(visible=False), # hide Next until a choice is made
        gr.update(visible=True),  # show emotion buttons row
        *button_interactivity
    )

def on_emotion_click(state, selected_emotion):
    """Handles emotion button click and records data, then shows Next."""
    # Try to save; don't let errors block UI updates
    try:
        response_time = time.time() - (state.get("start_time") or time.time())
        image_data = state["all_images"][state["current_index"]]
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                state["user_id"], image_data.name, image_data.source, image_data.emotion,
                selected_emotion, f"{response_time:.4f}", datetime.now().isoformat()
            ])
        print(f"[DEBUG] Clicked '{selected_emotion}' for {image_data.name} in {response_time:.3f}s")
    except Exception as e:
        print("-----------!! ERROR: Could not save data to CSV. !!-----------")
        print(e)
        print("----------------------------------------------------------------")

    # Disable buttons and reveal Next
    num_emotions = len(state["emotions"])
    button_interactivity = [gr.update(interactive=False)] * num_emotions
    button_interactivity += [gr.update()] * (4 - num_emotions)

    return (
        gr.update(visible=False), # emotion_buttons_row
        gr.update(visible=True),  # next_image_btn
        *button_interactivity
    )

def on_emotion_click_idx(state, idx):
    """Map a fixed button index to an emotion label."""
    # Guard in case fewer than 4 emotions exist
    if idx >= len(state["emotions"]):
        print(f"[DEBUG] Ignored click for idx {idx}; only {len(state['emotions'])} emotions configured.")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    selected_emotion = state["emotions"][idx]
    return on_emotion_click(state, selected_emotion)

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
            3.  As soon as you recognize the emotion, click the corresponding button below.
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

        with gr.Row(visible=False) as emotion_buttons_row:
            emotion_btn_1 = gr.Button(size="lg", interactive=True)
            emotion_btn_2 = gr.Button(size="lg", interactive=True)
            emotion_btn_3 = gr.Button(size="lg", interactive=True)
            emotion_btn_4 = gr.Button(size="lg", interactive=True)
            emotion_buttons = [emotion_btn_1, emotion_btn_2, emotion_btn_3, emotion_btn_4]

        next_image_btn = gr.Button("Next Image ▶", variant="secondary", visible=False)

    # --- Event Handlers ---
    app.load(
        fn=initialize_experiment,
        outputs=[state, status_text]
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
        outputs=[instructions_section, start_btn, main_section, emotion_buttons_row, *emotion_buttons]
    ).then(
        fn=show_next_image,
        inputs=[state],
        outputs=[state, image_display, progress_text, next_image_btn, emotion_buttons_row, *emotion_buttons]
    ).then(
        fn=None,
        js="() => window.deblurImage()"
    )

    # IMPORTANT: bind JS + Python in the SAME click call (no .then)
    emotion_btn_1.click(
        fn=lambda s: on_emotion_click_idx(s, 0),
        inputs=[state],
        outputs=[emotion_buttons_row, next_image_btn, *emotion_buttons],
        js="() => window.unblurImmediately()"
    )
    emotion_btn_2.click(
        fn=lambda s: on_emotion_click_idx(s, 1),
        inputs=[state],
        outputs=[emotion_buttons_row, next_image_btn, *emotion_buttons],
        js="() => window.unblurImmediately()"
    )
    emotion_btn_3.click(
        fn=lambda s: on_emotion_click_idx(s, 2),
        inputs=[state],
        outputs=[emotion_buttons_row, next_image_btn, *emotion_buttons],
        js="() => window.unblurImmediately()"
    )
    emotion_btn_4.click(
        fn=lambda s: on_emotion_click_idx(s, 3),
        inputs=[state],
        outputs=[emotion_buttons_row, next_image_btn, *emotion_buttons],
        js="() => window.unblurImmediately()"
    )

    next_image_btn.click(
        fn=show_next_image,
        inputs=[state],
        outputs=[state, image_display, progress_text, next_image_btn, emotion_buttons_row, *emotion_buttons]
    ).then(
        fn=None,
        js="() => window.deblurImage()"
    )

if __name__ == "__main__":
    print("Starting Gradio app...")
    print("Please create two folders: './AI' and './Human'")
    print("Place images in them named like 'any_name_happy.jpg', 'some_face_sad.png', etc.")
    app.launch()
