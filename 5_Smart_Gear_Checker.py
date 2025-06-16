import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace with your fine-tuned gear detection model if needed

# Define required gear list with descriptions
REQUIRED_GEAR = {
    "racket": "Your badminton racket - essential for playing",
    "shoes": "Court shoes with good grip for quick movements",
    "bottle": "Water bottle to stay hydrated during play",
    "backpack": "Bag to carry all your equipment",
    "towel": "Towel to wipe off sweat during breaks",
    "shuttle": "Badminton shuttles for gameplay",
    "socks": "Clean socks for comfort and to prevent blisters",
    "jacket": "Warm-up jacket for before and after play",
    "cap": "Cap or visor to protect from sun and sweat",
    "tshirt": "Comfortable t-shirt for playing"
}

# Define alternative names for gear items to improve detection
GEAR_ALTERNATIVES = {
    "bottle": ["bottle", "water bottle", "drink", "container", "cup", "glass", "jar", "vase", "flask", "thermos", "canteen", "jug", "pitcher", "mug", "tumbler", "beverage", "liquid", "fluid"],
    "racket": ["racket", "racquet", "bat", "paddle", "club", "stick", "handle", "grip", "shaft", "sports equipment", "game equipment"],
    "shoes": ["shoes", "shoe", "footwear", "sneakers", "boots", "slippers", "sandals", "flip flops", "athletic shoes", "sports shoes", "court shoes", "tennis shoes", "running shoes", "trainers"],
    "backpack": ["backpack", "bag", "sack", "rucksack", "knapsack", "pouch", "purse", "tote", "satchel", "briefcase", "luggage", "carrier", "container", "holder"],
    "towel": ["towel", "cloth", "rag", "napkin", "wipe", "fabric", "textile", "material", "handkerchief", "washcloth", "face cloth", "bath towel", "beach towel"],
    "shuttle": ["shuttle", "shuttlecock", "bird", "birdie", "feather", "feathers", "badminton shuttle", "badminton bird", "badminton birdie"],
    "socks": ["socks", "sock", "stocking", "stockings", "hosiery", "footwear", "ankle socks", "athletic socks", "sports socks"],
    "jacket": ["jacket", "coat", "outerwear", "outer layer", "warm-up", "warmup", "sweater", "hoodie", "hoody", "pullover", "zip-up", "zipper", "windbreaker", "rain jacket"],
    "cap": ["cap", "hat", "visor", "headwear", "headgear", "baseball cap", "sports cap", "sun hat", "beanie", "beanie hat", "headband"],
    "tshirt": ["tshirt", "t-shirt", "tee", "t shirt", "shirt", "top", "jersey", "polo", "blouse", "sweater", "sweatshirt", "athletic shirt", "sports shirt"]
}

# Page config
st.set_page_config(
    page_title="Smart Equipment Checker - Badminton Edition",
    page_icon="🎒",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #1e1e1e;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #1a1a1a;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #4fc3f7;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #b0bec5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #263238;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4fc3f7;
    }
    
    /* Gear items */
    .gear-item {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        background-color: #37474f;
    }
    
    /* Success box */
    .success-box {
        background-color: #1b5e20;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4caf50;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #e65100;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff9800;
    }
    
    /* Metric box */
    .metric-box {
        background-color: #37474f;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #546e7a;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #f0f0f0;
    }
    
    /* Links */
    a {
        color: #4fc3f7;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0d47a1;
        color: white;
        border: none;
        border-radius: 0.3rem;
    }
    
    /* File uploader */
    .stFileUploader>div {
        background-color: #263238;
        border: 1px solid #546e7a;
    }
    
    /* Radio buttons */
    .stRadio>div {
        background-color: #263238;
        border-radius: 0.3rem;
        padding: 0.5rem;
    }
    
    /* Selectbox */
    .stSelectbox>div>div {
        background-color: #263238;
        border: 1px solid #546e7a;
    }
    
    /* Slider */
    .stSlider>div>div>div {
        background-color: #263238;
    }
    
    /* Progress bar */
    .stProgress>div>div>div {
        background-color: #4fc3f7;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>🎒 Smart Equipment Checker</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #616161;'>Badminton Edition</h3>", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class='info-box'>
    <h4>📋 How It Works</h4>
    <p>This tool helps you ensure you have all the necessary equipment before heading to the court. 
    Simply use your camera, upload an image, or share a video of your gear, and our AI will check if you have everything you need!</p>
</div>
""", unsafe_allow_html=True)

# Add a section about model training
with st.expander("🔧 About Model Training"):
    st.markdown("""
    <div class='info-box'>
        <h4>🤖 Custom Model Training</h4>
        <p>For optimal detection of badminton gear, you can train a custom YOLO model specifically for these items. Here's how:</p>
        
        <h5>1. Data Collection</h5>
        <p>Collect at least 100 images of each gear item (racket, shuttle, shoes, etc.) from different angles and lighting conditions.</p>
        
        <h5>2. Data Annotation</h5>
        <p>Use tools like LabelImg or CVAT to annotate your images with bounding boxes around each gear item.</p>
        
        <h5>3. Training Configuration</h5>
        <p>Create a YAML file with your class names and paths to training/validation data:</p>
        <pre>
        path: path/to/dataset
        train: images/train
        val: images/val
        
        names:
          0: racket
          1: shuttle
          2: shoes
          3: socks
          4: bottle
          5: backpack
          6: towel
          7: jacket
          8: cap
          9: tshirt
        </pre>
        
        <h5>4. Train the Model</h5>
        <p>Run the following command to train your custom model:</p>
        <pre>
        yolo task=detect mode=train model=yolov8n.pt data=badminton_gear.yaml epochs=100 imgsz=640
        </pre>
        
        <h5>5. Use Your Custom Model</h5>
        <p>Replace the model path in this app with your trained model:</p>
        <pre>
        model = YOLO("path/to/your/trained_model.pt")
        </pre>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with options and help
with st.sidebar:
    st.markdown("## 🛠️ Options")
    mode = st.selectbox(
        "Choose Input Mode",
        ["Live Camera", "Upload Media"],
        help="Select how you want to check your gear"
    )
    
    st.markdown("## 💡 Tips")
    st.markdown("""
    - **Live Camera**: Best for quick checks of your current setup
    - **Image Upload**: Good for checking a specific arrangement of your gear
    - **Video Upload**: Ideal for showing multiple angles of your equipment
    
    Make sure your gear is well-lit and clearly visible for best results!
    """)
    
    st.markdown("## ⚙️ Settings")
    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Lower values will detect more objects but may include false positives"
    )

# Helper function to draw boxes and get gear summary
def detect_gear(frame, conf_threshold=0.3):
    results = model(frame)[0]
    detected_items = []
    detected_labels = []
    
    # Debug information
    debug_info = []
    
    # First pass: collect all detections with confidence above threshold
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        
        # Add to debug info
        debug_info.append(f"{label}: {conf:.2f}")
        
        if conf > conf_threshold:
            detected_labels.append(label)
            # Draw bounding boxes
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Second pass: check for gear items using alternatives with case-insensitive matching
    for gear_item, alternatives in GEAR_ALTERNATIVES.items():
        for alt in alternatives:
            # Check if any detected label contains the alternative name (case-insensitive)
            for detected_label in detected_labels:
                if alt.lower() in detected_label.lower():
                    detected_items.append(gear_item)
                    break
            if gear_item in detected_items:
                break
    
    # If no items detected but we have some detections, try with even lower threshold
    if not detected_items and detected_labels:
        debug_info.append("No gear items detected with current threshold, trying with lower threshold...")
        for gear_item, alternatives in GEAR_ALTERNATIVES.items():
            for alt in alternatives:
                for detected_label in detected_labels:
                    if alt.lower() in detected_label.lower():
                        detected_items.append(gear_item)
                        debug_info.append(f"Found {gear_item} using lower threshold matching")
                        break
                if gear_item in detected_items:
                    break
    
    return frame, set(detected_items), debug_info

# Main content area
st.markdown("<h3 class='sub-header'>🔍 Gear Detection</h3>", unsafe_allow_html=True)

# 1. Live Camera Mode
if mode == "Live Camera":
    st.markdown("""
    <div class='info-box'>
        <h4>📸 Live Camera Mode</h4>
        <p>Position your gear in front of the camera. The system will scan for 10 seconds to detect all your equipment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        detected_items_over_time = set()
        all_debug_info = []
        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame, detected, debug_info = detect_gear(frame, confidence_threshold)
            detected_items_over_time.update(detected)
            all_debug_info.extend(debug_info)
            
            # Update progress
            progress = (time.time() - start_time) / 10
            progress_bar.progress(progress)
            status_text.text(f"Scanning... {int(progress * 100)}% complete")
            
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()
        
        progress_bar.progress(1.0)
        status_text.text("✅ Scan complete!")
        found_items = detected_items_over_time
    
    with col2:
        st.markdown("### 🎯 Live Detection")
        st.markdown("Items detected so far:")
        for item in found_items:
            st.markdown(f"- ✅ {item.capitalize()}")
        
        # Show debug information
        with st.expander("🔍 Debug Information"):
            st.markdown("**All detected objects:**")
            for info in all_debug_info:
                st.markdown(f"- {info}")

# 2. Upload Media Mode
elif mode == "Upload Media":
    media_type = st.radio(
        "Select Media Type",
        ["Image", "Video"],
        horizontal=True,
        help="Choose whether to upload a single image or a video"
    )
    
    if media_type == "Image":
        st.markdown("""
        <div class='info-box'>
            <h4>🖼️ Image Upload Mode</h4>
            <p>Upload a clear photo of your badminton gear. Make sure all items are visible in the image.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload an image of your gear setup",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Read the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                
                # Process the image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_processed, detected, debug_info = detect_gear(frame_rgb, confidence_threshold)
                
                # Display the processed image
                st.image(frame_processed, channels="RGB", caption="Detected Gear", use_column_width=True)
            
            with col2:
                st.markdown("### 🎯 Detection Results")
                st.markdown("Items detected in your image:")
                for item in detected:
                    st.markdown(f"- ✅ {item.capitalize()}")
                
                # Show debug information
                with st.expander("🔍 Debug Information"):
                    st.markdown("**All detected objects:**")
                    for info in debug_info:
                        st.markdown(f"- {info}")
            
            st.success("✅ Image scan complete!")
            found_items = detected
    
    else:  # Video upload
        st.markdown("""
        <div class='info-box'>
            <h4>🎥 Video Upload Mode</h4>
            <p>Upload a video showing your badminton gear from different angles. The system will analyze frames throughout the video.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload a video of your gear setup",
            type=["mp4", "mov"],
            help="Supported formats: MP4, MOV"
        )
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            stframe = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            detected_items_over_time = set()
            all_debug_info = []
            frame_count = 0
            processed_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 10 == 0:  # Sample every 10th frame
                    frame, detected, debug_info = detect_gear(frame, confidence_threshold)
                    detected_items_over_time.update(detected)
                    all_debug_info.extend(debug_info)
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                    
                    # Update progress
                    processed_frames += 1
                    progress = processed_frames / (total_frames // 10)
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Processing video... {int(min(progress, 1.0) * 100)}% complete")
                
                frame_count += 1
            
            cap.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Video scan complete!")
            
            st.success("✅ Video scan complete!")
            found_items = detected_items_over_time
            
            # Show debug information
            with st.expander("🔍 Debug Information"):
                st.markdown("**All detected objects:**")
                for info in all_debug_info:
                    st.markdown(f"- {info}")

# Display Gear Checklist
if (mode == "Live Camera") or (mode == "Upload Media" and uploaded_file is not None):
    st.markdown("<h3 class='sub-header'>📋 Gear Checklist</h3>", unsafe_allow_html=True)
    
    # Add a manual override option
    st.markdown("### ⚙️ Manual Override")
    st.markdown("If the AI missed any items, you can manually mark them as detected:")
    
    manual_items = {}
    for item in REQUIRED_GEAR.keys():
        if item not in found_items:
            manual_items[item] = st.checkbox(f"Mark {item.capitalize()} as detected")
    
    # Update found_items with manual selections
    for item, is_selected in manual_items.items():
        if is_selected:
            found_items.add(item)
    
    # Add a section to help with detection issues
    st.markdown("### 🔍 Troubleshooting Detection Issues")
    st.markdown("""
    If the AI is having trouble detecting your gear:
    
    1. **Try different angles**: Position your gear so it's clearly visible
    2. **Improve lighting**: Ensure your gear is well-lit
    3. **Adjust confidence threshold**: Use the slider in the sidebar to lower the threshold
    4. **Use manual override**: If needed, manually mark items as detected
    5. **Try different input modes**: Sometimes one mode works better than another
    6. **Train a custom model**: For optimal results, train a custom YOLO model specifically for badminton gear
    """)
    
    # Create a more detailed checklist
    for item, description in REQUIRED_GEAR.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            if item in found_items:
                st.markdown(f"""
                <div class='success-box'>
                    <h4>✅ {item.capitalize()}</h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='warning-box'>
                    <h4>❌ {item.capitalize()}</h4>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Description:** {description}")
            if item in found_items:
                st.markdown("**Status:** ✅ Detected")
            else:
                st.markdown("**Status:** ❌ Missing")

    # Overall completion metric
    completion_percentage = (len(found_items) / len(REQUIRED_GEAR)) * 100
    st.markdown(f"""
    <div class='metric-box'>
        <h3>Gear Completion</h3>
        <h2>{len(found_items)} / {len(REQUIRED_GEAR)} ({completion_percentage:.0f}%)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Final recommendation
    if len(found_items) == len(REQUIRED_GEAR):
        st.markdown("""
        <div class='success-box'>
            <h3>🎉 You're fully equipped and ready to smash! 🏸</h3>
            <p>All your gear is packed and ready to go. Have a great game!</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        missing_items = [item for item in REQUIRED_GEAR.keys() if item not in found_items]
        st.markdown(f"""
        <div class='warning-box'>
            <h3>⚠️ Almost there!</h3>
            <p>You're missing the following items:</p>
            <ul>
                {''.join([f"<li>{item.capitalize()}</li>" for item in missing_items])}
            </ul>
            <p>Please make sure to pack these items before heading to the court.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a reset button
    if st.button("🔄 Scan Again", type="primary"):
        st.experimental_rerun() 