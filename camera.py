import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "mask_detector_cnn.h5"   # path to your saved model
CONFIDENCE_MIN  = 0.6                       # ignore predictions below this
CLASS_NAMES     = ["with_mask", "without_mask"]

# Colors: BGR format for OpenCV
COLORS = {
    "with_mask":    (0, 200, 0),    # green
    "without_mask": (0, 0, 220),    # red
}

# ── Load model & face detector ────────────────────────────────────────────────
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

# OpenCV's built-in face detector (no extra files needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Helper: preprocess a face crop for the model ─────────────────────────────
def prepare_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = img_to_array(face_img)
    face_img = preprocess_input(face_img)          # scale to [-1, 1]
    face_img = np.expand_dims(face_img, axis=0)    # add batch dimension
    return face_img

# ── Start camera ──────────────────────────────────────────────────────────────
print("[INFO] Starting camera — press Q to quit")
cap = cv2.VideoCapture(0)   # 0 = default webcam, change to 1 for external cam

if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)       # ignore tiny detections
    )

    for (x, y, w, h) in faces:
        # Add padding around the face crop for better context
        pad    = 20
        x1     = max(0, x - pad)
        y1     = max(0, y - pad)
        x2     = min(frame.shape[1], x + w + pad)
        y2     = min(frame.shape[0], y + h + pad)

        face_crop  = frame[y1:y2, x1:x2]
        face_input = prepare_face(face_crop)

        # Predict
        preds      = model.predict(face_input, verbose=0)[0]
        class_idx  = np.argmax(preds)
        confidence = preds[class_idx]

        if confidence < CONFIDENCE_MIN:
            continue   # skip low-confidence detections

        label = CLASS_NAMES[class_idx]
        color = COLORS[label]

        # ── Draw bounding box ─────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ── Draw label background ─────────────────────────────────────────────
        label_text = f"{label}: {confidence * 100:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 12),
            (x1 + text_w + 8, y1),
            color, -1                   # filled rectangle
        )

        # ── Draw label text ───────────────────────────────────────────────────
        cv2.putText(
            frame, label_text,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2     # white text
        )

    # ── FPS counter (optional) ────────────────────────────────────────────────
    cv2.putText(frame, "Press Q to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera closed.")