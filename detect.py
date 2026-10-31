import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("sign_model.h5")

# Labels A-Z
labels = [chr(i) for i in range(65, 91)]

IMG_SIZE = 64

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Define ROI (box for hand)
    x1, y1, x2, y2 = 300, 100, 600, 400
    roi = frame[y1:y2, x1:x2]

    # Preprocess
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    # Prediction
    pred = model.predict(img, verbose=0)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    # Only show if confident
    if confidence > 0.7:
        label = labels[class_index]
    else:
        label = "..."

    # Draw box
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # Show prediction
    cv2.putText(frame, f"{label} ({confidence:.2f})",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Sign Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()