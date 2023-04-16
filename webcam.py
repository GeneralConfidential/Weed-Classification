import cv2
import numpy as np
import tensorflow as tf

# Define constants
model_path = 'model.h5'
class_names = ['CELOSIA ARGENTEA L', 'CROWFOOT GRASS', 'PURPLE CHLORIS']

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Start video capture from default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame to the size expected by the model
    resized_frame = cv2.resize(frame, (224, 224))

    # Convert the image to RGB and preprocess for inference
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img_array = tf.keras.preprocessing.image.img_to_array(rgb_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Make predictions on the image
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Display the class label on the screen
    class_label = class_names[predicted_class]
    cv2.putText(frame, class_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
