import cv2
from deepface import DeepFace
from pip import main
import streamlit as st


model = DeepFace.build_model("Emotion")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'pain']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform face detection
def detect_faces(cap):
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = gray_frame[y:y + h, x:x + w]

            # Resize the face ROI to match the input shape of the model
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

            # Normalize the resized face image
            normalized_face = resized_face / 255.0

            # Reshape the image to match the input shape of the model
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)

            # Predict emotions using the pre-trained model
            preds = model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            emotion = emotion_labels[emotion_idx]

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Display the detected emotion
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)



    for (x, y, w, h) in faces:
        # ... (existing code)

        # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Adjust valence based on detected emotion
        if emotion in ['happy', 'surprise']:
            valence += 0.1  # Increase valence for positive emotions
        elif emotion in ['angry', 'sad']:
            valence -= 0.1  # Decrease valence for negative emotions

        # Display the detected emotion
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Streamlit app
def main():
    st.title("Real-time Emotion Detection")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to open the webcam.")
        return

    # Button to start face detection
    start_button = st.button("Start Face Detection")

    if start_button:
        detect_faces(cap)
        text = emotion_labels
        st.put_text(text)

        # Stop the face detection when the button is clicked again
        stop_button = st.button("Stop Face Detection")
        if stop_button:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()