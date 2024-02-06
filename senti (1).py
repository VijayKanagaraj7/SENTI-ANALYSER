import streamlit as st
import numpy as np
import cv2
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from deepface import DeepFace

nltk.download('stopwords')
nltk.download('vader_lexicon')

stemmer = nltk.SnowballStemmer("english")
data = pd.read_csv("D:/project/twitter.csv")
data.head()

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
data.head()

data = data[["tweet", "labels"]]
data.head()

nltk.download('stopwords')
stopword = set(stopwords.words('english'))

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment score and label
def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']

    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def clean(text):
    text = str(text).lower()
    text = re.sub('', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# Function to get user input for a journal entry
#def get_journal_entry():
 #   entry = input("Enter your journal entry: ")
  #  return entry

# Function to get the current date and time
def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function to save the journal entry to a CSV file
def save_to_csv(data):
    # Specify the CSV file path
    csv_file_path = 'journal_data.csv'

    # Check if the CSV file exists, if not, create it with headers
    try:
        with open(csv_file_path, 'r') as file:
            pass
    except FileNotFoundError:
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['date_time', 'entry'])

    # Append the new data to the CSV file
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

model = DeepFace.build_model("Emotion")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'pain']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform real-time emotion detection
def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        normalized_face = resized_face / 255.0
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        detected_emotion = emotion_labels[emotion_idx]
        return detected_emotion


# Streamlit app
def main():
    st.title("Real-time Emotion Detection and Sentiment Analysis")

    option = st.radio("Choose an option:", ["Face Detection", "Enter Journal Entry"])

    if option == "Face Detection":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Unable to open the webcam.")
            st.stop()
        valence = 0.5
        start_detection = st.checkbox("Start Face Detection")
        while start_detection:
            ret, frame = cap.read()
            detected_emotion = detect_emotion(frame)

    # Update valence based on detected emotion
            if detected_emotion in ['happy', 'surprise']:
                valence += 0.1
            elif detected_emotion in ['angry', 'sad']:
                valence -= 0.1

    # Display the webcam feed with emotion information
            st.image(frame, channels="BGR")
            st.text(f"Detected Emotion: {detected_emotion}")
            st.text(f"Updated Valence: {valence:.2f}")
            # Release the video capture object
            cap.release()
            cv2.destroyAllWindows()

    elif option == "Enter Journal Entry":
        st.header("Journal Entry and Sentiment Analysis")
        while True:
            #entry = get_journal_entry()
            entry=st.text_area("Enter your journal entry: ")
            date_time = get_current_datetime()

            # Save the entry to the CSV file
            #save_to_csv([date_time, entry])

            # Ask the user if they want to continue entering journal entries
            continue_input = st.text_area("Do you want to enter another journal entry? (yes/no): ").lower()

            if continue_input != 'yes' or 'y' or 'yess':
                break


        if st.button("Submit Journal Entry and Analyze Sentiment"):
            # Save the entry to the CSV file
            sentiment=get_sentiment(entry)
            #save_to_csv([date_time, entry])
            #df = pd.read_csv('journal_data.csv')
            # Apply cleaning to the 'entry' column
            #df['cleaned_entry'] = df['entry'].apply(clean)

            # Save the cleaned data to a new Excel file
            #excel_file_path = 'test.xlsx'
            #df.to_excel(excel_file_path, index=False)

            # Display the cleaned data
            #print("Cleaned Data:")
            #print(df[['date_time', 'cleaned_entry']])
            #print(f"\nCleaned data saved to {excel_file_path}")

            # Apply sentiment analysis to the 'cleaned_entry' column
            #df['sentiment'] = df['cleaned_entry'].apply(get_sentiment)

            # Count the number of positive, negative, and neutral entries
            #sentiment_counts = df['sentiment'].value_counts()

            st.success("Journal entry saved successfully!")
            st.subheader("Sentiment Analysis Result:")
            st.write(f"The sentiment of the entered text is: {sentiment}")


if __name__ == "__main__":
    main()