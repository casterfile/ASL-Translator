import cv2
import numpy as np
import mediapipe as mp
import pickle
from PIL import Image, ImageTk
import time

# Load the Random Forest model and labels
labels = {
    "a": "a", "b": "b", "c": "c", "d": "d", "e": "e", "f": "f", "g": "g", "h": "h", "i": "i",
    "j": "j", "k": "k", "l": "l", "m": "m", "n": "n", "o": "o", "p": "p", "q": "q", "r": "r",
    "s": "s", "t": "t", "u": "u", "v": "v", "w": "w", "x": "x", "y": "y", "z": "z",
    "1": "Back Space", "2": "Clear", "3": "Space", "4": ""
}

with open("./ASL_model.p", "rb") as f:
    model = pickle.load(f)

rf_model = model["model"]

# Initialize Mediapipe components
mp_hands = mp.solutions.hands  # Hand tracking solution
mp_drawing = mp.solutions.drawing_utils  # Drawing utility
mp_drawing_styles = mp.solutions.drawing_styles  # Pre-defined drawing styles

# Configure the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,  # Use dynamic mode for video streams
    max_num_hands=1,  # Track at most one hand
    min_detection_confidence=0.9  # Set a high detection confidence threshold
)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Strings to store the concatenated sentence
predicted_text = " "
same_characters = ""
final_characters = ""
count = 0

# Initialize time and previous character tracking
last_update_time = time.time()
previous_character = ""

# Function to update each frame and predict the character
def update_frame(video_label, text_area):
    global predicted_text, same_characters, final_characters, count, last_update_time, previous_character
    ret, frame = cap.read()  # Capture frame-by-frame
    if ret:
        # Process the frame to display hand landmarks and predict the character
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_image = hands.process(frame_rgb)
        hand_landmarks = processed_image.multi_hand_landmarks
        height, width, _ = frame.shape

        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmark, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Collect landmark coordinates for prediction
                x_coordinates = [landmark.x for landmark in hand_landmark.landmark]
                y_coordinates = [landmark.y for landmark in hand_landmark.landmark]
                min_x, min_y = min(x_coordinates), min(y_coordinates)
                
                normalized_landmarks = []
                for coordinates in hand_landmark.landmark:
                    normalized_landmarks.extend([
                        coordinates.x - min_x,
                        coordinates.y - min_y
                    ])
                
                # Predict the character using the model
                sample = np.asarray(normalized_landmarks).reshape(1, -1)
                predicted_character = rf_model.predict(sample)[0]

                if predicted_character != "4":
                    predicted_text += predicted_character
                    
                    # Accumulate the predicted character based on time and character change
                    current_time = time.time()
                    if (current_time - last_update_time >= 5 or predicted_character != previous_character) and predicted_character not in ["1", "2", "3", "4"]:
                        final_characters += predicted_character
                        last_update_time = current_time  # Reset the timer
                        previous_character = predicted_character  # Update the previous character
                        if len(final_characters) > 30:
                            final_characters = "";

                    # Handle special characters when count reaches 30
                    if count == 30:
                        if predicted_character == "1":  # Back Space
                            if final_characters:
                                final_characters = final_characters[:-1]  # Remove the last character

                        elif predicted_character == "2":  # Clear
                            final_characters = ""  # Clear all characters

                        elif predicted_character == "3":  # Space
                            final_characters += " "  # Add space

                        count = 0
                        same_characters = ""

                    # Coordinates and colors for current character
                    background_color = (0, 150, 250)  # Background color (orange)
                    text_color = (0, 0, 0)  # Text color (black)
                    font_scale = 1
                    thickness = 2

                    # Calculate the width and height of the text box
                    (text_width, text_height), baseline = cv2.getTextSize(predicted_character, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    # Calculate center position for the text box
                    frame_center_x = width // 2
                    frame_center_y = height // 2

                    # Calculate top-left corner for the background rectangle based on the centered position
                    background_top_left = (frame_center_x - text_width // 2 - 90, frame_center_y - text_height // 2 - 10)
                    background_bottom_right = (frame_center_x + text_width // 2 + 90, frame_center_y + text_height // 2 + 10)

                    # Draw the filled rectangle as the background for text in the center
                    cv2.rectangle(frame, background_top_left, background_bottom_right, background_color, -1)

                    # Draw the text on top of the rectangle, centered
                    cv2.putText(
                        img=frame,
                        text=labels[predicted_character],
                        org=(frame_center_x - text_width // 2, frame_center_y + text_height // 2),  # Center the text within the rectangle
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=text_color,
                        thickness=thickness,
                        lineType=cv2.LINE_AA
                    )

                    # Display the cumulative text at the bottom of the screen
                    bottom_text_color = (255, 255, 255)  # White color for the bottom text
                    bottom_font_scale = 1
                    bottom_thickness = 2
                    bottom_text_position = (100, height - 30)  # Position at the bottom of the frame

                    cv2.putText(
                        img=frame,
                        text=final_characters,
                        org=bottom_text_position,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=bottom_font_scale,
                        color=bottom_text_color,
                        thickness=bottom_thickness,
                        lineType=cv2.LINE_AA
                    )

        # Convert the frame to ImageTk format and update the label
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection
        video_label.configure(image=imgtk)

    video_label.after(10, lambda: update_frame(video_label, text_area))  # Repeat every 10 ms

# Function to release the video capture
def release_video():
    cap.release()
    cv2.destroyAllWindows()
