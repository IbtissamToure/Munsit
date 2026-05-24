import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    landmarks_list = []
    if results.multi_hand_landmarks:
        for hlmk in results.multi_hand_landmarks:
            landmarks = []
            for lm in hlmk.landmark:        
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(np.array(landmarks))  
    return landmarks_list, results

def draw_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for hlmk in results.multi_hand_landmarks:  
            mp_drawing.draw_landmarks(
                frame,
                hlmk,
                mp_hands.HAND_CONNECTIONS    
            )
    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    print("🎥 Camera started — Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks_list, results = extract_landmarks(frame)  
        frame = draw_landmarks(frame, results)
        cv2.putText(
            frame,
            f"Hands: {len(landmarks_list)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )
        cv2.imshow("Munsit - Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()