import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,   
    max_num_hands=1,
    min_detection_confidence=0.5
)

CLASSES = ['ALIF', 'BAA', 'TA', 'THA', 'JEEM', 'HAA', 'KHAA',
           'DELL', 'DHELL', 'RAA', 'ZAY', 'SEEN', 'SHEEN', 'SAD',
           'DAD', 'TAA', 'DHAA', 'AYN', 'GHAYN', 'FAA', 'QAAF',
           'KAAF', 'LAAM', 'MEEM', 'NOON', 'HA', 'WAW', 'YA']

def extract_landmarks_from_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)  
    
    return None 

def process_split(split="train"):
    """معالجة train أو valid"""
    images_dir = Path(f"data/signs/{split}/images")
    labels_dir = Path(f"data/signs/{split}/labels")
    
    X = []  
    y = []  
    skipped = 0
    
    image_files = list(images_dir.glob("*.jpg"))
    total = len(image_files)
    
    print(f"\n Processing {split}: {total} images...")
    
    for i, img_path in enumerate(image_files):
       
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue
            
        with open(label_path) as f:
            line = f.readline().strip()
            if not line:
                skipped += 1
                continue
            class_id = int(line.split()[0])
        
        
        landmarks = extract_landmarks_from_image(img_path)
        
        if landmarks is not None:
            X.append(landmarks)
            y.append(class_id)
        else:
            skipped += 1
        
        
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{total} — found: {len(X)}, skipped: {skipped}")
    
    print(f"\n Done! Found: {len(X)}, Skipped: {skipped}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    
    X_train, y_train = process_split("train")
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    
    
    X_valid, y_valid = process_split("valid")
    np.save("data/X_valid.npy", X_valid)
    np.save("data/y_valid.npy", y_valid)
    
    print("\n Dataset ready!")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_valid: {X_valid.shape}")
    print(f"y_valid: {y_valid.shape}")