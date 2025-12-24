import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import math

def download_model():
    """Télécharge le modèle si nécessaire"""
    model_path = 'hand_landmarker.task'
    
    if not os.path.exists(model_path):
        print("Téléchargement du modèle... (peut prendre quelques secondes)")
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_path)
        print("Modèle téléchargé avec succès!")
    
    return model_path

def calculate_distance(point1, point2):
    """Calcule la distance euclidienne entre deux points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_finger_extended(hand_landmarks, finger_tip_idx, finger_pip_idx):
    """Vérifie si un doigt est tendu"""
    tip = hand_landmarks[finger_tip_idx]
    pip = hand_landmarks[finger_pip_idx]
    wrist = hand_landmarks[0]
    
    # Distance du bout du doigt au poignet vs articulation au poignet
    tip_to_wrist = calculate_distance((tip.x, tip.y), (wrist.x, wrist.y))
    pip_to_wrist = calculate_distance((pip.x, pip.y), (wrist.x, wrist.y))
    
    return tip_to_wrist > pip_to_wrist

def detect_heart_gesture(detection_result, frame_shape):
    """Détecte si les deux mains forment un cœur"""
    if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) < 2:
        return False
    
    h, w = frame_shape[:2]
    hands = detection_result.hand_landmarks
    
    # Récupérer les positions des pouces et index des deux mains
    hand1_thumb_tip = (hands[0][4].x * w, hands[0][4].y * h)
    hand1_index_tip = (hands[0][8].x * w, hands[0][8].y * h)
    
    hand2_thumb_tip = (hands[1][4].x * w, hands[1][4].y * h)
    hand2_index_tip = (hands[1][8].x * w, hands[1][8].y * h)
    
    # Calculer les distances entre les doigts
    thumb_distance = calculate_distance(hand1_thumb_tip, hand2_thumb_tip)
    index_distance = calculate_distance(hand1_index_tip, hand2_index_tip)
    
    # Les pouces doivent être proches et les index aussi
    # Forme approximative d'un cœur
    if thumb_distance < 80 and index_distance < 80:
        # Vérifier que les mains sont à peu près au même niveau vertical
        hand1_center_y = (hand1_thumb_tip[1] + hand1_index_tip[1]) / 2
        hand2_center_y = (hand2_thumb_tip[1] + hand2_index_tip[1]) / 2
        
        if abs(hand1_center_y - hand2_center_y) < 100:
            return True
    
    return False

def detect_peace_sign(hand_landmarks):
    """Détecte le signe de la paix (V avec index et majeur)"""
    # Index et majeur tendus, autres doigts pliés
    index_extended = is_finger_extended(hand_landmarks, 8, 6)
    middle_extended = is_finger_extended(hand_landmarks, 12, 10)
    ring_folded = not is_finger_extended(hand_landmarks, 16, 14)
    pinky_folded = not is_finger_extended(hand_landmarks, 20, 18)
    
    return index_extended and middle_extended and ring_folded and pinky_folded

def detect_thumbs_up(hand_landmarks):
    """Détecte le pouce levé"""
    thumb_extended = is_finger_extended(hand_landmarks, 4, 2)
    index_folded = not is_finger_extended(hand_landmarks, 8, 6)
    middle_folded = not is_finger_extended(hand_landmarks, 12, 10)
    ring_folded = not is_finger_extended(hand_landmarks, 16, 14)
    pinky_folded = not is_finger_extended(hand_landmarks, 20, 18)
    
    return thumb_extended and index_folded and middle_folded and ring_folded and pinky_folded

def detect_ok_sign(hand_landmarks):
    """Détecte le signe OK (pouce et index formant un cercle)"""
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    
    # Distance entre pouce et index
    distance = calculate_distance((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
    
    # Autres doigts tendus
    middle_extended = is_finger_extended(hand_landmarks, 12, 10)
    ring_extended = is_finger_extended(hand_landmarks, 16, 14)
    pinky_extended = is_finger_extended(hand_landmarks, 20, 18)
    
    return distance < 0.05 and middle_extended and ring_extended and pinky_extended

def detect_fist(hand_landmarks):
    """Détecte un poing fermé"""
    all_folded = all([
        not is_finger_extended(hand_landmarks, 8, 6),   # Index
        not is_finger_extended(hand_landmarks, 12, 10),  # Majeur
        not is_finger_extended(hand_landmarks, 16, 14),  # Annulaire
        not is_finger_extended(hand_landmarks, 20, 18)   # Auriculaire
    ])
    return all_folded

def run_hand_tracking():
    # Télécharger le modèle
    model_path = download_model()
    
    # Créer le détecteur de mains
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Initialiser la capture vidéo
    cap = cv2.VideoCapture(0)
    
    print("Appuyez sur 'q' pour quitter")
    print("\nGestes détectés:")
    print("- Cœur (deux mains)")
    print("- Signe de paix (V)")
    print("- Pouce levé")
    print("- Signe OK")
    print("- Poing fermé")
    
    # Définir les connexions entre les landmarks
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Pouce
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Majeur
        (0, 13), (13, 14), (14, 15), (15, 16),  # Annulaire
        (0, 17), (17, 18), (18, 19), (19, 20),  # Auriculaire
        (5, 9), (9, 13), (13, 17)  # Paume
    ]
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            print("Frame vide, on continue.")
            continue
        
        # Convertir BGR à RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Créer un objet Image MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Détecter les mains
        detection_result = detector.detect(mp_image)
        
        gestures_detected = []
        
        # Détecter le geste du cœur (nécessite 2 mains)
        if detect_heart_gesture(detection_result, frame.shape):
            gestures_detected.append("❤️ COEUR")
        
        # Dessiner les landmarks si des mains sont détectées
        if detection_result.hand_landmarks:
            h, w, c = frame.shape
            
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Dessiner les points
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # Dessiner les connexions
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                
                # Détecter les gestes pour chaque main
                hand_gesture = None
                if detect_peace_sign(hand_landmarks):
                    hand_gesture = "✌️ PAIX"
                elif detect_thumbs_up(hand_landmarks):
                    hand_gesture = "👍 POUCE LEVE"
                elif detect_ok_sign(hand_landmarks):
                    hand_gesture = "👌 OK"
                elif detect_fist(hand_landmarks):
                    hand_gesture = "✊ POING"
                
                if hand_gesture and hand_gesture not in gestures_detected:
                    gestures_detected.append(hand_gesture)
        
        # Afficher les gestes détectés
        y_offset = 50
        for gesture in gestures_detected:
            cv2.putText(frame, gesture, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            y_offset += 60
        
        # Afficher le résultat (miroir)
        cv2.imshow("Detection de gestes", cv2.flip(frame, 1))
        
        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    print(f"MediaPipe version: {mp.__version__}")
    print(f"CV version: {cv2.__version__}")

    run_hand_tracking()