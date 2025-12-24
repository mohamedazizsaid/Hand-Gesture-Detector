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
    
    # Variables pour le dessin
    canvas = None
    px, py = 0, 0
    draw_color = (255, 255, 0) # Cyan
    brush_thickness = 5
    eraser_thickness = 50
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            print("Frame vide, on continue.")
            continue
            
        # Initialiser le canvas si nécessaire (même taille que la frame)
        if canvas is None:
            canvas = np.zeros_like(frame)
        
        # Miroir horizontal pour une interaction plus naturelle
        frame = cv2.flip(frame, 1)
        
        # Convertir BGR à RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Créer un objet Image MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Détecter les mains
        detection_result = detector.detect(mp_image)
        
        gestures_detected = []
        is_drawing = False
        
        # Détecter le geste du cœur (nécessite 2 mains)
        # Note: detect_heart_gesture utilisait les coordonnées normalisées, 
        # mais ici on a flippé l'image. Il faudrait adapter si le coeur dépend de la gauche/droite.
        # Pour l'instant on laisse tel quel, le flip affecte l'affichage surtout.
        
        # Dessiner les landmarks si des mains sont détectées
        if detection_result.hand_landmarks:
            h, w, c = frame.shape
            
            # On ne prend que la première main détectée pour le dessin pour éviter les conflits
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # --- Logique de dessin ---
            # Index levé ?
            index_up = is_finger_extended(hand_landmarks, 8, 6)
            # Majeur levé ?
            middle_up = is_finger_extended(hand_landmarks, 12, 10)
            # Annulaire levé ?
            ring_up = is_finger_extended(hand_landmarks, 16, 14)
            
            # Coordonnées du bout de l'index (inversées car on a flippé l'frame)
            # Attention: hand_landmarks sont normalisés (0-1). 
            # Comme on a flippé l'image avec cv2.flip(frame, 1), l'image affichée est inversée en X.
            # MAIS MediaPipe détecte sur l'image RGB non flippée (si on passait frame_rgb avant le flip).
            # ARGH. Attend.
            # J'ai flippé 'frame' AVANT de créer mp_image. Donc mp_image EST flippée.
            # Donc les coords x de landmarks correspondent bien à l'image flippée. C'est bon.
            
            x1 = int(hand_landmarks[8].x * w)
            y1 = int(hand_landmarks[8].y * h)
            
            # Mode DESSIN : Index levé SEULEMENT (Majeur baissé pour être sûr, ou juste Index haut)
            # Pour être précis : Index Haut, les autres fermés c'est mieux.
            if index_up and not middle_up and not ring_up:
                cv2.circle(frame, (x1, y1), 10, draw_color, -1)
                
                if px == 0 and py == 0:
                    px, py = x1, y1
                
                cv2.line(canvas, (px, py), (x1, y1), draw_color, brush_thickness)
                px, py = x1, y1
                is_drawing = True
                
            # Mode GOMME/PAUSE : Si Index et Majeur levés (Signe de paix/Selection), on ne dessine pas
            # On réinitialise juste le point précédent pour ne pas tracer de trait "téléporté"
            else:
                px, py = 0, 0
                
            # --- Fin Logique dessin ---

            for idx, hand_landmarks_list in enumerate(detection_result.hand_landmarks):
                # Dessiner les points
                for landmark in hand_landmarks_list:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # Dessiner les connexions
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks_list[start_idx]
                    end = hand_landmarks_list[end_idx]
                    
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                
                # Détecter les gestes pour chaque main
                # Note: detect_heart_gesture a besoin de la liste complète 'detection_result' non modifiée
                
                hand_gesture = None
                if detect_peace_sign(hand_landmarks_list):
                    hand_gesture = "✌️ PAIX"
                elif detect_thumbs_up(hand_landmarks_list):
                    hand_gesture = "👍 POUCE LEVE"
                elif detect_ok_sign(hand_landmarks_list):
                    hand_gesture = "👌 OK"
                elif detect_fist(hand_landmarks_list):
                    hand_gesture = "✊ POING"
                
                if hand_gesture and hand_gesture not in gestures_detected:
                    gestures_detected.append(hand_gesture)
        else:
            # Si pas de main, reset coords
            px, py = 0, 0

        # Fusionner le canvas avec la frame
        # Créer un masque gris du canvas
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        
        # Isoler la zone du dessin dans frame (la mettre en noir là où il y a du dessin)
        frame = cv2.bitwise_and(frame, img_inv)
        # Ajouter le dessin (canvas)
        frame = cv2.bitwise_or(frame, canvas)

        # Afficher les gestes détectés
        y_offset = 50
        for gesture in gestures_detected:
            cv2.putText(frame, gesture, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            y_offset += 60
            
        cv2.putText(frame, "Index: DESSIN | 'c': EFFACER | 'q': QUITTER", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Afficher le résultat
        # Note: on a déjà fait le flip au début, donc pas besoin de le refaire ici
        cv2.imshow("Detection de gestes + Dessin", frame)
        
        # Gestion clavier
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame)
            print("Canvas effacé !")
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    print(f"MediaPipe version: {mp.__version__}")
    print(f"CV version: {cv2.__version__}")

    run_hand_tracking()