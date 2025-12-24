# 🖐️ Détecteur de Gestes avec MediaPipe

Application de reconnaissance de gestes en temps réel utilisant votre webcam et MediaPipe.

## ✨ Gestes détectés

| Geste | Description | Mains requises |
|-------|-------------|----------------|
| ❤️ Cœur | Pouces et index qui se touchent | 2 mains |
| ✌️ Paix | Index et majeur tendus | 1 main |
| 👍 Pouce levé | Seul le pouce tendu | 1 main |
| 👌 OK | Cercle pouce-index, autres doigts tendus | 1 main |
| ✊ Poing | Tous les doigts fermés | 1 main |

## 📋 Prérequis

```bash
pip install opencv-python mediapipe numpy
```

## 🚀 Utilisation

1. Lancez le script :
```bash
python hand_gesture_detector.py
```

2. Placez vos mains devant la webcam

3. Faites les gestes - ils seront détectés et affichés à l'écran !

4. Appuyez sur **'q'** pour quitter

## 🎯 Comment ça marche ?

Le programme utilise **MediaPipe Hand Landmarker** qui détecte 21 points clés sur chaque main. En analysant les positions et distances entre ces points, l'algorithme reconnaît les différents gestes.

### Au premier lancement
Le modèle MediaPipe (~10 MB) sera automatiquement téléchargé dans le répertoire courant.

## ⚙️ Personnalisation

### Ajuster la sensibilité
Modifiez les seuils de distance dans les fonctions de détection :

```python
# Pour le cœur (ligne ~60)
if thumb_distance < 80 and index_distance < 80:  # Réduire pour plus de précision
```

### Ajouter un nouveau geste
Créez une fonction qui analyse les positions des landmarks :

```python
def detect_mon_geste(hand_landmarks):
    # Votre logique de détection
    return True  # si geste détecté
```

## 📊 Structure du code

- `download_model()` - Télécharge le modèle MediaPipe
- `calculate_distance()` - Calcule la distance entre deux points
- `is_finger_extended()` - Vérifie si un doigt est tendu
- `detect_*()` - Fonctions de détection pour chaque geste
- `run_hand_tracking()` - Boucle principale de capture et détection

## 🐛 Dépannage

**La webcam ne s'ouvre pas ?**
- Vérifiez que votre webcam fonctionne
- Essayez de changer `cv2.VideoCapture(0)` en `cv2.VideoCapture(1)`

**Les gestes ne sont pas détectés ?**
- Assurez-vous d'avoir un bon éclairage
- Positionnez vos mains bien face à la caméra
- Augmentez les seuils de distance dans le code

**Performance lente ?**
- Fermez les autres applications utilisant la webcam
- Réduisez `num_hands` de 2 à 1 si vous n'avez besoin que d'une main

## 📝 Licence

Libre d'utilisation pour projets personnels et éducatifs.

## 🤝 Contribution

N'hésitez pas à ajouter de nouveaux gestes et partager vos améliorations !

---

Développé avec ❤️ et MediaPipe