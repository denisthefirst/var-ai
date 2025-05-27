----

header: VAR-AI | Data Science | Denis Maric & Chris Kiriakou 
paginate: false
style: |
  pre {
    font-size: 14px;
  }

----

# Video Assistant Referee - AI
## Data Science
Denis Maric & Chris Kiriakou

![bg cover](./assets/world.gif)

<style scoped>

h1 {
    font-size: 80px;
    text-align: center;
    padding: 10px;
    margin: 10px;
}

h2 {
    font-size: 50px;
    text-align: center;
    padding: 10px;
    margin: 10px;
}

section {
    text-align: center;
}

header {
    color: #FFFFFF00;
}

</style>

----

![bg](./assets/cucurella-hand-play.gif)

<style scoped>
header {
    color: #FFFFFF00;
}
</style>

<!--
Musiala und Cucurella Clip in groß über die gesammte Folie
-->

----

# Gliederung

![bg contain right:33%](/home/chris/pics/screenshots/risk-landscape.png)

1. Einführung
2. Datenlabelprozess mit Label-Studio
3. YOLO Finetuning in Google Colab
4. Balltracking mit OpenCV
5. Detektierung von Richtungsänderungen
6. Handspielerkennung
7. Herausforderungen
8. Fazit 

<!-- paginate: true -->

---

# Einführung

![bg right:33% vertical](./assets/ship.gif)

* Erkennen von Handspielen mithilfe von Objekterkennung
* Verbindung des VAR-System mit Mashinellen-Lernens
* Handspiel: Bei signifikanten Änderungen der Flugbahn soll der Abstand zwischen Ball und Hand geprüft werden.

----

# Vorbereitung der Daten
 
![bg right:33% vertical](./assets/example-scene-goal-wirtz.gif)
![bg right vertical](./assets/example-scene-yamal.gif)
![bg right vertical](./assets/example-scene-header.gif)

* Sammlung relevanter Videos (verschiedene Einstellungen, Lichtverhältnisse)
* Extrahieren einzelner Bilder aus Spielszenen
* Insgesammt 1408 Bilder für Labelprozess

<style scoped>
section:after {
    color: white;
    text-shadow: 
        -1px -1px 0 black,
        1px -1px 0 black,
        -1px  1px 0 black,
        1px  1px 0 black;
}
</style>

----
 
![bg right:45% vertical height:40%](./assets/label-studio-bounding-box.png)

# Labeln der Daten

- Verwendetes Tool: ![height:20px](./assets/label-studio-logo.png) [Label-Studio](https://labelstud.io/)
- Erstellen der Klasse `sports ball` 
 
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
  <Label value="sports ball" background="#FFA39E"/></RectangleLabels>
</View>
```

- **Auch** labeln verschwommener Bälle!
- YOLO-Format Export

----

![bg right:30% vertical height:50%](./assets/imgsz.svg)

# YOLO Finetuning in ![height:40px](./assets/googlecolab.svg) Google Colab (1/4)

**Warum**? GPU im Training signifikant schneller als CPU:

| Hardware                           | Zeit [min] |
|------------------------------------|------------|
| T4-GPU (1408 Bilder, `imgsz=1280`) | ~30        |
| CPU (800 Bilder, `imgsz=640`)      | ~75        |
 
Datensatz auf Google-Drive in Colab verwenden:
```python
from google.colab import drive
drive.mount('/content/drive')
```

----

# YOLO Finetuning in ![height:40px](./assets/googlecolab.svg) Google Colab (2/4)
 
- Split: 80/20 Traings-/Validierungsdaten 
  
Konfigurationsdatei für YOLO Datensatz:
```yaml
names:
- sports ball
nc: 1
path: /content/dataset
train: images/train
val: images/val
```
----

# YOLO Finetuning in ![height:40px](./assets/googlecolab.svg) Google Colab (3/4)

Traingsparameter:
- YOLO Model: `yolo11s.pt`
- Trainingszeit [h] (passt anz. Epochen an): `4.5`
- Early-Stopping: `patience = 10`
- Gewichtungsanpassungen in Memory: `cache = True`
- Bildgröße, höher = weniger Informationsverlust: `imgsz = 1280`
- Learningrate: `lr0 = 0.005`
- Passende Batch-Größe für geg. Hardware auswählen `batch = -1`
- Innere Layer "einfrieren": `freeze = 10`

----

# YOLO Finetuning in ![height:40px](./assets/googlecolab.svg) Google Colab (4/4)

Model auf GPU trainieren:
```python 
import torch
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else exit)
model = YOLO(MODEL)
model = model.to(device)
```
<!--
Trainingsergebnisse (Plots) einfügen Trainings- & Validierungs-Batch (mit Label)
-->

----

<!--
Testen des Models am Szenario Clip (ohne Confidence) - Erkenntnis: Andere Objekte werden auch als Ball erkannt
-->

----

# Inference mit Confidence

Testen der Objekterkennung an unserem Szenario:
```
yolo task=detect mode=predict model=./models/ball-detection.pt source=./data/raw/videos/cucurella-hand-play.mp4 conf=0.7
```
* Confidence: `0.7` entnommen aus F1-Score
* Erkennung des Balls bei 90% aller validierten Daten mit einer Genauigkeit von 70%

<!--
Inference mit Musiala - Cucurella Clip
F1-Score Plot aufzeigen & Anpassung der Confidence
-->

----

<!--
Testen des Models am Szenario Clip (mit Confidence) - Erkenntnis: Nur Ball wird erkannt
-->

----

# Tracking mit OpenCV: Prüfen nach erkannten Objekten
 
Über alle Frames des Videos wird iteriert & Geprüft ob Objekte erkannt wurden:
```python
# Ret is boolean value, frame is an actual image
ret, frame = cap.read() 
# Store tracking results 
results = model.track(frame, conf=0.7, persist=True) 
out.write(frame) 
# Checks if there are any boxes saved
if results[0].boxes is not None: 
    boxes = results[0].boxes 
    if hasattr(boxes, 'cls'): 
        class_names = results[0].names 
        for i, cls_id in enumerate(boxes.cls): 
            cls_id = int(cls_id) 
            class_name = class_names[cls_id] 
```

----

# Tracking mit OpenCV: Erfassung der Position
 
Enthält der Erkannte Frame eine Klasse `sports ball`, so wird...
- die Position erfasst
- der Mittelpunkt des Balls berechnet
```python
if class_name == 'sports ball': 
    box = boxes.xyxy[i].cpu().numpy() 
    x1, y1, x2, y2 = box 
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) 
    center_x = int((x1 + x2) / 2) 
    center_y = int((y1 + y2) / 2)
    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1) 
    ball_coordinates.append([frame_number, center_x, center_y])
    track_id = 0  # Default value
    if hasattr(boxes, 'id') and boxes.id is not None:
        try:
            track_id = int(boxes.id[i])  # i-th ID
        except (IndexError, TypeError, ValueError):
            track_id = 0
```

----

# Tracking mit OpenCV: Speichern der Ballflugbahn 

- Speichern der einzelnen Punkte in in einer Flugbahn Liste
- Aufzeichnen der Flugbahn
```python
# Add point to trajectory
if track_id not in ball_trajectories:
    ball_trajectories[track_id] = []
ball_trajectories[track_id].append((center_x, center_y))
# Draw trajectory line
if len(ball_trajectories[track_id]) > 1:
    for i in range(1, len(ball_trajectories[track_id])):
        # Draw line from previous point to current point
        cv2.line(frame, 
            ball_trajectories[track_id][i-1], 
            ball_trajectories[track_id][i], 
            (255, 0, 0), 2)
```
----

# Fazit

- Unternehmen müssen flexibel auf neue Herausforderungen reagieren
- Alternative Transportwege und diversifizierte Lieferanten sind entscheidend
- Gesetzliche Rahmenbedingungen erfordern schnelle Anpassungen
- Nachhaltige Innovationen stärken langfristig die Widerstandsfähigkeit globaler Lieferketten

----

# Vielen Dank für's zuhören!
 
<!-- paginate: false -->

----

# Quellen

*
