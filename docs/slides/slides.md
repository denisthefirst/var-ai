----

header: VAR-AI | Data Science | Denis Maric & Chris Kiriakou 
paginate: false
style: |
  pre {
    font-size: 16px;
  }
  code {
    font-size: 22px;
  }
  li, p, td, th {
    font-size: 26px;
  }
  .columns {
    display: flex;
    gap: 1rem;
  }
  .columns > div {
    flex: 1 1 0;
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
2. Labeln mit Label-Studio
3. Google Colab
4. YOLO Finetuning
5. Ballerfassung mit OpenCV
6. Detektierung der Flugbahnänderung
7. Handspielerkennung
8. Fazit 

<!-- paginate: true -->
<!--
Notizen:
-->
---

# Einführung

![bg right:33% vertical](./assets/ship.gif)

* Erkennen eines Handspiels mithilfe von Objekterkennung
* Verbindung des VAR-Systems mit maschinellem Lernen
* Handspielerkennung: Bei signifikanten Änderungen der Flugbahn soll der Abstand zwischen Ball und *Hand* geprüft werden.

----

# Vorbereitungen für Labelprozess
 
![bg right:33% vertical](./assets/example-scene-goal-wirtz.gif)
![bg right vertical](./assets/example-scene-yamal.gif)
![bg right vertical](./assets/example-scene-header.gif)

* Sammlung relevanter Videos (verschiedene Einstellungen, Lichtverhältnisse)
* Extrahieren einzelner Bilder aus Spielszenen (Videos)
* Insgesamt 1408 Bilder für Labelprozess

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

# Labeln der Bilder

- Verwendetes Tool: ![height:20px](./assets/label-studio-logo.png) [Label-Studio](https://labelstud.io/)
- Erstellen der Klasse `sports ball` 
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
  <Label value="sports ball" background="#FFA39E"/></RectangleLabels>
</View>
```
- **Auch** Labeln unscharfer Bälle!
- Export im YOLO-Format

----

![bg right:27% vertical height:40%](./assets/imgsz.svg)

# ![height:40px](./assets/googlecolab.svg) Google Colab
 
**Warum Colab?** GPU im Training signifikant schneller als CPU

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

# ![height:40px](./assets/yolo.svg) YOLOv11 Finetuning
 
<div class="columns">
<div>
 
- **Finetuning**:
    - Verwendetes Model: YOLOv11s
    - Model besitzt bereits vortrainierte Gewichtungen (COCO Datensatz)
    - Gewichtungen mit eigenem Datensatz anpassen
- Split: 80/20 Traings-/Validierungsdaten 
  
</div>
<div>

Konfigurationsdatei für YOLO Datensatz:
```yaml
names:
- sports ball
nc: 1
path: /content/dataset
train: images/train
val: images/val
```

</div>
</div>

----

# Finetuning

<div class="columns">
<div>

Trainingsparameter:
- **YOLO-Model**: `yolo11s.pt`
- **Early-Stopping**: `patience = 10`
- **Bildgröße**: größer = weniger Informationsverlust `imgsz = 1280`
- **Batchgröße**: Passende für geg. Hardware auswählen `batch = -1`
- **Freezing**: Hidden-Layer *einfrieren* `freeze = 10`
 
</div>
<div>

Model auf GPU trainieren:
```python 
import torch
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else exit)

model = YOLO(MODEL)
model = model.to(device)
model.train(
    data=DATASET_CONFIGURATION_PATH,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=device,
    freeze=FREEZE,
    lr0=LEARNING_RATE,
    project=OUTPUT_DIR,
    patience=PATIENCE,
    epochs=EPOCHS,
    cache=CACHE,
    seed=SEED,
    plots=PLOTS,
    name=NAME)
```

</div>
</div>

----

![bg right:50% height:50%](./assets/var-ai-f1-curve.png)

# Confidence

Testen der Objekterkennung an unserem Szenario:
```
yolo task=detect \
    mode=predict \
    model=./models/ball-detection.pt \
    source=./data/raw/videos/cucurella-hand-play.mp4 \
    conf=0.663
```
* **Confidence**: `0.663`, hier arbeitet das Model am besten
* Im Bereich von `0.4` - `0.7` kaum Unterschied bemerkbar
 
----

![bg vertical right:33% height:90%](./assets/var-ai-object-detection-conf-5.gif)
![bg right:33% height:90%](./assets/var-ai-object-detection-conf-40.gif)
![bg right:33% height:90%](./assets/var-ai-object-detection-conf-90.gif)

# F1-Score Relevanz

- Bei einer Confidence von 66,3% ist der F1-Score am höchsten
- **F1-Score**:
    - Precision: Wie viele der erkannten Bälle **waren wirklich Bälle**? 
    - Recall: Wie viele der tatsächlich **vorhandenen Bälle** wurden erkannt?

$$F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

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

<!--
Man sieht dass man bei einer zu niedrigen Confidence z.B. 0.05 Probleme mit der Genauigkeit bekommt
(es werden Bäller erkannt wo keine sind). Bei einem zu hohen Confidence-Wert, wie es bei 0.9 zu sehen ist, werden
keine Bälle mehr erkannt.
-->

----

# Tracking mit OpenCV

<div class="columns">
<div>

Objekterkennung mit Finetuned YOLO Model:
- Über alle Frames des Videos iterieren
- Prüfen ob Objekte erkannt wurden
 
<br>

Enthält der erkannte Frame `sports ball`:
- Erfassen der Position mit OpenCV
- Berechnen des Ballmittelpunkts
- Speichern der Koordinaten in Liste

</div>
<div>

```python
results = model.track(frame, conf=0.4, persist=True) 
if results[0].boxes is not None: 
    boxes = results[0].boxes 
    if hasattr(boxes, 'cls'): 
        class_names = results[0].names 
        for i, cls_id in enumerate(boxes.cls): 
            cls_id = int(cls_id) 
            class_name = class_names[cls_id] 
```
```python
if class_name == 'sports ball': 
    box = boxes.xyxy[i].cpu().numpy() 
    x1, y1, x2, y2 = box 
    center_x = int((x1 + x2) / 2) 
    center_y = int((y1 + y2) / 2)
    coord_at_frame = [frame_number, center_x, center_y]
    ball_coordinates.append(coord_at_frame)
    track_id = 0  # Default value
    if hasattr(boxes, 'id') and boxes.id is not None:
        try:
            track_id = int(boxes.id[i])  # i-th ID
        except (IndexError, TypeError, ValueError):
            track_id = 0
```

</div>
</div>

----

![](./assets/var-ai-opencv-tracking.gif)

<!--
Gif mit Box Flugbahn
-->

----

![bg right:35% width:400px](./assets/var-ai-points-between-frames-combined.png)

# Ausreißer entfernen

Alle Punkte die innerhalb von 5 Bildern eine euklidische Distanz von > 50px haben, werden verworfen:
```python
frame_value_i = cords_file.iloc[i]['frame']
x_previous = cords_file.loc[cords_file['frame'] == frame_value_i, 'x'].values
y_previous = cords_file.loc[cords_file['frame'] == frame_value_i, 'y'].values

frame_value_j = cords_file.iloc[j]['frame']
x_current = cords_file.loc[cords_file['frame'] == frame_value_j, 'x'].values
y_current = cords_file.loc[cords_file['frame'] == frame_value_j, 'y'].values

previous = np.array([x_previous, y_previous])
current = np.array([x_current, y_current])

distances = np.linalg.norm(previous - current)

if(abs(frame_value_i - frame_value_j) < 10 and distances > 50):
    cords_file = cords_file.drop(cords_file.index[j])
    cords_file = cords_file.reset_index(drop=True)
else:
    i += 1
```

<!--
Schaubild der gefilterten Bildpunkte im Koordinatensystem anzeigen
-->

----

![bg right:33% width:400px](./assets/var-ai-rdp-simplification.png)

# Vereinfachen der Flugbahn

- RDP-Algorithmus (Ramer-Douglas-Peucker)
- Einzelne Punkte der Flugbahn werden wegelassen sodass die gorbe Struktur erhalten bleibt
- Wenn Punkte innerhalb des Abstands liegen, kann die Kurve durch eine einzige Linie ersetzt werden
- **Vorteile der Vereinfachung**:
    - Keine aufwendige Berechnung der Winkel zwischen den Geraden!
    - Vergleich zw. Hand & Ball muss nur an 3 Punkten vorgenommen werden

<!--
Schaubild der Vereinfachung

- `epsilon=10` gibt an wie stark eine Linie vereinfacht werden darf
 
```python
from rdp import rdp

points = processed_cords_file[['x', 'y']].values
frames = processed_cords_file['frame'].values

simplified_rdp = rdp(points, epsilon=10)
```
-->

----

# Pose-Estimation: Handgelenk & Ellbogen erfassen

<div class="columns">
<div>

- Definieren relevanter Keypoints
- Erfassen & Speichern der Keypoints für späteren Vergleich

![center](./assets/pose-keypoints.png)
 
</div>
<div>

```python
keypoint_indices = {
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,  
    'right_wrist': 10
}
```

```python
for person_idx, person_kpts in enumerate(keypoints):
    for part_name, part_idx in keypoint_indices.items():
        if part_idx < len(person_kpts):
            x, y, conf = person_kpts[part_idx].tolist()
            
            all_frames_data.append({
                'frame': frame_idx,
                'person_id': person_idx,
                'body_part': part_name,
                'x': x,
                'y': y,
                'confidence': conf
            })
```

</div>
</div>

<!--
-->

----

# Vergleich Keypoint- & Ballposition

Jede Euklidische-Distanz die weniger als 30px bei einer der signifikanten Flugbahnabweicheung beträgt, wird als Handspiel erkannt:
```python
ref_row = trajectory_changes_df[trajectory_changes_df['frame'] == frame][['x', 'y']].iloc[0]
ref_point = np.array([ref_row['x'], ref_row['y']])
frame_points = hand_position_df_filtered[hand_position_df_filtered['frame'] == frame][['x', 'y']]
for _, row in frame_points.iterrows():
    point = np.array([row['x'], row['y']])
    distance = np.linalg.norm(point - ref_point)
    
    if distance <= tolerance:
        results.append({
            'frame': frame,
            'hand_x': row['x'],
            'hand_y': row['y'],
            'trajectory_x': ref_point[0],
            'trajectory_y': ref_point[1],
            'distance': distance
        })
```
----

![](./assets/var-ai-handplay-detected.jpg)

----

# Fazit

* Beispielszenario funktioniert gut
* Schwierigkeiten bei bewegter Kameraführung 
* Handspiel ist auch immer Ermessenssache des Schiedsrichters
* Beabsichtigtes/Unbeabsichtigtes Handspiel
* Erweiterungen:
    * Entfernung Arm zu Körper
    * Analyse ob ein absichtliches Handspiel vorliegt (Berücksichtigung des Kontextes)

----

> Considering the contact with the ball happened as he was moving his hand toward his body (which was in a vertical position), and he wasn’t deliberately trying to make his body unnaturally bigger, it indicated that Cucurella did not intend to stop the ball with his hand. Hence, due to the vertical position of his hand, the penalty wasn’t awarded.

----

# Vielen Dank für's zuhören!
 
<!-- paginate: false -->

----

# Quellen

*
