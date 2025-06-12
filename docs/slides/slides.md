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

![bg](./assets/var-ai-szenario-intro.gif)

<style scoped>
header {
    color: #FFFFFF00;
}
</style>

<!--
Sicher kann sich noch jeder an die folgende Situation vor rund einem Jahr erinnern.

Im Viertelfinale der Europameisterschaft trifft die deutsche Nationalmannschaft auf
die spanische. Beim Spielstand von 1:1 kommt es dann in der Verlängerung zu dieser
Situation.

Ganz objektiv betrachtet liegt hier ein Handspiel vor. Der Schiedsrichter sieht dies
aber während des Spiels nicht & eine Wertung bleibt aus.

Wir haben uns die Frage gestellt ob man mit Maschinellem Lernen, also Objekterkennung &
Tracking, bei der Erkennung den Schiedsrichter unterstützen könnte.

Klar ist auch das es den VAR (Video-Assistant-Refree) gibt der sich solche Situationen
nachträglich anschaut und den Schiedsrichter unterstützt. Aber dennoch ist der Gedanke
interessant ein KI gestütztes System für so eine Aufgabe zu verwenden.

Das war die Idee unseres Projekt. Das erkennen eines Handspiels an genau diesem Szenario.

Ihr werdet uns heute bei unsere Umsetzung und den Workflow begleiten. 
-->

----

# Gliederung

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
Es können noch Bilder hinzugefügt werden, damit Folie etwas anschaulicher ist!

Mögliche Einteilung:
1. - 4. Chris
2. - 8. Denis

Dabei geben wir euch eine kurze Einführung zum genauen Ziel, gehen dann auf das
Labeln der Bilder ein die man für das darauffolgende Training bzw. Finetunen eines
YOLO-Models (You Only Look Once) benötigt.

Dazu werden wir auf die verwendeten Tools wie Google-Colab & Yolo eingehen.

Nach dem Training werden wir mithilfe von OpenCV auf das Balltracking
und das erfassen der Flugbahn eingehen.

Gegen Ende gehen wir auf die Umsetzung der Beziehung zwischen Ball und "Hand" ein
um zu zeigen wie die Handspielerkennung letzendlich funktiioniert.

Zum Schluss haben wir noch ein Fazit welches unsere Ergebnisse darstellt.
-->

---

# Einführung

![bg right:33% vertical](./assets/ship.gif)

* Erkennen eines Handspiels mithilfe von Objekterkennung
* Verbindung des VAR-Systems mit maschinellem Lernen
* Handspielerkennung: Bei signifikanten Änderungen der Flugbahn soll der Abstand zwischen Ball und *Hand* geprüft werden.
 
<!--
Dann kommen wir nun zu unserem ersten Punkt. Der Einführung.

Wir sind ohne praktische Vorerfahrung in so ein Objekterkennungs-/Tracking Projekt eingestiegen
und wussten also nicht genau was uns erwarten wird.

Wir wussten aber, dass wir ein Handspiel mithilfe von Objekterkennung erkennen wollten.

Der bereits erwähnte Gedanke dass man dadurch maschinelles Lernen mit dem VAR System verbinden könnte war recht interessant.
Also das Konzept möglicherweise auch in der Praxis anwenden zu können.

Also haben wir festgelegt wie ein grober Ablauf aussehen könnte:

Bei einer signifikanten Änderung der Flugbahn soll der Abstand zwischen Ball und "Hand" geprüft werden. Liegt dieser beim
Zeitpunkt der erwähnten Änderung unter einem festgelegten Wert, liegt ein Handspiel vor.

Welche Implikationen das nachher dann hat, also der bis jetzt noch nicht genauer festgelegte Abstandswert, werdet Ihr im
Laufe der Präsentation noch erfahren.
-->

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

<!--
Um überhaupt ersteinmal einen Ball erkennen zu können, benötigen wir eine Objekterkennung die
speziell darauf Trainiert ist, einen Ball zu erkennen.

Und um ein Model für die Objekterkennung des Balls trainieren zu können, benötigt man viele
Bilder von einem Ball.

Das heißt wir haben uns auf die Suche nach Bildern gemacht die in diesem Setting einen Ball darstellen.

Das war die Zusammenfassung des Spiels Deutschland gegen Spanien. Dort haben wir uns dann
verschiedene Szenen herausgesucht aus denen wir dann Bilder bzw. Frames extrahiert haben.

Dabei ist es auch wichtig das Bilder vorhanden sind in denen der Ball unscharf ist.
Denn wenn der Ball wie in unserem Beispiel stark beschleunigt wird, ist er auf den
einzelnen Bildern verschwommen und verzerrt. Das führt dazu, dass er nicht mehr als
Ball erkannt wird.

Wir haben das ganze mit verschiedenen Mengen an Bildern gemacht, kamen dann aber zum
Schluss auf 1408 Bilder die für ein Training für ein gutes Ergebnis notwendig waren.

Man muss aber dazu sagen, dass die Anzahl der benötigten Bilder variieren kann,
je nachdem was erkannt werden soll und wie viel in den Bildern dargestellt wird.

In unserem Szenario mit vielen Menschen im Hintergrund, war die genannte Anzahl
dann ausreichend.
-->
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

<!--
So, jetzt hat man die Bilder aus dem Video extrahiert die man fürs Training eines
Objekterkennungsmodels verwenden möchte.

Aber wie sagt man dem Model jetzt was & wo ein Ball auf einem Bild ist?

Dieser Prozess nennt sich Labeln. Dabei muss man festlegen an welcher Stelle im Bild (x- & y-Koordinate)
sich der Ball befindet. Hierfür haben wir Label-Studio verwendet. Es gibt aber auch reichlich andere
Tools die den gleichen oder ähnlichen Funktionsumfang bieten. Ich kann mich erinnern dass uns am Anfang
auch Makesens AI vorgeschlagen wurde.

Die Tools funktionieren aber alle eigentlich gleich.

Man definiert sich so ein Template (meistens geschieht dies aber in einer grafischen Benutzeroberfläche)
in dem man eine Boundingbox festlegt. Diese Boundingbox definiert dann die x/y Position des zu erkennenden
Objektes. 

Und diesen Prozess, also dass aufzeichnen der Boundingbox, wendet man dann auf 1408 Bilder an.
Also ich empfehle dass man sich gemütlich einen Kaffee macht und neben bei Musik hört, denn dafür wird eine
Menge Zeit draufgehen.

Hat man das dann geschafft, Exportiert man die Bilder sowie die Label im YOLO-Format. Das ist das Format
welches man für das Finetunen bzw. Training benötigt.
-->

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

<!--
Der nächste Schritt ist das Training des Objekterkennungsmodels.

Hierbei will ich kurz darauf eingehen, warum wir hierfür Google-Colab
verwendet habe.

Google-Colab für diejenigen die es nicht kennen ist eine "Cloud" Jupyter-Notebook-Umgebung
Das ist echt eine feine Sache, weil Colab gerade für das Training von großen Datensätzen
effiziente Hardware anbietet. Im speziellen GPUs & TPUs die wir lokal nicht zur Verfügung
hatten.

Um sich da mal ein Bild zu machen wie effizient das mit Leistungsstarker Hardware läuft habe ich
die Zeiten hier in der Tabelle dargestellt. Man sieht das mit meiner CPU das Training eines
Datensatzes mit einer größe von 800 Bildern und einer Image-Size 640 ca. 75 Minuten gedauert hat.
Ein fast doppelt so großer Datensatz mit der doppelten Image-Size hat in Colab mit der T4-GPU nur
ca. 30 Minuten gedauert. Mit dem Vorteil dass sich mein Zimmer im Mai nicht unnötig aufgeheizt hat.
Also man sieht schon, dass es wesentlich effizienter ist hier mit Colab zu arbeiten.

Was man auch noch erwähnen kann, ist dass man die Hardware kostenlos nutzen darf.
Das ist aber an die Einschränkung gebunden, dass die verfügbare Ressource nicht von
Premium Nutzern zur selben Zeit verwendet wird. Die haben natürlich Vorzug.

Kleiner Tipp:
Beim Training hat sich herausgestellt, dass man Sonntag abends weniger bis keine Beschränkungen hatte.
Montags jedoch wurde die Nutzung der effizienten Hardware grundsätzlich gesperrt.

Es gibt natürlich die Möglichkeit sich für, ich glaube 11€ im Monat, Colab-Premium zu abonnieren.
Dabei erhält man sogenannte Compute-Units. Das sind dann 100 an der Zahl die man im Laufe des Monats
verbrauchen kann.

Die Nutzung einer T4-GPU in der Stunde kostet glaube ich 0.4 Compute-Units.

Warum wir eine Image-Size von 1280 statt den Standartmäßigen 640 verwendet haben, darauf kommen wir
gleich noch zu sprechen.

Ein weiterer "Added Bonus" ist der, dass man bei Google-Colab auch gleich seinen Google-Drive Ordner
einbinden kann. Somit hat man persistenten Speicher auf dem man seine Datensätze & trainierte Modelle
abspeichern kann.
-->

<!--
Kann das nicht mehr reproduzieren:

# `imgsz`: 640 vs. 1280

**Pro 640**:
- Eine geringe Image-Size generalisiert besser
- Schnelleres Training da kleinere Daten

**Contra 640**:
- Bessere generalisierung führte zu geringerer Präzision
- Informationsverlust (Muster des Balls ging verloren)
-->

<!--
Ich ziehe jetzt hier schonmal einige Erkenntnisse aus dem Training vor, verzeiht mir also
wenn es Chronologisch nicht ganz stimmt.

Aber man stellt sich ja intuitiv die Frage warum wir eine größere Image-Size verwendet haben.

Wenn man sich das mit dem Wissen aus den ersten Vorlesungen betrachtet, bietet
eine geringere Auflösung den Vorteil einer besseren Generalisierung. Will man z.B. einen
Schuh erkennen, dann sind die Details eher weniger wichtig. Es kommt eher auf die Struktur
und die Kanten des Schuhs an.

Durch die geringere Auflösung hat man außerdem den Vorteil das die Daten kleiner sind
und man das Objekerkennungsmodel schneller trainieren kann.

Es hat sich aber herausgestellt, wie man anhand der Bilder erkennen kann, dass Objekte
im Hintergrund fälschlicher Weise als Ball erkannt wurden. Speziell das runde Muster
auf dem T-Shirt der Person im Hintergrund.

Durch das Erhöhen der Image-Size auf 1280 konnten wir diese Problem beheben und gehen
daher davon aus dass durch das Downscaling auf 640 Informationsverlust einhergeht und man somit
mit der Präzision Probleme bekommt. Man muss hierbei erwähnen dass die Bilder alle eine
Auflösung von 1920x1080 Pixel hatten. Also selbst bei der Verwendung von 1280 findet ein 
Informationsverlust statt.
-->

----

# ![height:40px](./assets/yolo.svg) YOLO11 Finetuning
 
<div class="columns">
<div>
 
- **Finetuning**:
    - Verwendetes Model: YOLO11s
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

<!--
Kommen wir nun zum Training beziehungsweise dem Finetuning des YOLO-Models.

YOLO also You Only Look Once, sind vorgewichtete Objekterkennungsmodelle die das Unternehmen
Ultralytics anbietet.

Vorgewichtet heißt hierbei, dass die Modele bereits ein Neuronales-Netzwerk besitzen welches
mithilfe von COCO Common Object in Context trainiert wurden.

Wir passen diese vorgewichteten Kanten lediglich mit unserem Datensatz an. Es besteht zwar die
Möglichkeit das Neuronale-Netz von grund auf zu Trainieren, dazu genügen aber 1408 Bilder nicht.

Bei dem Aufspalten der Trainings- & Validierungsdaten haben wir uns für ein Verhältnis von
80/20 entschieden. Also 80 Prozent des Datensatzes wird für das Training und 20 Prozent für 
das validieren verwendet.

Hier auch kurz noch ein Beispiel wie unsere Konfigurationsdatei für das Training aussieht:
-->

----

# ![height:40px](./assets/yolo.svg) YOLOv11 Finetuning in ![height:40px](./assets/googlecolab.svg) Google Colab

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
    imgsz=1280,
    batch=-1,
    device=device,
    freeze=10,
    lr0=0.005,
    project=OUTPUT_DIR,
    patience=10,
    epochs=60,
    cache=True,
    seed=42,
    plots=True,
    name=NAME)
```

</div>
</div>

<!--
Wie bereits erwähnt haben wir Colab für das Training verwendet, Dabei haben wir uns
auf das Model YOLO11s festgelegt. Das 's' in YOLO11s steht für small, gibt also
die größe des Models an. Es gibt hier auch noch n, l & x. Das s Model war aber 
der Sweetspot für uns. Die Präzision die wir damit erreichen konnten hat absolut
ausgereicht und auf unserer Hardware lief es während der Inference (kommt nach dem Training)
auch reibungslos.

Hier gehe ich noch kurz auf die Trainingsparameter ein:

Die Batchgröße von -1 führt dazu dass die passend zum Speicher der Hardware
die Batchgröße angepasst wird.

Die Bildgröße wird hier auch angegeben.

Das Freezing von -10 führt dazu dass die inneren 10 Schichten des Netzes bestehen
bleiben. Somit spart man an Trainingszeit, mit dem nachteil das die Genauigkeit sinkt,
Was aber wie sich herausstellte vernachlässigbare Auswirkungen hatte.

Man kann aber sagen dass die Werte durch probieren und lesen der YOLO Dokumentation
festgelegt wurden.

Das Early-Stopping hilft dabei das Training zu beenden, falls sich in neuen Trainingsepochen
kein Lernerfolg abzeichnet. Speziell wird hier nach 10 Epochen ohne Veränderung das Training
beendet.

Wichtig um die GPU für das Training zu verwenden: Model auf die GPU laden.
Das wird mit model.to(device) erreicht.
-->

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
 
<!--
Kommen wir zu den Ergebnissen des Trainings:

Man bekommt nach dem Training Statistiken zu der Performance des trainierten Models.

Hier haben wir die F1-Confidence-Kurve dargestellt. Man kann erkennen das unser
trainiertes Model am besten mit einer Confidence von 0.633 arbeitet.
Wobei man im Bereich von 0.4 bis 0.7 kaum bemerkbare Unerschiede wahrnimmt.
Bleibt man also in diesem Bereich erreicht man immer gute Ergebnisse bei der Objekterkennun.

Behaltet dieses Schaubild mal im Kopf, dass wird gleich noch interessant wenn wir uns die 
Ergebnisse in der Praxis anschauen.

Hier auch noch gezeigt wie man das Model zum testen anwenden kann:
-->
 
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
![bg vertical right:33% width:300px](./assets/var-ai-rdp-animation.gif)
![bg right:33% width:400px](./assets/var-ai-rdp-simplification.png)

# Vereinfachen der Flugbahn

- RDP-Algorithmus (Ramer-Douglas-Peucker)
- Einzelne Punkte der Flugbahn werden wegelassen sodass die grobe Struktur erhalten bleibt
- Wenn Punkte innerhalb des Abstands liegen, kann die Kurve durch eine einzige Linie ersetzt werden
- **Vorteil der Vereinfachung**:
    * Vergleich zw. Hand & Ball muss nur an 3 Punkten vorgenommen werden

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

# Vielen Dank für's zuhören!
 
<!-- paginate: false -->

<!--
Vielleicht noch etwas hinzufügen, Name und Bilder von uns. Wird am Ende der Präsentation immer gern gesehen.
-->
----

# Quellen

- Datensatz (1408 labled Bilder) [https://drive.google.com/drive/folders/1h9DUoNAdYnxkmZG605bA3RJbBkVvsU0-?usp=drive_link](https://drive.google.com/drive/folders/1h9DUoNAdYnxkmZG605bA3RJbBkVvsU0-?usp=drive_link)
- Fine-Tune Notebook (Google Colab) [https://colab.research.google.com/drive/1E4L6ZLPrKpIl-WGFn52tqyORbYHB6ulE?usp=sharing](https://colab.research.google.com/drive/1E4L6ZLPrKpIl-WGFn52tqyORbYHB6ulE?usp=sharing)
- VAR-AI GitHub Repository [https://github.com/denisthefirst/var-ai](https://github.com/denisthefirst/var-ai)
- Ultralytics YOLO11 [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)
- Ultralytics YOLO11 Pose-Estimation [https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-pose-estimation](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-pose-estimation)
- YOLO Fine-Tuning [https://docs.ultralytics.com/guides/model-evaluation-insights/](https://docs.ultralytics.com/guides/model-evaluation-insights/)
- YOLO Train-Settings [https://docs.ultralytics.com/modes/train/#train-settings](https://docs.ultralytics.com/modes/train/#train-settings)
- Ultralytics YOLO Logo [https://cdn.prod.website-files.com/646dd1f1a3703e451ba81ecc/64994922cf2a6385a4bf4489_UltralyticsYOLO_mark_blue.svg](https://cdn.prod.website-files.com/646dd1f1a3703e451ba81ecc/64994922cf2a6385a4bf4489_UltralyticsYOLO_mark_blue.svg)
- Label-Studio, Label-Studio Logo [https://labelstud.io/](https://labelstud.io/)
- Google Colab Logo [https://simpleicons.org/?q=colab](https://simpleicons.org/?q=colab)
- Videos, Bilder (Fußballszenen): [https://www.youtube.com/watch?v=GZXNfRqIQuo](https://www.youtube.com/watch?v=GZXNfRqIQuo)
- Präsentationsframework: [https://marp.app/](https://marp.app/)
- RDP-Algorithmus: [https://rdp.readthedocs.io/en/latest/](https://rdp.readthedocs.io/en/latest/)
- RDP-Algorithmus Animation: [https://rdp.readthedocs.io/en/latest/_images/rdp.gif](https://rdp.readthedocs.io/en/latest/_images/rdp.gif)

<!--
Bitte fehlende Quellen hinzufügen
-->
 
<style scoped>
li {
    font-size: 14px;
}
</style>
