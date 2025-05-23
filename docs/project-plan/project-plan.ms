.nr PI 2n \" Set indentation for bullet lists aka. PI to 2n
.TL
Projektplan: VAR-AI (Videoassistant Referee Artificial Inteligence),
Handspielerkennung durch Objekterkennung & Maschinelles Lernen
.AU
Denis Maric (212457)
Chris Kiriakou (209285)
.AI
Datascience Im Unternehmenskontext \[-] Hochschule Heilbronn
.AB
Ziel dieses Projekts ist die Entwicklung eines KI-gestützten Assistenzsystems
zur Unterstützung des Video Assistant Referee (VAR) im Fußball, speziell zur
Erkennung von potenziellen Handspielen. Mithilfe von Objekterkennung und
maschinellem Lernen soll ein Modell trainiert werden, das Spielszenen analysiert
und erkennt, ob ein Handspiel vorliegt.

Im folgenden Projektplan werden die Projektphasen, die technische Umsetzung
sowie Rollen und Zuständigkeiten dargestellt
.AE
.bp

.NH
Projektziel
.PP
Das System soll automatisch potenzielle Handspiele in Videosequenzen erkennen.
Ein Handspiel liegt vor, wenn der Ball nachweislich seine Flugbahn verändert und
dabei Kontakt mit einer Handfläche oder einem Arm hat.

.LP
Die Analyse & Erkennung basiert auf folgenden Schritten:
.IP \[bu]
.B Verfolgung
der Ballflugbahn (OpenCV)
.IP \[bu]
.B Objekterkennung
von Ball und Hand (YOLO)
.IP \[bu]
.B Vergleich
zwischen erwarteter und tatsächlicher Flugbahn.

.LP
Bei signifikanten Abweichungen der kalkulierten und ermittelten Ballflugbahn,
wird die Objekterkennung aktiviert. Hierbei wird die Position der nächsten Hand
sowie des Balls erfasst und deren Abstand zum Zeitpunkt der signifikanten Abweichung
berechnet. Ist Abstand zwischen Hand und Ball unter ein Minimum, so liegt ein
Handspiel vor.

.NH
Projektphasen
.LP
Die Gesamtkonzeption kann in die folgenden Phasen gegliedert werden.
.NH 2
Datensammlung
.IP \[bu]
.B "Ziel" :
Sammlung relevanter Videodaten von Spielszenen.
.IP \[bu]
.B "Quellen" :
SoccerNet (500+ Spiele, Fokus auf Replays/Nahaufnahmen),
Youtube (z. B. Sportschau-Zusammenfassungen).
.IP \[bu]
.B "Tools" :
yt-dlp, SoccerNetDownloader (Python)
.NH 2
Datenaufbereitung
.IP \[bu]
.B "Ziel" :
Vorbereitung der Daten für das Modelltraining.
.IP \[bu]
.B "Schritte" :
.RS
.IP \[bu]
Extrahieren von Einzelbildern aus Videomaterial.
.IP \[bu]
Labeln der Objekte (Ball, Hand, ggf. Arm) mittels Bounding Boxes.
.IP \[bu]
Tool: Label Studio.
.RE
.NH 2
Bildanalyse
.IP \[bu]
.B "Ziel" :
Berechnung der Ballflugbahn.
.IP \[bu]
.B "Methode" :
.RS
.IP \[bu]
Anwendung linearer Regression auf die Ballpositionen in aufeinanderfolgenden
Frames.
.IP \[bu]
Identifikation abrupter Flugbahnabweichungen.
.IP \[bu]
Optional: Analyse der Körperhaltung (abgestreckter Arm).
.RE
.IP \[bu]
.B "Tool" :
OpenCV
.PS
arrowhead = 0
box "Ball-" "tracking"
arrow
box "Validierung" "Flugkurve"
arrow
Obj_Det: box "Objekt-" "erkennung"
Pose: move right 1.2; box wid 1.2i "Körperhaltung" "(abgestreckter Arm)" dashed
arrow from Obj_Det.e to Pose.w dashed "optional" above
.PE
.bp
.NH 2
Training
.IP \[bu]
.B "Ziel" :
Extrahieren der Daten aus einem YOLO-Model, zusätzliches eigenes Erstellen von gelabelten Daten.
.IP \[bu]
.B "Vorgehen" :
.RS
.IP \[bu]
Import der gelabelten Bilder.
.IP \[bu]
Fokus auf Erkennung von Ball und Hand.
.IP \[bu]
Ballverfolgung über Bildsequenzen hinweg zur Flugbahnermittlung.
.RE
.NH 2
Evaluation
.IP \[bu]
.B "Ziel" :
Validierung des Modells und Optimierung.
.IP \[bu]
.B "Methode" :
.RS
.IP \[bu]
Verwendung von ca. 20 % der Trainingsdaten als Validierungsdaten.
.IP \[bu]
Inferenz auf neuen Clips, Abgleich mit manuell gelabelten Daten
.IP \[bu]
Anpassung des Trainings bei Bedarf.
.RE
.NH 2
Abschluss
.IP \[bu]
.B "Ziel" :
Zusammenfassung und Präsentation der Ergebnisse.
.IP \[bu]
.B "Aktivitäten" :
.RS
.IP \[bu]
Auswertung der Modellergebnisse.
.IP \[bu]
Dokumentation des Workflows.
.IP \[bu]
Vorbereitung der Abschlusspräsentation.
.RE
.NH
Zeitplan
.LP
.TS
center box tab (:);
cb cb cb
l | c | l.
Verantwortlichkeitsbereich:Zeitraum:Person
_
Projektplanung:23.05 \[-] 08.04:Denis Maric
Datensammlung:01.04 \[-] 15.04:Chris Kiriakou
Datenaufbereitung:08.04 \[-] 22.04:Chris Kiriakou
Bildanalyse:15.04 \[-] 29.04:Denis Maric
Training:22.04 \[-] 20.05:Denis Maric
Evaluation:29.04 \[-] 03.06:Chris Kiriakou
Abschluss:03.06 \[-] 16.06:Denis Maric, Chris Kiriakou
.TE
.PP
.B "Hinweis" :
Die benannte Zuständige Person führt jeweils die Hauptverantwortung in dieser
Phase, Teamarbeit und Austausch finden übergreifend statt. Die
Verantwortlichkeit schließt die andere Person natürlich
.B "nicht"
von der jeweiligen Phase aus.
.NH
Bewertung & Erfolgskriterien
.IP \[bu]
.B "Metriken" :
Precision, Recall, F1-Score zur Bewertung der Modellgenauigkeit.
.IP \[bu]
.B "Zielwert" :
Mindestens 80 % Genauigkeit bei der Handspielerkennung.
.IP \[bu]
.B "Testdaten" :
Separate Spielszenen mit verifizierten Handspiel-Vorfällen.
.NH
Risiken & Herausforderungen
.IP \[bu]
.B "Datenqualität" :
Schlechte Videoauflösung oder verdeckte Hände können die Genauigkeit mindern.
.IP \[bu]
.B "Labeling-Aufwand" :
Hoher manueller Aufwand zur präzisen Annotation.
.IP \[bu]
.B "Finden geeigneter Daten" :
Situationen mit Handspiel sind eher selten, mühsames heraussuchen aus großen 
Datenmengen (SoccerNet).
.NH
Ausbilck
.LP
Bei Fortführung des Projekts könnte ebenfalls die Körperhaltung (abgestreckter
Arm) analysiert werden um noch genauer in der Auswertung zu werden.
.NH
Systemstruktur
.PP
Nachfolgenden die Systemstruktur des Projekts. Die einzelnen Komponenten
wurden grob skizziert um den Ablauf darzustellen:
.PS
pad = 0.3;
Data_Source: [
    box wid 1.2 "Youtube" "(einzelne Clips)"
    move right 0.2
    box "SoccerNet"
]
box radius 0.1 dashed \
    ht last [].ht+pad wid last [].wid+pad at last []
move up 0.15 from last [].nw; "Datenquellen (extern)" above ljust
Soc_Net: move left 0.2 from last [].se
Yt: move right 0.2 from last [].sw

move down 0.8 from Data_Source.s; Scraper: [
    move left 0.8; box "yt-dlp"
    move right 0.8; box wid 1.4 "SoccerNetDownloader"
]
Scraper_Sys: box radius 0.1 dashed \
    ht last [].ht+pad wid last [].wid+pad at last []
move down 0.15 from last [].sw; "Scraper" below ljust
Soc_Net_Dl: move left 0.2 from Scraper.ne
Yt_Dlp: move right 0.2 from Scraper.nw

move right 1 from Scraper.e; Storage: [
    box "Bilder" "(unlabled)" 
]
Storage_Sys: box radius 0.1 dashed \
    ht last [].ht+pad wid last [].wid+pad at last []
move up 0.15 from last [].n; "Storage (e.g. cloud)" above

move down 1 left 2 from Storage.s; Label_Studio: [
    box "Bilder" "(labled)" 
]
Label_Studio_Sys: box radius 0.1 dashed \
    ht last [].ht+pad wid last [].wid+pad at last []
move down 0.15 from last [].s; "Label Studio" below

move down right 1.5 from Label_Studio.e; Training: [
    box "YOLOv8" "Hand & Ball" 
]
Training_Sys: box radius 0.1 dashed \
    ht last [].ht+pad wid last [].wid+pad at last []
move up 0.15 from last [].n; "Training" above

move down 1 left 0.1 from Training.sw; Eval: [
    Model: box wid 1.4 "Trainiertes Model"
    move right 0.2 from Model.e
    Pred: box "Vorhersage"
    move right 0.2 from Pred.e
    Prob: box wid 1.2 "Wahrscheinligkeit" "(in %)"
    arrow from Model.e to Pred.w
    arrow from Pred.e to Prob.w
]
Eval_Sys: box radius 0.1 dashed \
    ht last [].ht+pad wid last [].wid+pad at last []
move down 0.15 from last [].sw; "Evaluation" below ljust
Model_Nw: move right 0.5 from Eval.nw
Model_Sw: move right 1 from Eval.sw

move down 0.5 from Eval_Sys.s
Comp: box radius 0.1 "Bilder" "(labled)"
move left 0.1 from Comp.w
Val: box radius 0.1 "Val. Bilder" "ca. ~20%"

arrow from Soc_Net_Dl to Soc_Net
arrow from Yt_Dlp to Yt
arrow from Scraper_Sys.e to Storage_Sys.w
spline from Storage_Sys.s down 0.2 then left 2 to Label_Studio_Sys.n ->
arrow from Label_Studio_Sys.e to Training_Sys.w
spline from Training_Sys.sw left 2.5 to Model_Nw ->
spline from Eval.e right to Training_Sys.s -> "wenn % < 20%" ljust
arrow from Val.n to Model_Sw
line <-> from Comp.n to Eval.s "vergleich"
.PE
