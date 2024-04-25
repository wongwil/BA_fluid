# Automatisierte Erkennung von Fluidbewegungen mittels KI (ZHAW School of Engineering)
Sven Aebersold und William Wong

## Kurzfassung
Wir erforschen eine Methode zur Automatisierung der Analyse des Medikamententransports und konzentrieren uns dabei auf die Erkennung von Vibrationsschäden. 
Die Experimente der Roche Pharma (Schweiz) AG, die mit Hochgeschwindigkeitskameras aufgezeichnet werden, zielen darauf ab, die Transportbedingungen von flüssigen Medikamenten zu optimieren. Ein grosser Teil der Arbeit besteht darin, ein Dataset zu erstellen, sodass ein Model trainiert und getestet werden kann. Dazu gehört das Aufnehmen der Daten im Labor der ZHAW und das Labeln der Daten.
Wir verwenden ein Convolutional Neural Network (CNN), das auf unser Dataset trainiert wurde. Wir führen Data augmentation und Balancing durch,
um die Einschränkungen des kleinen Datensatzes auszugleichen. Es werden verschiedene Hyperparameter getestet, um die Leistung des Modells zu verbessern. Unser bestes Modell erreicht einen 96% FBeta-Score für die Erkennung von Flüssigkeitsschwingungen.
Wir haben festgestellt, dass Anpassungen der Filmbedingungen die Performance beeinträchtigen, wie z.B. durch einer anderen Beleuchtung. Ein grösseres, vielfältigeres Dataset
kann die Genauigkeit und die Verwendbarkeit des Modells erhöhen.
![image](https://github.com/wongwil/BA_fluid/assets/11984597/8b21006e-d9cd-489f-9f5f-c81c269cc48a)

## Reproduktion der Experimente
Bitte stellen Sie sicher, dass alle aufgelisteten Packete in [server_installation.txt](https://github.com/wongwil/BA_fluid/blob/main/server_installation.txt) installiert sind. Desweiteren müssen natürlich alle Pfade, die auf die Datasets oder anderen Files zugreifen, angepasst werden.

- [keras_mode.py](https://github.com/wongwil/BA_fluid/blob/main/project/keras_model.py): Training des Models.  Aufgrund der Grösse haben wir unsere Models nicht im Repository, deshalb bitten wir Sie diese selbst trainiert mit dem File zu trainieren.
- [mainwindow.py](https://github.com/wongwil/BA_fluid/blob/main/project/mainwindow.py): GUI für die Nutzung des Models.

![image](https://github.com/wongwil/BA_fluid/assets/11984597/2fe7dbab-f68c-45e0-a34d-a15c89a4d40b)

1. Laden eines gewünschten Videos und Start der automatischen Analyse
2. Speichern und Wiederladen von Analysen als CSV-File
3. Navigierung im Video
4. Übersicht der Verteilung aller Klassen aus der Analyse
5. Für jedes Frame wird die vorhergesagte Klasse angezeigt, womit man auch navigieren kann

