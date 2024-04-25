# Automatisierte Erkennung von Fluidbewegungen mittels KI


## Kurzfassung
Wir erforschen eine Methode zur Automatisierung der Analyse des Medikamententransports und konzentrieren uns dabei auf die Erkennung von Vibrationsschäden. 
Die Experimente der Roche Pharma (Schweiz) AG, die mit Hochgeschwindigkeitskameras aufgezeichnet werden, zielen darauf ab, die Transportbedingungen von flüssigen Medikamenten zu optimieren. 
Wir verwenden ein Convolutional Neural Network (CNN), das auf selbst aufgenommenen und gelabellte Daten aus dem Labor der ZHAW trainiert wurde. Wir führen Data augmentation und Balancing durch,
um die Einschränkungen des kleinen Datensatzes auszugleichen. Es werden verschiedene Hyperparameter getestet, um die Leistung des Modells zu verbessern. Unser bestes Modell erreicht einen 
96% FBeta-Score für die Erkennung von Flüssigkeitsschwingungen.
Wir haben festgestellt, dass Anpassungen der Filmbedingungen die Performance beeinträchtigen, wie z.B. durch einer anderen Beleuchtung. Ein grösseres, vielfältigeres Dataset
kann die Genauigkeit und die Verwendbarkeit des Modells erhöhen.

## 