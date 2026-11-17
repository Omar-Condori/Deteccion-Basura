# Revisión de fuga de datos

Dataset: `datasets/waste` · imágenes: train=12390, val=1539, test=1654

- Duplicados exactos (MD5) en splits distintos: **1** grupos
- Mismo original (antes de `.rf.`) en splits distintos: **0**
- Casi-duplicados (dHash, Hamming ≤ 4) entre splits distintos: **46555** pares {'test/train': 23478, 'train/val': 20218, 'test/val': 2859}
- Imágenes de val/test con un casi-duplicado en train: **1674** (52.4 % de val+test)

Pares completos en `leakage_pairs.csv`.

## Verificación manual (nota)
dHash compara la apariencia global de la imagen, así que marca falsos positivos cuando distintos objetos se fotografían con el
mismo fondo y cámara (p. ej. la cinta verde de `plastic_bottle`). Se revisaron visualmente pares de `plastic_bottle` con Hamming 1 y 3:
eran **objetos distintos sobre la misma cinta**, no la misma foto. Por tanto la cifra de pares es una cota superior, pero confirma
**similitud de escena/dominio** entre train y val/test: las métricas de val/test miden el rendimiento en ese mismo montaje y
pueden ser optimistas fuera de él. Duplicados exactos entre splits: 1 grupo (MD5).
