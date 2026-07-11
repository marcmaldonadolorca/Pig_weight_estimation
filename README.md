# Pig weight estimation from depth images

Estimating pig weight **without a scale**, from overhead depth images — my BSc
thesis in Computer Engineering (UAB, grade 9.2/10), built with a real farm's
data for a smart-farming use case: pigs should reach slaughter weight (~120 kg)
as precisely as possible, and weighing them by hand is slow and stressful for
the animals.

## Results

| Stage | Metric | Value |
|---|---|---|
| Segmentation (U-Net, InceptionV3 backbone) | IoU / accuracy | **0.98 / 0.99** |
| Weight regression (CNN on segmented depth) | MAE (validation) | **3.6 kg** |
| Baseline (features + linear regression) | MAE | 6.43 kg |
| Size-group classification | accuracy | 84 % |

3.6 kg of mean absolute error with only **~600 samples** — the gain came from
data treatment, not model size: background removal, head exclusion, and
augmentation mattered more than architecture.

## Pipeline

1. **Data preparation** — depth + infrared images matched to scale readings
   (pig ID, weight, timestamp).
2. **Segmentation** — classic morphology (Otsu) vs **YOLOv5** detection vs
   **U-Net** semantic segmentation; U-Net won.
3. **3D processing** — point clouds and meshes with **Open3D**.
4. **Regression** — feature-based linear baseline, then CNN regression on the
   segmented depth images.

## Repo layout

```text
src/data/            dataset generation (YOLO labels, weights, sizes)
src/model/           segmentation + regression + classification training
src/visualization/   growth-evolution plots
reports/             thesis reports (final report: reports/informe_final/)
```

Full write-up (Spanish): [`reports/informe_final/informe_final.pdf`](reports/informe_final/informe_final.pdf)

## Stack

Python · TensorFlow/Keras · OpenCV · YOLOv5 · Open3D

## Limitations

Farm dataset is **not redistributable** (commercial provenance), so the repo is
code + reports only. Validation is a single split (no cross-validation — noted
in the report), and the 2021-era dependencies in `requirements.txt` reflect the
project's date.
