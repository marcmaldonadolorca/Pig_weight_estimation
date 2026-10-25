# Pig weight estimation from depth images

Estimating pig weight **without a scale**, from overhead depth images. This was my BSc
thesis in Computer Engineering (UAB, grade 9.2/10), built with a real farm's data for a
smart-farming use case: pigs should reach slaughter weight (~120 kg) as precisely as possible,
and weighing them by hand is slow and stressful for the animals.

![From an overhead depth frame to a 3D surface: YOLOv5 detection, U-Net segmentation, Open3D point cloud and Poisson mesh](docs/img/pipeline.png)

*The full pipeline on one real animal: the farm's overhead infrared frame, the segmented
silhouette, the depth-coloured point cloud and the reconstructed surface.*

## Results

| Stage | Metric | Value |
|---|---|---|
| Segmentation (U-Net, InceptionV3 encoder, infrared frames) | IoU / accuracy | **0.98 / 0.99** |
| Weight regression (CNN on segmented, centred depth, head excluded) | MAE (validation) | **3.6 kg** |
| Baseline (hand-crafted features + linear regression) | MAE (in-sample, no hold-out) | 6.43 kg |
| Size-group classification (small / medium / large) | accuracy | 84 % |

About **600 samples**. The gain came from data treatment. With the CNN regressor, the
validation MAE went from 7.6 kg on raw depth frames to 4.6 kg with the background removed,
4.2 kg with the pig centred, and 3.6 kg centred with the head excluded from the mask (the
head's position varies a lot from frame to frame).

**How to read these numbers** (see [Validation protocol](#validation-protocol)): the 3.6 kg
comes from a single random split by image, with the same animals on both sides, so it is
optimistic for unseen pigs; the baseline figure is measured on the data it was fitted on, so
the two MAEs are not measured the same way.

## Pipeline

1. **Data preparation**: depth + infrared frames matched to scale readings (pig RFID chip,
   weight, timestamp).
2. **Segmentation**: classic morphology (Otsu, background subtraction) vs **YOLOv5**
   detection vs **U-Net** semantic segmentation; U-Net won.
3. **3D processing**: point clouds and Poisson meshes with **Open3D**.
4. **Regression**: feature-based linear baseline, then CNN regression (InceptionV3 via
   `ktrain`) on the segmented depth images.

## Validation protocol

Read from the code, not assumed:

- **Weight regression** ([`src/model/train_regressor.py`](src/model/train_regressor.py)):
  `ktrain.vision.images_from_array(..., val_pct=0.1, random_state=21)`, which calls
  scikit-learn's `train_test_split` on the image array. The split is **random by image**, not
  by animal. The dataset is a log of repeated weighings of the same pigs (each row carries the
  pig's RFID chip), so frames of the same animal end up in both train and validation; the
  matching step can also assign more than one frame to the same scale reading, and nothing
  keeps those frames on the same side of the split. The 3.6 kg MAE is therefore an optimistic estimate for pigs the model has
  never seen. A fair estimate needs a split grouped by chip (for example `GroupShuffleSplit`
  or `GroupKFold` on the `Chip` column) and, with ~600 samples, cross-validation instead of a
  single 90/10 split (the report already notes the missing cross-validation).
- **Segmentation and size classification** use the same pattern: one random 80/20 split by
  image (`train_test_split(test_size=0.2)`).
- **Linear baseline** ([`src/model/linear_regressor.py`](src/model/linear_regressor.py)):
  fitted and scored on the same samples, so 6.43 kg is a training error.
- The 0.98 IoU comes from the InceptionV3-encoder U-Net trained on infrared frames. The
  committed [`train_segmentation.py`](src/model/train_segmentation.py) is the later variant
  trained on depth crops with head-less masks and a ResNet-50 encoder; the report gives it
  0.99 accuracy and no IoU.

## How to run

The farm data is not included (see [Limitations](#limitations)), so the scripts document the
pipeline rather than reproduce it out of the box. Every script uses paths relative to the
**repository root** and is run from there.

```bash
uv venv --python 3.8 --seed .venv && source .venv/bin/activate   # or any Python 3.8 environment
pip install -r requirements.txt
export MPLBACKEND=Agg
```

Two environment notes, both found while testing the install on a current Linux:

- On glibc 2.41 or newer, PyTorch 1.10 fails to import (`cannot enable executable stack`).
  Prefix the commands with `GLIBC_TUNABLES=glibc.rtld.execstack=2`, or use a Python 3.8
  container.
- The vendored YOLOv5 code downloads `Arial.ttf` when imported, and Python 3.8 does not follow
  the redirect that URL now returns. Fetch it once:
  `mkdir -p ~/.config/Ultralytics && curl -L -o ~/.config/Ultralytics/Arial.ttf https://ultralytics.com/assets/Arial.ttf`.

Expected raw layout: `data/raw/images/images/` (infrared PNG), `data/raw/images/images_3D/`
(16-bit depth PNG), `data/raw/images/groundtruth/` and `groundtruthV2/` (masks with and without
the head, labelled in Hasty.ai) and `data/raw/weights/*.xlsx` (scale logs).

Main path, in order:

| Step | Command | Produces |
|---|---|---|
| 1 | `python src/data/generate_weight_data.py` | `data/processed/weights.csv`: each depth frame matched to the nearest scale reading (4.6-minute clock offset, ≤ 100 s tolerance) |
| 2 | `python src/data/generate_size_data.py` | adds a weight-quintile `Size` column |
| 3 | `python src/data/segmentation.py` | classic baselines and the fixed crop (`easyCrop`) that feeds the U-Net, in `data/interim/`; choose the function by uncommenting its call at the end of the file |
| 4 | `python src/model/train_segmentation.py` | `models/segmentation/headlesspigs.h5` |
| 5 | `python src/model/segmentation_inference.py` | masks in `data/processed/masksV2/` |
| 6 | `python src/data/center_point_cloud.py` | depth frame and mask centred on the pig, in `data/interim/shifted/` (it works on the 2D images, despite the name) |
| 7 | `python src/model/train_regressor.py` | the `ktrain` regressor, saved to `models/regression/regression` |
| 8 | `python src/model/predict_weight.py` | `data/processed/weights_predicted.csv`; it loads a predictor named `test_one_out`, so point it at the model saved in step 7 |
| 9 | `python src/visualization/pigs_evolution.py` | growth curves per pig |

Side experiments: `src/data/generate_yolo_data.py` builds YOLO labels from the masks and
`python -m src.model.yolo.detect --weights <yolov5.pt> --source data/raw/images/images --save-txt --project data/interim/masks/yolo/boundingboxes --name exp`
runs detection (the YOLOv5 weights were trained on Google Colab and are not included);
`src/data/generate_pointcloud.py` and `generate_mesh.py` build the Open3D point clouds and
Poisson meshes; `src/data/eval_masks.py` scores the masks; `src/model/train_classificator.py`
is the size classifier. `src/model/linear_regressor.py` is kept as the record of the baseline:
as committed it depends on a cached feature array from an earlier run (`X2`) and does not run
end to end.

## Repo layout

```text
src/data/            dataset generation (weights, sizes, crops, masks, point clouds)
src/model/           segmentation, regression and classification training; vendored YOLOv5
src/visualization/   growth-evolution plots
reports/             thesis reports (final report: reports/informe_final/)
```

Full write-up (Spanish): [`reports/informe_final/informe_final.pdf`](reports/informe_final/informe_final.pdf)

## Stack

Python 3.8 · TensorFlow/Keras · ktrain · segmentation_models · OpenCV · YOLOv5 (PyTorch) · Open3D

## Limitations

- The farm dataset is **not redistributable** (commercial provenance), so the repository is
  code + reports only.
- Validation is a single random split by image with the same pigs on both sides, and no
  cross-validation (see [Validation protocol](#validation-protocol)); the reported MAE is
  optimistic for new animals.
- The dependencies are those of 2021 (TensorFlow 2.7, PyTorch 1.10, Open3D 0.13), which pins
  the project to Python 3.8.

## License

Code released under the [MIT License](LICENSE). Note: the licence covers the **code**
only. The depth-image dataset comes from a commercial farm, is not redistributable and
is not part of this repository.
