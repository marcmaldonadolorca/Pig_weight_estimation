```` bash
├── Pigs
│   ├── data
│   │   ├── interim
│   │   │   ├── acc
│   │   │   ├── augmented_data
│   │   │   ├── convex_hull
│   │   │   ├── crop
│   │   │   │   ├── background.png
│   │   │   │   ├── groundtruth
│   │   │   │   ├── groundtruthV2
│   │   │   │   ├── images
│   │   │   │   └── images_3D
│   │   │   ├── gauss_noise
│   │   │   ├── masks
│   │   │   │   ├── difference
│   │   │   │   ├── difference3D
│   │   │   │   ├── mean
│   │   │   │   ├── mean3D
│   │   │   │   ├── otsu
│   │   │   │   ├── otsu_crop
│   │   │   │   ├── semantic_segmentationV1
│   │   │   │   ├── semantic_segmentationV2
│   │   │   │   ├── shifted
│   │   │   │   └── yolo
│   │   │   │       ├── boundingboxes
│   │   │   │       └── images
│   │   │   ├── shifted
│   │   │   ├── var
│   │   │   └── weights
│   │   ├── processed
│   │   │   ├── elipsoid
│   │   │   ├── masks
│   │   │   ├── masksV2
│   │   │   ├── pcd
│   │   │   ├── plots
│   │   │   ├── ply
│   │   │   ├── weights.csv
│   │   │   ├── weights_predicted.csv
│   │   │   └── yolo_boundingbox
│   │   └── raw
│   │       ├── images
│   │       │   ├── groundtruth
│   │       │   ├── groundtruthV2
│   │       │   ├── images
│   │       │   └── images_3D
│   │       └── weights
│   ├── models
│   │   ├── regression
│   │   └── segmentation
│   │       └── yoloV5
│   ├── readme.md
│   ├── reports
│   │   ├── informe_final
│   │   ├── informe_inicial
│   │   ├── informe_seguiment_1
│   │   └── informe_seguiment_2
│   ├── requirements.txt
│   └── src
│       ├── data
│       │   ├── apply_gaussian_noise.py
│       │   │       To add gaussian noise to images
│       │   ├── center_point_cloud.py
│       │   │       To center the segmentated pig in the image
│       │   ├── clean_segmentation_mask.py
│       │   │       Morfologic operations for the segmentation mask
│       │   ├── eval_masks.py
│       │   │       To get the acc of the segmentations
│       │   ├── exploration.py
│       │   │       To explore data, duplicates, outlayers...
│       │   ├── generate_mesh.py
│       │   │       To generate a pointcloud
│       │   ├── generate_pointcloud.py
│       │   │       To generate a point cloud
│       │   ├── generate_size_data.py
│       │   │       To labels images for classification
│       │   ├── generate_weight_data.py
│       │   │       To extrat data from de xls files for training models
│       │   ├── generate_yolo_data.py
│       │   │       To generate yolo data for train the YOLO model
│       │   └── segmentation.py
│       │           First segmentation attempts
│       ├── model
│       │   ├── models
│       │   ├── utils
│       │   ├── yolo
│       │   │       Yolo code for object detection
│       │   ├── data_augmentation.py
│       │   │       To add gaussian noise to images
│       │   ├── linear_regressor.py
│       │   │       To train the linear regressor and generate elipsoid figures
│       │   ├── models
│       │   ├── predict_weight.py
│       │   │       To predict the weights with the CNN
│       │   ├── segmentation_inference.py
│       │   │       To add gaussian noise to images
│       │   ├── train_classificator.py
│       │   │       To train de CNNs for clasification
│       │   ├── train_regressor.py
│       │   │       To train de CNNs for weight prediction
│       │   ├── train_segmentation.py
│       │   │       To train de CNNs for image segmentation
│       │   ├── utils
│       │   └── yolo
│       │       └── detect.py
│       │               To train de CNNs for weight prediction
│       └── visualization
│           └── pigs_evolution.py
│                   To get the pig weight evolution plot


````