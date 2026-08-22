# Cones Detection: Viola-Jones & Hard Negative Mining Pipeline

This guide defines the complete pipeline for the training, optimization via Hard Negative Mining (HNM), and validation of the Viola-Jones cone detector.

The legacy training tools (`opencv_createsamples`, `opencv_traincascade`) are isolated in a Docker environment based on OpenCV 3.4 to avoid conflicts with the OpenCV 4 runtime dependency.

## 1. Host Project Compilation

The system requires native compilation of the utility binaries, updated to include the Hard Negatives extractor.

```bash
cmake -S . -B build
cmake --build build -j4
```

## 2. Analysis and Preparation of the Base Dataset

The native executables generate the manifests and random background samples.

```bash
# 1. Statistical analysis (determines optimal scaleFactor and minNeighbors)
./build/analyze_vj_dataset dataset

# 2. Base negative patches generation (36x36) and annotation files
./build/prepare_vj_dataset dataset training

# 3. Deterministic dataset partitioning (e.g., 70/15/15)
./build/split_vj_dataset training 12345 70 15 15
```

## 3. Training Environment Isolation (Docker)

Haar/LBP training requires OpenCV 3.4. The compilation disables test modules to prevent Out-Of-Memory crashes.

1. Creation of the `Dockerfile.opencv3`:

```dockerfile
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential cmake git libjpeg-dev libpng-dev libtiff-dev && rm -rf /var/lib/apt/lists/*
RUN git clone --branch 3.4 --depth 1 https://github.com/opencv/opencv.git /opt/opencv
WORKDIR /opt/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_opencv_apps=ON -D BUILD_SHARED_LIBS=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF .. && make -j4 && make install
WORKDIR /workspace
```

2. Image compilation:

```bash
docker build -t opencv3-trainer -f Dockerfile.opencv3 .
```

*Architectural note*: Allocate a minimum of 10GB of RAM to the Docker daemon via the hypervisor settings to prevent system kills during feature calculation.

## 4. Path Resolution and Initial Training (Base Cascade)

OpenCV 3.4 resolves relative paths based on the manifest location. It is mandatory to inject symlinks to simulate the project root.

```bash
# Symlink Setup
ln -s ../../dataset training/manifests/dataset
ln -s ../../training training/manifests/training
```

### 4.1 Vector Generation

*Replace `<NUM_STANDING_TRAIN>` with the value extracted from `training/split_report.txt`.*

```bash
docker run --rm -v $(pwd):/workspace opencv3-trainer \
    opencv_createsamples \
    -info training/manifests/train_annotations_standing.txt \
    -vec training/positives_standing.vec \
    -num <NUM_STANDING_TRAIN> \
    -w 24 -h 36

```

### 4.2 Base Cascade Training

Memory parameters (`-precalcValBufSize`, `-precalcIdxBufSize`) reduce bottlenecks. `numPos` must be conservative (~85% of the total in `.vec`) to prevent stalls.

```bash
mkdir -p training/cascade_base

docker run --rm -v $(pwd):/workspace opencv3-trainer \
    opencv_traincascade \
    -data training/cascade_base \
    -vec training/positives_standing.vec \
    -bg training/manifests/train_negatives.txt \
    -numPos <NUM_POS_CALCULATED> -numNeg <NUM_NEG_CALCULATED> \
    -numStages 10 \
    -w 24 -h 36 \
    -minHitRate 0.995 \
    -maxFalseAlarmRate 0.5 \
    -precalcValBufSize 2048 \
    -precalcIdxBufSize 2048

```

## 5. Hard Negative Mining (HNM) and JSON Inference

The base cascade contains numerous false positives. Running `hard_negative_miner` on the entire dataset generates metrics for the segmentation team and extracts structured noise patches (36x36 px) misclassified with high confidence.

```bash
# Native script execution (JSON output + Hard Negative patches)
./build/hard_negative_miner dataset training/cascade_base/cascade.xml configs/hyperparams.json hnm_output/
```

*Output produced in `hnm_output/`:*

* `json_predictions/`: `.json` files usable downstream for segmentation.
* `hnm_patches/`: PNG images of the noise (e.g., vehicle portions, trees).
* `train_negatives_hnm.txt`: Relative manifest of the new samples.

## 6. Data Fusion and Final Re-Training

The HNM patches must be mixed with the random background to avoid the phenomenon of *catastrophic forgetting*.

```bash
# Merging random background manifest + hard negatives
cat training/manifests/train_negatives.txt hnm_output/train_negatives_hnm.txt > training/manifests/combined_negatives.txt
```

*Replace `<NEW_NUM_NEG>` with 50% of the total lines counted in `combined_negatives.txt`.*

```bash
mkdir -p training/cascade_final

docker run --rm -v $(pwd):/workspace opencv3-trainer     opencv_traincascade     -data training/cascade_final     -vec training/positives_standing.vec     -bg training/manifests/combined_negatives.txt     -numPos <NUM_POS_CALCULATED> -numNeg <NEW_NUM_NEG>     -numStages 10     -w 24 -h 36     -minHitRate 0.995     -maxFalseAlarmRate 0.5     -precalcValBufSize 2048     -precalcIdxBufSize 2048
```

## 7. Validation and Visual Inspection

The final visual test (bounding boxes rendering) must be performed exclusively on the validation split, using the C++ detector that implements Non-Maximum Suppression based on Intersection-over-Minimum (IoM).

```bash
# Temporary files cleanup
rm training/manifests/dataset
rm training/manifests/training

# Batch execution
mkdir -p validation_results
cp training/splits/val_images.txt ./val_images.txt
./build/main val_images.txt training/cascade_final/cascade.xml configs/hyperparams.json validation_results/