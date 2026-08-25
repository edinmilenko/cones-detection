# Cones Detection: Viola-Jones & Hard Negative Mining Pipeline

This guide defines the complete pipeline for the training, optimization via Hard Negative Mining (HNM), and validation of the Viola-Jones cone detector using Haar features.

The legacy training tools (`opencv_createsamples`, `opencv_traincascade`) are isolated in a Docker environment based on OpenCV 3.4 to avoid conflicts with the OpenCV 4 runtime dependency.

## 1. Host Project Compilation

The system requires native compilation of the utility binaries, updated to include the Hard Negatives extractor and the analytical evaluator.

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

Haar training requires OpenCV 3.4. The compilation disables test modules to prevent Out-Of-Memory crashes.

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

Generates the initial `.vec` file using the positive samples designated for training.

```bash
docker run --rm -v $(pwd):/workspace opencv3-trainer \
    opencv_createsamples \
    -info training/manifests/train_annotations_standing.txt \
    -vec training/positives_standing.vec \
    -num 13760 \
    -w 24 -h 36

```

### 4.2 Base Cascade Training

Memory parameters (`-precalcValBufSize`, `-precalcIdxBufSize`) reduce bottlenecks by caching feature values in RAM. `numPos` is capped at ~83% to prevent infinite loops.

```bash
mkdir -p training/cascade_base

docker run --rm -v $(pwd):/workspace opencv3-trainer \
    opencv_traincascade \
    -data training/cascade_base \
    -vec training/positives_standing.vec \
    -bg training/manifests/train_negatives.txt \
    -numPos 11500 -numNeg 4000 \
    -numStages 10 \
    -w 24 -h 36 \
    -featureType HAAR \
    -minHitRate 0.995 \
    -maxFalseAlarmRate 0.5 \
    -precalcValBufSize 2048 \
    -precalcIdxBufSize 2048

```

## 5. Hard Negative Mining (HNM) and JSON Inference

The base cascade produces numerous False Positives. The `hard_negative_miner` evaluates the entire dataset, exports JSON predictions, and crops 36x36 px patches that triggered high-confidence false alarms.

```bash
# Native script execution (JSON output + Hard Negative patches)
./build/hard_negative_miner dataset training/cascade_base/cascade.xml configs/hyperparams.json hnm_output/

```

## 6. Subsampling and Final Re-Training

To avoid memory saturation and *catastrophic forgetting*, the pipeline subsamples the generated HNM patches and mixes them with the random background negatives (cap: 15,000 total negatives, 15,000 total positives).

```bash
# MacOS/BSD POSIX subsampling using sort -R
sort -R hnm_output/train_negatives_hnm.txt | head -n 7500 > hnm_output/sampled_hnm.txt
sort -R training/manifests/train_negatives.txt | head -n 7500 > training/manifests/sampled_random.txt

# Merging random background manifest + hard negatives
cat training/manifests/sampled_random.txt hnm_output/sampled_hnm.txt > training/manifests/combined_negatives.txt
```

Generate the new subsampled positive vector:

```bash
docker run --rm -v $(pwd):/workspace opencv3-trainer \
    opencv_createsamples \
    -info training/manifests/train_annotations_standing.txt \
    -vec training/positives_final.vec \
    -num 15000 \
    -w 24 -h 36
```

Train the final Haar cascade on the balanced dataset:

```bash
mkdir -p training/cascade_final

docker run --rm -v $(pwd):/workspace opencv3-trainer \
    opencv_traincascade \
    -data training/cascade_final \
    -vec training/positives_final.vec \
    -bg training/manifests/combined_negatives.txt \
    -numPos 12500 -numNeg 7500 \
    -numStages 10 \
    -w 24 -h 36 \
    -featureType HAAR \
    -minHitRate 0.995 \
    -maxFalseAlarmRate 0.5 \
    -precalcValBufSize 2048 \
    -precalcIdxBufSize 2048
```

## 7. Analytical Validation and Visual Inspection

Validation is strictly performed on the `val` split. It requires an analytical evaluation of Precision, Recall, and FPPI using the custom IoU thresholding script, followed by the visual rendering of bounding boxes using Intersection-over-Minimum (IoM) NMS.

```bash
# Temporary files cleanup
rm training/manifests/dataset
rm training/manifests/training

# Prepare validation manifest
cp training/splits/val_images.txt ./val_images.txt

# 1. Analytical Evaluation (Precision, Recall, FPPI)
./build/evaluate_model val_images.txt training/cascade_final/cascade.xml configs/hyperparams.json

# 2. Batch Execution & Visual Debug
mkdir -p validation_results
./build/main val_images.txt training/cascade_final/cascade.xml configs/hyperparams.json validation_results/
```