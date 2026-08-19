# Batch Processing Feature

This document describes the new batch processing capabilities added to the Viola-Jones cone detector.

## Overview

The runtime detector now supports three modes of operation:

1. **Single Image Mode** - Process one image (backward compatible)
2. **List File Mode** - Process multiple images from a text list
3. **Directory Mode** - Process all images in a directory

## Usage

### Single Image Mode (Backward Compatible)

```bash
./build/main <image.jpg> <cascade.xml> <config.json> [output.jpg]
```

Example:
```bash
./build/main images/dataset/scene001.jpg cascade.xml configs/hyperparams.json output.jpg
```

**Output:** Single debug image with detections drawn on it.

---

### List File Mode

```bash
./build/main <list_of_images.txt> <cascade.xml> <config.json> <output_dir>
```

**Requirements:**
- List file must have `.txt` extension
- One image path per line
- Lines starting with `#` are treated as comments
- Empty lines are ignored

**Example list file (batch_demo.txt):**
```
# Sample list of images for batch processing
images/dataset/scene001.jpg
images/dataset/scene002.jpg
images/dataset/scene003.jpg
```

**Usage:**
```bash
./build/main batch_demo.txt cascade.xml configs/hyperparams.json results/
```

**Output:** All images are processed, output saved to `<output_dir>/<basename>_detected.jpg`

---

### Directory Mode

```bash
./build/main <directory/> <cascade.xml> <config.json> <output_dir>
```

**Behavior:**
- Scans the specified directory (not subdirectories) for `.jpg`, `.jpeg`, and `.png` files, case-insensitively
- Processes files in alphabetical order
- Creates the output directory when it does not exist

**Example:**
```bash
./build/main images/dataset/ cascade.xml configs/hyperparams.json results/
```

**Output:** All images in `images/dataset/` are processed, output saved to `results/<basename>_detected.jpg`. Duplicate basenames receive a numeric suffix to avoid overwriting a prior result.

---

## Output Format

### Single Image Mode
- **Input:** Single image file
- **Output:** Single image file (specified or default `detected_cones.jpg`)
- **Format:** JPEG with detection rectangles and labels

### Batch Mode
- **Input:** Multiple images (via list or directory)
- **Output:** Multiple image files
- **Naming Convention:** `<original_basename>_detected.jpg`
- **Location:** Specified output directory

## Example Workflow

### 1. Prepare a batch list
```bash
# Create list file
cat > my_batch_list.txt << EOF
# My traffic cone detection batch
images/road/intersection1.jpg
images/road/intersection2.jpg
images/road/highway1.jpg
images/road/highway2.jpg
images/road/intersection3.jpg
EOF
```

### 2. Run batch detection
```bash
./build/main my_batch_list.txt cascade.xml configs/hyperparams.json output/
```

### 3. View results
```bash
# List output files
ls output/

# Or use image viewer
eog output/*_detected.jpg
```

## Performance Tips

### Large Batches
For processing hundreds/thousands of images:

```bash
# Create large batch list
find images/dataset/ -type f \( -name "*.jpg" -o -name "*.png" \) > large_batch.txt

# Process
./build/main large_batch.txt cascade.xml configs/hyperparams.json output/
```

### Parallel Processing (Advanced)
For very large datasets, consider running multiple instances:

```bash
# Split batch into chunks
split -l 100 large_batch.txt chunk_

# Process in parallel
for chunk in chunk_*; do
    ./build/main "$chunk" cascade.xml configs/hyperparams.json output/ &
done
wait
```

## Error Handling

The batch processor includes robust error handling:

- **Missing or unreadable images:** Skipped with warning, processing continues
- **Corrupt cascade:** All processing stops with clear error message
- **Invalid config:** Detected before processing begins
- **Write failures:** Logged per-image, processing continues
- **Exit status:** `0` when every image succeeds; `2` when one or more batch items fail

**Example error output:**
```
Processing: images/scene002.jpg ... Failed to read image
Processing: images/scene003.jpg ... OK (3 cones)
Processing: images/scene004.jpg ... OK (1 cone)

========================================
Batch Processing Complete
========================================
Total images: 5
Processed: 3
Failed: 2
Total detections: 4
Output directory: results/
```

## Comparison with Single Image Mode

| Feature | Single Image | Batch Mode |
|---------|-------------|------------|
| Input | Single file | List file or directory |
| Output | Single file | Multiple files |
| Progress | Immediate | Progress per image |
| Error handling | Immediate stop | Per-image recovery |
| Use case | Quick testing | Production pipelines |

## Configuration

Batch mode uses the same configuration as single image mode via `configs/hyperparams.json`:

```json
{
    "scaleFactor": 1.1,
    "minNeighbors": 3,
    "minSize": [16, 24],
    "maxSize": [200, 300]
}
```

No additional configuration needed for batch processing.

## Troubleshooting

### "No images found in directory"
- Check directory path is correct
- Verify images have supported extensions (.jpg, .jpeg, .png)

### "No valid image paths found in list file"
- Ensure list file has `.txt` extension
- Check for empty lines or comments at the end
- Verify image paths are valid and accessible

### Slow processing
- Check if images are large (consider resizing before detection)
- Consider reducing `maxSize` in config
- Use parallel processing for large batches

## Future Enhancements

Potential improvements for batch processing:
- [ ] Progress bar visualization
- [ ] Batch statistics summary (detections per image, average confidence)
- [ ] Output format options (JSON annotations, CSV statistics)
- [ ] Custom output naming schemes
- [ ] Filter options (process only specific extensions)
