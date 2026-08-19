#!/bin/bash

# Test script for batch processing feature
# Run from the project root directory

echo "========================================"
echo "Batch Processing Test Script"
echo "========================================"

# Check if build exists
if [ ! -d "build" ]; then
    echo "ERROR: Build directory not found. Run: cmake -S . -B build && cmake --build build -j4"
    exit 1
fi

# Check if main executable exists
if [ ! -f "build/main" ]; then
    echo "ERROR: main executable not found. Run cmake build first."
    exit 1
fi

echo ""
echo "1. Testing Single Image Mode..."
echo "   Command: ./build/main <image> <cascade.xml> <config.json> [output]"
echo "   Expected: Single image processed, output saved"
echo ""

echo "2. Testing List File Mode..."
echo "   Command: ./build/main <list.txt> <cascade.xml> <config.json> <output_dir>"
echo "   Expected: All images in list processed, outputs in directory"
echo ""

echo "3. Testing Directory Mode..."
echo "   Command: ./build/main <directory/> <cascade.xml> <config.json> <output_dir>"
echo "   Expected: All images in directory processed"
echo ""

echo "========================================"
echo "Usage Examples:"
echo "========================================"
echo ""
echo "# Single image (backward compatible)"
echo "./build/main images/test.jpg cascade.xml configs/hyperparams.json output.jpg"
echo ""
echo "# Batch from list file"
echo "./build/main batch_demo.txt cascade.xml configs/hyperparams.json results/"
echo ""
echo "# Batch from directory"
echo "./build/main images/dataset/ cascade.xml configs/hyperparams.json results/"
echo ""

echo "========================================"
echo "Batch Processing Feature Ready!"
echo "========================================"