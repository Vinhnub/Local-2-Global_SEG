#!/bin/bash
"""
Build CANN index using compiled binary
"""

BASE_DIR="/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG"
CANN_BINARY="$BASE_DIR/google-research/cann/bazel-bin/main/colored_c_nn_random_grids_index_main"
CANN_DATA="$BASE_DIR/cann_data"
INDEX_OUTPUT="$BASE_DIR/index_output"

mkdir -p "$INDEX_OUTPUT"

echo "=========================================="
echo "BUILD CANN INDEX"
echo "=========================================="

# Kiểm tra binary
if [ ! -f "$CANN_BINARY" ]; then
    echo "✗ CANN binary not found!"
    exit 1
fi

echo "✓ CANN binary: $CANN_BINARY"
echo ""

# Show help để biết flags
echo "Checking available flags..."
$CANN_BINARY --help 2>&1 | head -20
echo ""

# Build index cho từng dataset
for DATASET in roxford5k rparis6k; do
    echo "=========================================="
    echo "Dataset: $DATASET"
    echo "=========================================="
    
    DATABASE_DIR="$CANN_DATA/$DATASET/database"
    INDEX_FILE="$INDEX_OUTPUT/${DATASET}_cann_index"
    
    DESC_COUNT=$(ls -1 $DATABASE_DIR/*.desc 2>/dev/null | wc -l)
    if [ $DESC_COUNT -eq 0 ]; then
        echo "✗ No .desc files found!"
        continue
    fi
    
    echo "  Database: $DESC_COUNT images"
    echo "  Index output: $INDEX_FILE"
    echo ""
    
    # THỬ CÁC CÁCH KHÁC NHAU
    
    # Cách 1: Chỉ index (không output_index_file)
    echo "Trying: index_descriptor_files only..."
    $CANN_BINARY \
        --index_descriptor_files "$DATABASE_DIR/*.desc"
    
    # Cách 2: Thử với output_file
    echo ""
    echo "Trying: with output_file..."
    $CANN_BINARY \
        --index_descriptor_files "$DATABASE_DIR/*.desc" \
        --output_file "$INDEX_FILE"
    
    # Cách 3: Thử với pairs_file (có thể tự tạo index)
    echo ""
    echo "Trying: with pairs_file..."
    $CANN_BINARY \
        --index_descriptor_files "$DATABASE_DIR/*.desc" \
        --pairs_file "$INDEX_OUTPUT/${DATASET}_pairs.csv"
    
    if [ $? -eq 0 ]; then
        echo "✓ Command completed!"
    else
        echo "✗ Failed"
    fi
    
    echo ""
done

echo "=========================================="
echo "✓ DONE!"
echo "=========================================="