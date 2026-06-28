#!/bin/bash
# ==========================================
# BUILD DATABASE INDEX VỚI CANN (RESUME VERSION)
# Index: TOÀN BỘ database
# Query: TỪNG BATCH (để tránh crash)
# AUTO RESUME + SKIP COMPLETED BATCHES
# ==========================================

BASE_DIR="/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG"
CANN_BINARY="$BASE_DIR/google-research/cann/bazel-bin/main/colored_c_nn_random_grids_index_main"
CANN_DATA="$BASE_DIR/cann_data"
INDEX_OUTPUT="$BASE_DIR/index_output"

mkdir -p "$INDEX_OUTPUT"

echo "=========================================="
echo "BUILD DATABASE INDEX WITH CANN"
echo "Paper L2G: Precompute 700 neighbors"
echo "=========================================="

# ===== THAM SỐ =====
BATCH_SIZE=1000
NUM_GRIDS=5

for DATASET in roxford5k rparis6k; do
    echo ""
    echo "=========================================="
    echo "Dataset: $DATASET"
    echo "=========================================="

    DB_DIR="$CANN_DATA/$DATASET/database"

    if [ ! -d "$DB_DIR" ]; then
        echo "  ✗ Database not found: $DB_DIR"
        continue
    fi

    # list database
    ls -1 "$DB_DIR"/*.desc > "$DB_DIR/all_descriptors.txt"
    TOTAL_DB=$(cat "$DB_DIR/all_descriptors.txt" | wc -l)

    echo "  Total database: $TOTAL_DB images"
    echo "  Batch size: $BATCH_SIZE queries/batch"
    echo "  Num grids: $NUM_GRIDS"
    echo ""

    OUTPUT_CSV="$INDEX_OUTPUT/${DATASET}_self_pairs.csv"

    # reset final output (giữ batch files để resume)
    > "$OUTPUT_CSV"

    BATCH_COUNT=0
    TOTAL_ROWS=0

    # ===== AUTO RESUME START =====
    RESUME_START=0

    LAST_BATCH=$(ls "$INDEX_OUTPUT"/${DATASET}_batch_*.csv 2>/dev/null \
        | sed 's/.*batch_//;s/.csv//' \
        | sort -n \
        | tail -1)

    if [ -n "$LAST_BATCH" ]; then
        RESUME_START=$((LAST_BATCH + BATCH_SIZE))
        echo "  ↪ Resume detected, starting from: $RESUME_START"
    else
        echo "  ↪ No checkpoint found, start from 0"
    fi

    # ===== MAIN LOOP =====
    for ((START=RESUME_START; START<TOTAL_DB; START+=BATCH_SIZE)); do

        END=$((START + BATCH_SIZE))
        [ $END -gt $TOTAL_DB ] && END=$TOTAL_DB

        BATCH_COUNT=$((BATCH_COUNT + 1))

        BATCH_OUTPUT="$INDEX_OUTPUT/${DATASET}_batch_${START}.csv"

        # =========================
        # SKIP IF ALREADY DONE
        # =========================
        if [ -s "$BATCH_OUTPUT" ]; then
            echo "  ⏭ Skip batch START=$START (already done)"
            continue
        fi

        echo "  Batch $BATCH_COUNT | START=$START END=$END"

        # create query batch file
        sed -n "$((START+1)),${END}p" "$DB_DIR/all_descriptors.txt" \
            > "$DB_DIR/query_batch_${START}.txt"

        # ===== RUN CANN =====
        $CANN_BINARY \
            --index_descriptor_files "$DB_DIR/all_descriptors.txt" \
            --query_descriptor_files "$DB_DIR/query_batch_${START}.txt" \
            --pairs_file "$BATCH_OUTPUT" \
            --num_features 700 \
            --num_grids $NUM_GRIDS

        # ===== CHECK OUTPUT =====
        if [ -s "$BATCH_OUTPUT" ]; then
            ROWS=$(wc -l < "$BATCH_OUTPUT")
            TOTAL_ROWS=$((TOTAL_ROWS + ROWS))

            FIRST_QUERY=$(head -1 "$BATCH_OUTPUT" | cut -d',' -f1)
            if [ -n "$FIRST_QUERY" ]; then
                NEIGHBORS=$(grep "^$FIRST_QUERY" "$BATCH_OUTPUT" | wc -l)
                echo "    ✓ DONE: $ROWS rows | $NEIGHBORS neighbors/query"
            else
                echo "    ✓ DONE: $ROWS rows"
            fi

            cat "$BATCH_OUTPUT" >> "$OUTPUT_CSV"
        else
            echo "    ✗ FAILED batch START=$START"
        fi

        # cleanup temp query file
        rm -f "$DB_DIR/query_batch_${START}.txt"

    done

    echo ""
    echo "=========================================="
    echo "✓ $DATASET COMPLETED!"
    echo "Total rows: $TOTAL_ROWS"

    if [ $TOTAL_DB -gt 0 ]; then
        echo "Avg neighbors/query = $((TOTAL_ROWS / TOTAL_DB))"
    fi

    echo ""
    echo "Sample output:"
    head -3 "$OUTPUT_CSV" 2>/dev/null || echo "  (no data)"

done

echo ""
echo "=========================================="
echo "✓ ALL COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved in: $INDEX_OUTPUT/"