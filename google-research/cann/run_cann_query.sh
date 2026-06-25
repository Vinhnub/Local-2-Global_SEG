#!/bin/bash
# ==========================================
# QUERY DATABASE VỚI CANN (RESUME VERSION)
# Index: TOÀN BỘ database (đã index sẵn)
# Query: TỪNG BATCH (để tránh crash)
# AUTO RESUME + SKIP COMPLETED BATCHES
# ==========================================

BASE_DIR="/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG"
CANN_BINARY="$BASE_DIR/google-research/cann/bazel-bin/main/colored_c_nn_random_grids_index_main"
CANN_DATA="$BASE_DIR/cann_data"
INDEX_OUTPUT="$BASE_DIR/index_output"
QUERY_OUTPUT="$BASE_DIR/query_results"

mkdir -p "$QUERY_OUTPUT"

echo "=========================================="
echo "QUERY DATABASE WITH CANN"
echo "Paper L2G: Retrieve top-1600 similar images"
echo "=========================================="

# ===== THAM SỐ THEO PAPER =====
BATCH_SIZE=70        # 70 queries cho ROxford5K và RParis6K
NUM_GRIDS=5
TOP_K=1600           # ← SỬA: 1600 theo paper

for DATASET in roxford5k rparis6k; do
    echo ""
    echo "=========================================="
    echo "Dataset: $DATASET"
    echo "=========================================="

    DB_DIR="$CANN_DATA/$DATASET/database"
    Q_DIR="$CANN_DATA/$DATASET/query"

    if [ ! -d "$DB_DIR" ]; then
        echo "  ✗ Database not found: $DB_DIR"
        continue
    fi

    if [ ! -d "$Q_DIR" ]; then
        echo "  ✗ Query not found: $Q_DIR"
        continue
    fi

    # list database và query
    ls -1 "$DB_DIR"/*.desc > "$DB_DIR/all_descriptors.txt"
    ls -1 "$Q_DIR"/*.desc > "$Q_DIR/all_queries.txt"
    
    TOTAL_DB=$(cat "$DB_DIR/all_descriptors.txt" | wc -l)
    TOTAL_Q=$(cat "$Q_DIR/all_queries.txt" | wc -l)

    echo "  Total database: $TOTAL_DB images"
    echo "  Total queries: $TOTAL_Q images (70 queries)"
    echo "  Batch size: $BATCH_SIZE queries/batch (1 batch = tất cả 70 queries)"
    echo "  Top-K: $TOP_K neighbors per query (paper: 1600)"
    echo "  Num grids: $NUM_GRIDS"
    echo ""

    OUTPUT_CSV="$QUERY_OUTPUT/${DATASET}_query_results.csv"

    # reset final output
    > "$OUTPUT_CSV"

    BATCH_COUNT=0
    TOTAL_ROWS=0

    # ===== AUTO RESUME START =====
    RESUME_START=0

    LAST_BATCH=$(ls "$QUERY_OUTPUT"/${DATASET}_query_batch_*.csv 2>/dev/null \
        | sed 's/.*query_batch_//;s/.csv//' \
        | sort -n \
        | tail -1)

    if [ -n "$LAST_BATCH" ]; then
        RESUME_START=$((LAST_BATCH + BATCH_SIZE))
        echo "  ↪ Resume detected, starting from: $RESUME_START"
    else
        echo "  ↪ No checkpoint found, start from 0"
    fi

    # ===== MAIN LOOP =====
    for ((START=RESUME_START; START<TOTAL_Q; START+=BATCH_SIZE)); do

        END=$((START + BATCH_SIZE))
        [ $END -gt $TOTAL_Q ] && END=$TOTAL_Q

        BATCH_COUNT=$((BATCH_COUNT + 1))
        BATCH_IMAGES=$((END - START))

        BATCH_OUTPUT="$QUERY_OUTPUT/${DATASET}_query_batch_${START}.csv"

        # =========================
        # SKIP IF ALREADY DONE
        # =========================
        if [ -s "$BATCH_OUTPUT" ]; then
            echo "  ⏭ Skip batch START=$START (already done)"
            continue
        fi

        echo "  Batch $BATCH_COUNT | START=$START END=$END ($BATCH_IMAGES queries)"

        # create query batch file
        sed -n "$((START+1)),${END}p" "$Q_DIR/all_queries.txt" \
            > "$Q_DIR/query_batch_${START}.txt"

        # ===== RUN CANN =====
        # Index: TOÀN BỘ database
        # Query: TỪNG BATCH
        $CANN_BINARY \
            --index_descriptor_files "$DB_DIR/all_descriptors.txt" \
            --query_descriptor_files "$Q_DIR/query_batch_${START}.txt" \
            --pairs_file "$BATCH_OUTPUT" \
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
        rm -f "$Q_DIR/query_batch_${START}.txt"

    done

    # ===== THỐNG KÊ =====
    echo ""
    echo "=========================================="
    echo "✓ $DATASET QUERY COMPLETED!"
    echo "Total rows: $TOTAL_ROWS"
    echo "Expected rows: $TOTAL_Q × $TOP_K = $((TOTAL_Q * TOP_K)) rows"

    if [ $TOTAL_Q -gt 0 ]; then
        echo "Avg neighbors/query = $((TOTAL_ROWS / TOTAL_Q))"
    fi

    if [ $TOTAL_ROWS -eq $((TOTAL_Q * TOP_K)) ]; then
        echo "  ✅ PERFECT! Each query has $TOP_K neighbors"
    else
        echo "  ⚠ Warning: Missing $((TOTAL_Q * TOP_K - TOTAL_ROWS)) rows"
    fi

    echo ""
    echo "Sample output (first 3 rows):"
    head -3 "$OUTPUT_CSV" 2>/dev/null || echo "  (no data)"

done

echo ""
echo "=========================================="
echo "✓ ALL QUERIES COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved in: $QUERY_OUTPUT/"
echo "  - roxford5k_query_results.csv"
echo "  - rparis6k_query_results.csv"