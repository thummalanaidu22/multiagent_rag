#!/bin/bash
# Full pipeline: ingest → generate questions → evaluate → chat
set -e

VENV="$(dirname "$0")/venv/bin/python"

echo "=== Step 1: Ingest Documents ==="
$VENV ingest.py

echo ""
echo "=== Step 2: Generate 50 Evaluation Questions ==="
$VENV generate_questions.py --num-questions 50

echo ""
echo "=== Step 3: Run Evaluation ==="
$VENV evaluate.py --k 5

echo ""
echo "=== Step 4: Start Chatbot ==="
$VENV main.py
