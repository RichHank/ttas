#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_pipeline.py --refresh --write-html

echo "TTAS cache written to outputs/cache"
echo "Run the dashboard with: python dashboard/app.py"
