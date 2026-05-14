$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_pipeline.py --refresh --write-html

Write-Host "TTAS cache written to outputs/cache"
Write-Host "Run the dashboard with: python dashboard/app.py"
