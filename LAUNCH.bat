@echo off
echo Starting TERRA-VIS...

if not exist "python\python.exe" (
    echo [TERRA-VIS] Portable Python not found. Running initial setup...
    call setup.bat
)

echo [TERRA-VIS] Checking for NVIDIA GPU...
wmic path win32_VideoController get name | findstr /i "NVIDIA" > nul
if %errorlevel% equ 0 (
    echo [TERRA-VIS] NVIDIA GPU detected. Installing PyTorch with CUDA support...
    python\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [TERRA-VIS] No NVIDIA GPU detected. Installing lightweight CPU PyTorch...
    python\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo [TERRA-VIS] Installing remaining dependencies...
python\python.exe -m pip install -r requirements.txt

echo [TERRA-VIS] Launching Secure Engine...
start "" python\python.exe api/server.pyc --ckpt dummy --port 8000
echo Waiting for systems to come online...
timeout /t 10 /nobreak > nul
start "" http://localhost:8000/app/index.html
echo Running. Close this window to shut down.
pause
