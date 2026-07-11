@echo off
echo Cleaning old dist...
if exist dist rmdir /s /q dist
mkdir dist\TERRA-VIS
mkdir dist\TERRA-VIS\api
mkdir dist\TERRA-VIS\src

echo Compiling Python source code to bytecode...
xcopy /E /I api dist\TERRA-VIS\api
xcopy /E /I src dist\TERRA-VIS\src
.\venv310\Scripts\python.exe -m compileall -b dist\TERRA-VIS\api
.\venv310\Scripts\python.exe -m compileall -b dist\TERRA-VIS\src

echo Removing raw source code...
del /S /Q dist\TERRA-VIS\api\*.py
del /S /Q dist\TERRA-VIS\src\*.py

echo Copying required files...
xcopy /E /I vault dist\TERRA-VIS\vault
copy index.html dist\TERRA-VIS\
copy tools\build_portable_python.bat dist\TERRA-VIS\setup.bat
copy LAUNCH.bat dist\TERRA-VIS\
copy requirements.txt dist\TERRA-VIS\
copy README.txt dist\TERRA-VIS\

echo Build complete! 
