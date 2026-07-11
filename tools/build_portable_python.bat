@echo off
echo [1] Downloading Portable Python 3.10...
powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile 'python_embed.zip'"
powershell -Command "Expand-Archive -Path 'python_embed.zip' -DestinationPath '..\python'"
del python_embed.zip

echo [2] Enabling site-packages...
powershell -Command "(Get-Content '..\python\python310._pth') -replace '#import site', 'import site' | Set-Content '..\python\python310._pth'"

echo [3] Installing pip...
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '..\python\get-pip.py'"
..\python\python.exe ..\python\get-pip.py
del ..\python\get-pip.py

echo Portable Python environment created in \python folder.
