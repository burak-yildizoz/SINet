@echo off
set pyver=Python 3.6.6
for /f "delims=" %%a in ('python -V') do set "readValue=%%a"
if "%readValue%" == "%pyver%" (echo %readValue%) else (echo Python version should be %pyver% && pause && exit)
pip install torch==1.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==3.4.2.16
pip install scipy==1.2.2
pause
