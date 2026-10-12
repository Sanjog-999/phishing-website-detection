@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ===== Phishing Detection Project =====

if not exist app.py (
  echo ERROR: app.py is folder mein nahi hai. Ye file Phising detection folder mein rakho.
  pause
  exit /b
)

if not exist phishing_model.pkl (
  echo Model abhi nahi bana hai. Dataset CSV dhundh rahe hain...
  set "CSVFILE="
  for %%f in (*.csv) do set CSVFILE=%%f
  if "!CSVFILE!"=="" (
    echo ERROR: Is folder mein koi .csv dataset file nahi mili.
    echo Kaggle se "Phishing Site URLs" download karke isi folder mein rakho, phir dobara chalao.
    pause
    exit /b
  )
  echo Dataset mila: !CSVFILE!  -- training shuru...
  python train_model.py "!CSVFILE!"
  if errorlevel 1 (
    echo Training mein error aaya. Upar ka message padho.
    pause
    exit /b
  )
)

echo App start ho rahi hai... browser apne aap khulega.
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:5000"
python app.py
pause