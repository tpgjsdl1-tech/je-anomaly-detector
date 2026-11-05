@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo 회계전표 이상징후 탐지 보조 툴을 시작합니다...
echo 브라우저가 열리지 않으면 http://localhost:8501 로 접속하세요.
echo 종료하려면 이 창을 닫으세요.
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py --browser.gatherUsageStats false
pause
