@echo on & setlocal EnableDelayedExpansion
chcp 65001

uvicorn app.main:app --reload
pause
# uvicorn app.main:app --host '192.168.30.17' --port 8000 --reload