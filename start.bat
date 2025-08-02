@echo off
title NOUS RESEARCH
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python NOUS.py
pause
