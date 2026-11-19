.PHONY: setup sample run app test lint docker
setup:   ; pip install -r requirements.txt
sample:  ; python scripts/make_sample_data.py
run:     ; python run_pipeline.py --data data/911.csv
app:     ; streamlit run app/dashboard.py
test:    ; python -m pytest -q
lint:    ; ruff check . && ruff format --check .
docker:  ; docker build -t emergency-911 . && docker run -p 8501:8501 -v $$(pwd)/data:/app/data -v $$(pwd)/reports:/app/reports emergency-911
