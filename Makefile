dev:
	uv run --env-file .env uvicorn podcastfy.api.fast_app:app --reload --host 0.0.0.0 --port 8000

lint:
	black podcastfy/*.py
	black tests/*.py
	mypy podcastfy/*.py

test:
	poetry run pytest -n auto
    
doc-gen:
	sphinx-apidoc -f -o ./docs/source ./podcastfy
	(cd ./docs && make clean && make html)
	