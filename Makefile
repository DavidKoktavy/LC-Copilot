.PHONY: install test lint eval gate serve mcp mcp-demo ask docker clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests evals scripts

eval:
	python -m evals.run

gate:
	python -m evals.run --gate

serve:
	uvicorn lc_copilot.api.main:app --reload --port 8000

mcp:
	python -m lc_copilot.mcp_server

mcp-demo:
	python scripts/mcp_client_demo.py

ask:
	python -m lc_copilot.cli "$(Q)"

docker:
	docker build -t lc-copilot . && docker run --rm -p 8000:8000 lc-copilot

clean:
	rm -rf runs data/portfolio.db evals/reports .pytest_cache
