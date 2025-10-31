freeze-constraints:
	python3 -m venv .tmp-constraints-venv && \
	source .tmp-constraints-venv/bin/activate && \
	pip install --upgrade pip setuptools wheel && \
	pip install -r requirements.txt && \
	pip freeze --exclude-editable > constraints.txt
