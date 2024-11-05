install:
	python -m venv lewagon_signlanguage
	. lewagon_signlanguage/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Pleas activate your VENV:
# mac: source lewagon_signlanguage/bin/activate
# windows: .\lewagon_signlanguage\Scripts\activate
