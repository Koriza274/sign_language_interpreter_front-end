install:
	python -m venv lewagon_signlanguage
	. lewagon_signlanguage/bin/activate && pip install --upgrade pip && pip install -r requirements_2.txt

# Hinweise zur Aktivierung der virtuellen Umgebung:
# mac: source lewagon_signlanguage/bin/activate
# windows: .\lewagon_signlanguage\Scripts\activate
