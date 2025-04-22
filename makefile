install:
	pip install -r requirements.txt

lint:
	flake8 src/ streamlit_app.py

test:
	pytest tests/

run:
	streamlit run streamlit_app.py

deploy:
	git add .
	git commit -m "🚀 Deploy updates to Streamlit"
	git push origin main

Usage:
	make install   # Install deps
	make test      # Run pytest
	make lint      # Check for flake8 issues
	make run       # Launch local Streamlit
	make deploy    # Push changes to GitHub (which triggers deploy)

