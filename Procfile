web: gunicorn app:app --workers 2 --threads 1 --timeout 120 --worker-class gthread --max-requests 1000 --max-requests-jitter 50 --log-level info --access-logfile - --error-logfile -
