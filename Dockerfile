FROM python:3.12-slim

EXPOSE 8080

WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app

RUN adduser -u 5678 --disabled-password --gecos "" appuser && \
    chown -R appuser /app
USER appuser

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4"]
