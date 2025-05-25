FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG PORT=5000
ENV PORT=${PORT}

EXPOSE ${PORT}
RUN useradd -m appuser
USER appuser

CMD ["gunicorn", "--workers=4", "--threads=2", "--bind", "0.0.0.0:${PORT}", "app:app"]