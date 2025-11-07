FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]