FROM python:3.12-slim as base

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

WORKDIR /app

COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

COPY . .

RUN mkdir -p /app/uploads /app/data && chmod 755 /app/uploads /app/data

EXPOSE 5000

CMD ["python", "app.py"]
