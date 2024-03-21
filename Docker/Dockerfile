FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt


EXPOSE 9696

CMD [ "python", "app.py" ]
