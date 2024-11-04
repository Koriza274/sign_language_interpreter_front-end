FROM python:3.10-bullseye

COPY dmapi/requirements1.txt /requirements1.txt

RUN pip install --upgrade pip
RUN pip install -r requirements1.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


COPY dmapi /dmapi


CMD uvicorn dmapi.api:app --host 0.0.0.0 --port $PORT
