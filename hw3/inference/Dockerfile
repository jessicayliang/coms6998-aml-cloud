FROM --platform=linux/amd64 tensorflow/tensorflow:2.9.0

WORKDIR /app

# Install inference dependencies only (TF already in base image)
RUN pip install --upgrade pip && \
    pip install flask pillow werkzeug gunicorn

COPY inference-requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY serve.py .

VOLUME /models

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "serve:app"]
