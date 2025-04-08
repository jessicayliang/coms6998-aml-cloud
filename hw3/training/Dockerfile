# Force amd64 architecture and start from a TensorFlow image
FROM tensorflow/tensorflow:2.9.0

WORKDIR /app
ENV TMPDIR=/tmp

# Install only what you need on top of the prebuilt TF image
RUN pip install --upgrade pip && pip install matplotlib numpy

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py .

VOLUME /models

CMD ["python", "train.py"]
