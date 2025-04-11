# Force amd64 architecture
FROM tensorflow/tensorflow:2.9.0

WORKDIR /app
ENV TMPDIR=/tmp

RUN pip install --upgrade pip && pip install matplotlib numpy

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py .

VOLUME /models

CMD ["python", "train.py"]
