FROM tensorflow/tfx:1.12.0

WORKDIR /pipeline
COPY ./ ./

RUN pip install -r requirements.txt

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"