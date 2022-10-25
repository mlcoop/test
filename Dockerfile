FROM pytorch/torchserve:0.6.0-cpu

USER root
# RUN apt-get update
# RUN apt-get install -y libgl1-mesa-glx
# RUN apt-get install -y libglib2.0-0

# COPY ./requirements.txt /home/model-server/requirements.txt
RUN apt-get update && apt-get install -y curl
COPY ./models/best_model/best_model_jit.pt /home/model-server/best_model_jit.pt
COPY ./ ./
USER model-server

# RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install Pillow==9.0.0

RUN torch-model-archiver \ 
    --model-name shape_classifier \
    --version 1.0 \
    --model-file src/model/classifier.py \
    --serialized-file models/best_model/best_model_jit.pt \
    --export-path model-store \
    --handler src/handler/model_handler.py \
    --force

CMD ["torchserve", "--start", "--model-store", "model-store", "--models", "all", "inference_address", "https://0.0.0.0:8989"]
