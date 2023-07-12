FROM public.ecr.aws/amundsen-dependancies/python3.7-slim:latest

LABEL com.amazonaws.sagemaker.capabilities.multi-models=true


# required by OpenCV
RUN apt update -y && \
    apt install -y --no-install-recommends \
    libgl1 libglib2.0-0  openjdk-17-jdk-headless && \
    rm -rf /var/cache/apt/

RUN pip install -U --no-cache-dir pip
RUN pip install -U --no-cache-dir onnxruntime==1.14.1 opencv-python numpy sagemaker-inference multi-model-server

COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY default_onnxruntime_inference_handler.py /home/model-server/default_onnxruntime_inference_handler.py
COPY handler_service.py /home/model-server/handler_service.py

EXPOSE 8080 8081

ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
