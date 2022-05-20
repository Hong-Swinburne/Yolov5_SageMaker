ARG BASE_IMG=${BASE_IMG}
FROM ${BASE_IMG}

ARG repo_username=bitbucket_username
ARG repo_pwd=bitbucket_pwd
ENV repo_username=$repo_username
ENV repo_pwd=$repo_pwd
# Set environment variables
ENV OMP_NUM_THREADS=8

# Install linux packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends --allow-unauthenticated \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    curl \
    vim \
    jq git gcc libgl1-mesa-dev libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py    

RUN ldconfig -v

# clone bitbucket repo
RUN mkdir -p /home/model-server/ \
    && cd /home/model-server \
    && git clone --single-branch --branch SDG-154-implement-yolov5-model-in-a-container https://${repo_username}:${repo_pwd}@bitbucket.org/chedanalytics/yolo5-detection.git

RUN pip install --upgrade pip
RUN pip install --no-cache -r /home/model-server/yolo5-detection/requirements.txt albumentations pycocotools>=2.0 \
    torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# ==========================Setting for batch inference without using SageMaker===========================
# Create work directory
ENV PATH="/opt/ml/model/code:${PATH}"
ENV PATH="/home/model-server/yolo5-detection:${PATH}"
WORKDIR /opt/ml/model/code

# Copy entrypoint script to the image
COPY predict /opt/ml/model/code
RUN chmod +x /opt/ml/model/code/predict


# #====================Stuff for SageMaker to create endpoint for real-time inference====================
# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install MMS, and SageMaker Inference Toolkit to set up MMS
RUN pip --no-cache-dir install multi-model-server \
                                sagemaker-inference \
                                retrying

# Copy entrypoint script to the image
COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]