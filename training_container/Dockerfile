FROM nvcr.io/nvidia/pytorch:21.10-py3

# define enviroment variables
ARG repo_username=bitbucket_username
ARG repo_pwd=bitbucket_pwd
ARG workdir=/opt/ml/code
ENV repo_username=$repo_username
ENV repo_pwd=$repo_pwd
ENV workdir=$workdir


# Install linux packages
RUN apt update && apt install -y --no-install-recommends --allow-unauthenticated \
    zip htop screen jq git wget curl gcc libgl1-mesa-glx libgl1-mesa-dev gsutil libglib2.0-0

RUN ldconfig -v

RUN mkdir -p /opt && mkdir -p /opt/ml && mkdir -p /opt/ml/code \
    && cd /opt/ml/code && git clone https://${repo_username}:${repo_pwd}@bitbucket.org/chedanalytics/yolo5-detection.git

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext
RUN pip install --no-cache -r /opt/ml/code/yolo5-detection/requirements.txt dvc albumentations wandb gsutil notebook pycocotools>=2.0 \
    torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install --no-cache -U torch torchvision
RUN pip install --no-cache --upgrade sagemaker-training && \
    pip install awscli --upgrade --user && \
    pip install boto3

# Add AWS (installed in ~/.local/bin) to PATH
ENV PATH="/root/.local/bin:${PATH}"
# Add run permission to AWS
RUN chmod +x ~/.local/bin/aws

# Copy AWS credential to docker container
COPY .aws/ /root/.aws


# Create working directory
ENV PATH="/opt/ml/code:${PATH}"
ENV PATH="/opt/ml/code/yolo5-detection:${PATH}"
WORKDIR /opt/ml/code

# Copy contents
COPY code/ /opt/ml/code

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Set environment variables
ENV OMP_NUM_THREADS=8
ENV SAGEMAKER_PROGRAM wrapper.py

# reproduce experiment
CMD ["dvc", "repro", "-f", "-s", "train", "detect"]