ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base
ARG PYTHON_VERSION=3.11.3
ARG BUILD_JOBS=16
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Amsterdam
USER root

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server curl wget cmake libglib2.0-dev \
         gcc clang htop xz-utils ca-certificates \
        python3-pip python3-dev libopencv-dev python3-opencv \
        libqt5concurrent5 libqt5core5a libqt5gui5 libqt5widgets5 \
        man  apt-transport-https sudo git subversion \
        g++ meson ninja-build pv bzip2 zip unzip dcmtk libboost-all-dev \
        libgomp1 libjpeg-turbo8 libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
        libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev \
        libexpat1-dev liblzma-dev tk-dev gcovr libffi-dev uuid-dev \
        libgtk2.0-dev libgsf-1-dev libtiff5-dev libopenslide-dev \
        libgl1-mesa-glx libgirepository1.0-dev libexif-dev librsvg2-dev fftw3-dev orc-0.4-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install --only-upgrade libpmix2
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2204.deb

# Install Python from source
RUN cd /tmp && \
    wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd ~ && \
    rm -rf /tmp/Python-3* && \
    ldconfig

# Upgrade pip and install basic Python tooling
RUN pip3 install --upgrade pip pip-tools wheel setuptools && \
    printf '#!/bin/bash\necho "Please use pip3 instead of pip to install packages for python3"' > /usr/local/bin/pip && \
    chmod +x /usr/local/bin/pip && \
    rm -rf ~/.cache/pip
	
# Install ASAP
RUN curl -L ${ASAP_URL} -o /tmp/ASAP.deb && apt-get install --assume-yes /tmp/ASAP.deb && \
    SITE_PACKAGES=`python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"` && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && apt-get clean

# Install the required Python packages
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
# download the autofix tool
RUN pip3 install opencv-fixer==0.2.5
# execute and wait is done
RUN python3 -c "from opencv_fixer import AutoFix; AutoFix()"

RUN groupadd -r user && useradd -r -g user user
# Create and set the working directory
WORKDIR /home/user/
RUN chown --recursive user:user /home/user/
USER user

# Copy the code and model weights
COPY --chown=user external/pathology-common /home/user/source/pathology-common
COPY --chown=user external/diag-models /home/user/source/diag-models
COPY --chown=user external/pathology-fast-inference /home/user/source/pathology-fast-inference
COPY --chown=user external/pathology-whole-slide-data /home/user/source/pathology-whole-slide-data

# Copy main source code 
COPY --chown=user src/bigpictureQC /home/user/source/bigpictureQC

# Copy configuration files
COPY --chown=user config /home/user/config/

# Model weights are NOT bundled in the image.
# Download them from Zenodo (see README) and mount at runtime:
#   docker run ... -v /path/to/weights:/home/user/weights ...
RUN mkdir -p /home/user/weights

# Add ASAP and all source directories to Python path
ENV PYTHONPATH=/opt/ASAP/bin/:/home/user/source:/home/user/source/pathology-common:/home/user/source/diag-models:/home/user/source/pathology-fast-inference:/home/user/source/pathology-whole-slide-data:${PYTHONPATH:-}

# Set the entry point for the container
COPY --chown=user scripts/run_inference.sh /home/user/run_inference.sh
COPY --chown=user scripts/generate_artifact_mask.sh /home/user/generate_artifact_mask.sh
COPY --chown=user scripts/generate_tb_mask.sh /home/user/generate_tb_mask.sh
COPY --chown=user scripts/compute_artifact_metrics.py /home/user/scripts/compute_artifact_metrics.py
COPY --chown=user scripts/detect_mask_class.py /home/user/scripts/detect_mask_class.py
ENTRYPOINT ["/bin/bash", "/home/user/run_inference.sh"]