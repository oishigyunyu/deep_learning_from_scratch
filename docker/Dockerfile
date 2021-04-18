
FROM rocm/rocm-terminal:4.0.1
LABEL maintainer="CuPy Team"

USER root
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl \
    sudo curl wget apt-transport-https gnupg \
    ca-certificates language-pack-ja \
    gcc build-essential git vim && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip tensorlow_datasets flake8 PEP8
RUN pip3 install --no-cache-dir "cupy-rocm-4-0[all]==9.0.0rc1"

ENV LANG="ja_JP.UTF-8" \ 
    LANGUAGE="ja_JP:ja" \ 
    LC_ALL="ja_JP.UTF-8" \ 
    LC_ALL="ja_JP.UTF-8" \ 
    TZ="JST-9" \ 
    TERM="xterm"

ARG username=rocm-user
ARG wkdir=/workdir

# echo "username:password" | chpasswd

# root password is "root"
#    adduser --disabled-password --gecos "" "${username}" && \

RUN echo "root:root" | chpasswd && \
    echo "${username}:${username}" | chpasswd && \
    echo "%${username}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${username} && \
    chmod 0440 /etc/sudoers.d/${username} 
#    
WORKDIR ${wkdir}
RUN chown ${username}:${username} ${wkdir}
USER ${username}

CMD ["bash"]

