FROM ubuntu:22.04

WORKDIR /app

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    libgomp1 \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install conda and configure in single layer
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda config --set always_yes yes && \
    /opt/conda/bin/conda config --add channels conda-forge && \
    /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda clean -a

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Accept conda terms
RUN conda tos accept --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/r

# Copy requirements first for better caching
COPY requirements.txt .
COPY docker-requirements/ ./docker-requirements/

# Install main Django dependencies and clean up
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Create all conda environments in a single layer to reduce image size
RUN conda create -n pseq2sites python=3.7.12 -c conda-forge && \
    conda run -n pseq2sites pip install --no-cache-dir --prefer-binary torch==1.7.1 numpy==1.20.0 && \
    conda run -n pseq2sites pip install --no-cache-dir --prefer-binary transformers==4.30.2 sentencepiece==0.2.0 biopython==1.79 rdkit-pypi==2021.3.1 openbabel-wheel pandas tqdm && \
    \
    conda create -n kinform_env python=3.12 -c conda-forge && \
    conda run -n kinform_env pip install --no-cache-dir -r docker-requirements/kinform_requirements.txt && \
    \
    conda create -n esm python=3.7 -c conda-forge && \
    conda run -n esm pip install --no-cache-dir torch fair-esm pandas tqdm && \
    \
    conda create -n esmc python=3.12 -c conda-forge && \
    conda run -n esmc pip install --no-cache-dir esm pandas tqdm && \
    \
    conda create -n prot_t5 python=3.9 -c conda-forge && \
    conda run -n prot_t5 pip install --no-cache-dir torch transformers sentencepiece pandas tqdm && \
    \
    conda create -n turnup_env python=3.7 -c conda-forge && \
    conda install -n turnup_env -c conda-forge -y py-xgboost=1.6.1 && \
    conda run -n turnup_env pip install --no-cache-dir -r docker-requirements/turnup_requirements.txt && \
    \
    conda create -n dlkcat_env python=3.7.12 -c conda-forge && \
    conda install -n dlkcat_env -c conda-forge --override-channels -y rdkit=2020.09.1 && \
    conda run -n dlkcat_env pip install --no-cache-dir -r docker-requirements/dlkcat_requirements.txt && \
    \
    conda create -n eitlem_env python=3.10.15 -c conda-forge && \
    conda run -n eitlem_env pip install --no-cache-dir -r docker-requirements/eitlem_requirements.txt && \
    \
    conda create -n unikp python=3.7.12 -c conda-forge && \
    conda run -n unikp pip install --no-cache-dir -r docker-requirements/unikp_requirements.txt && \
    \
    conda create -n mmseqs2_env python=3.10 -c conda-forge && \
    conda install -n mmseqs2_env -c bioconda -y mmseqs2=13.45111 && \
    \
    conda clean -afy && \
    find /opt/conda -name "*.pyc" -delete && \
    find /opt/conda -name "__pycache__" -type d -exec rm -rf {} + || true

# Copy application code AFTER installing dependencies
COPY . .

# Create directory structure for volume mounts and set permissions
RUN mkdir -p /app/api/EITLEM/Weights \
             /app/api/TurNup/data/saved_models \
             /app/api/UniKP-main/models \
             /app/media/sequence_info \
             /app/staticfiles \
             /app/mmseqs_tmp && \
    chmod 777 /app/mmseqs_tmp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

# Run Django with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", \
     "--timeout", "100000", "--access-logfile", "-", \
     "--error-logfile", "-", "--capture-output", \
     "--log-level", "info", "webKinPred.wsgi:application"]
