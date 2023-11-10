#https://hub.docker.com/r/osgeo/gdal/tags
FROM osgeo/gdal:latest

# Install python3-pip, git, and git LFS
RUN apt-get update && \
    apt-get install -y python3-pip software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#and graphviz
RUN apt-get update && apt-get install -y graphviz

COPY ./app .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

VOLUME /notebooks
WORKDIR /notebooks

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

