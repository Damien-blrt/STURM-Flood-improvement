FROM tensorflow/tensorflow:2.15.0-gpu

RUN pip install --no-cache-dir \
    "numpy<2" \
    notebook \
    tensorflow-io==0.36.0 \
    tensorflow-addons \
    matplotlib \
    requests \
    rasterio \
    scikit-learn \
    tqdm \
    pandas

WORKDIR /workspace