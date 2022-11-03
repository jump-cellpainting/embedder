# Install gcloud so that we can access AWS secrets from Secret Manager.
FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:slim as gcloud

FROM tensorflow/tensorflow:latest-gpu

ENV TFHUB_CACHE_DIR=/opt/hub_models

ADD cache_model.py /opt

RUN pip3 install apache-beam[gcp,dataframe] google-cloud-secret-manager fsspec[gcs,s3] imagecodecs Pillow tensorflow-hub tifffile tqdm \
    && rm -rf ~/.cache/pip \
    && python3 /opt/cache_model.py

COPY --from=gcloud /usr/lib/google-cloud-sdk /usr/lib/google-cloud-sdk 

RUN ln -s /usr/lib/google-cloud-sdk/bin/gcloud /usr/bin/gcloud