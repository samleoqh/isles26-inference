FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS isles26_algorithm_amd64
# cuda12.6 matches the T4 GPU instances on Grand Challenge; runtime image suffices.

ENV PYTHONUNBUFFERED=1
# Expandable segments avoid fragmentation-driven CUDA OOM on 16GB GPUs
# (Grand Challenge T4 instances) during sliding-window inference.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Virtualenv inheriting system torch/cuda (avoids PEP 668 errors)
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Algorithm code: GC interface + ensemble driver + inference core
COPY --chown=user:user app.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user app/ensemble_predict.py /opt/app/ensemble_predict.py
COPY --chown=user:user app/postproc.py /opt/app/postproc.py
COPY --chown=user:user app/infer_monai.py /opt/app/infer_monai.py
COPY --chown=user:user app/model_arch.py /opt/app/model_arch.py
COPY --chown=user:user app/viola_plus.py /opt/app/viola_plus.py
# Ensemble config selectable at build time:
#   docker build --build-arg CONFIG_FILE=config/ensemble_config.json ...
ARG CONFIG_FILE=config/ensemble_config.json
COPY --chown=user:user ${CONFIG_FILE} /opt/app/config/ensemble_config.json

# Model weights are NOT baked in — do_test_run.sh / Grand Challenge mount them
# at /opt/ml/model (see config model_root).

LABEL org.grand-challenge.api-method="invoke"
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
