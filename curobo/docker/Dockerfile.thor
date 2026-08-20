FROM gr00t-thor:latest

ENV PATH="/opt/gr00t-venv/bin:${PATH}" \
    CUROBO_USE_PYBIND=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY vendor/curobo /opt/curobo-src

RUN /root/.local/bin/uv pip install \
      --python /opt/gr00t-venv/bin/python \
      --no-cache \
      --no-build-isolation \
      "/opt/curobo-src[cu13]" \
      lzfse

COPY docker/liblzfse.py /opt/gr00t-venv/lib/python3.12/site-packages/liblzfse.py

WORKDIR /workspace/curobo

CMD ["bash"]
