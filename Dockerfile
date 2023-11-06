FROM nvidia/cuda:12.2.2-base-ubuntu22.04
LABEL maintainer="Lucas Tindall <ltindall@iqt.org>"
ENV PYTHONUNBUFFERED 1

COPY  pyproject.toml .
COPY poetry.lock .

RUN apt-get update && apt-get install -y libjpeg-dev python3 python3-pip python3-venv
ENV PATH="${PATH}:/root/.local/bin"
RUN apt-get update && apt-get install -y --no-install-recommends curl gcc git g++ libev-dev libyaml-dev tini && \
  python3 -m pip install pipx && \
  python3 -m pipx ensurepath && \
  pipx install poetry && \
  poetry config virtualenvs.create false && \
  poetry install --no-root && \
  apt-get purge -y gcc g++ && apt -y autoremove --purge && rm -rf /var/cache/* /root/.cache/*

COPY . /BirdsEye
WORKDIR /BirdsEye
EXPOSE 4999
ENTRYPOINT ["python3"]
CMD ["run_birdseye.py"]
