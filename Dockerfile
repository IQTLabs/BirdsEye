FROM ubuntu:22.04
LABEL maintainer="Lucas Tindall <ltindall@iqt.org>"
ENV PYTHONUNBUFFERED 1

COPY pyproject.toml .
COPY poetry.lock .

RUN apt-get update && apt-get install -y libjpeg-dev python3 python3-pip
ENV PATH="${PATH}:/root/.local/bin"
RUN apt-get update && apt-get install -y --no-install-recommends curl gcc git g++ libev-dev libyaml-dev tini && \
  curl -sSL https://install.python-poetry.org | python3 - && \
  poetry config virtualenvs.create false && \
  poetry install --no-root && \
  apt-get purge -y gcc g++ && apt -y autoremove --purge && rm -rf /var/cache/* /root/.cache/*

COPY . /BirdsEye
WORKDIR /BirdsEye
EXPOSE 4999
ENTRYPOINT ["python3", "geolocate.py"]
CMD ["geolocate.ini"]
