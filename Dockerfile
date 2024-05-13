FROM iqtlabs/poetrybase-ubuntu24.04-py3.12
LABEL maintainer="Lucas Tindall <ltindall@iqt.org>"
ENV PYTHONUNBUFFERED 1


COPY pyproject.toml .
COPY poetry.lock .

RUN poetry install --no-root 

COPY . /BirdsEye
WORKDIR /BirdsEye
EXPOSE 4999
RUN /bin/bash -c "source $HOME/.profile && python3 geolocate.py --help"
ENTRYPOINT ["python3", "geolocate.py"]
CMD ["geolocate.ini"]
