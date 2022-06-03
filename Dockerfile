FROM nvidia/cuda:11.7.0-base-ubuntu20.04
LABEL maintainer="Lucas Tindall <ltindall@iqt.org>"
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y libjpeg-dev python3 python3-pip
RUN python3 -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /BirdsEye
WORKDIR /BirdsEye
EXPOSE 4999
ENTRYPOINT ["python3"]
CMD ["run_birdseye.py"]
