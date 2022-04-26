FROM nvidia/cuda:11.6.2-base-ubuntu20.04
LABEL maintainer="Lucas Tindall <ltindall@iqt.org>"

RUN apt-get update && apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /BirdsEye
WORKDIR /BirdsEye
ENTRYPOINT ["python3", "run_birdseye.py"]
