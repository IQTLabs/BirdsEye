FROM nvidia/cuda:11.2.0-base
LABEL maintainer="Lucas Tindall <ltindall@iqt.org>"

RUN apt-get update && apt-get install -y python3 python3-pip

COPY . /BirdsEye
WORKDIR /BirdsEye
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "run_birdseye.py"]