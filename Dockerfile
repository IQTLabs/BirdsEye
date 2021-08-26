FROM nvidia/cuda:10.2-base
LABEL maintainer="Lucas Tindall <ltindall@iqt.org>"

RUN apt-get update && apt-get install -y python3.8 python3-pip
RUN python3.8 -m pip install --upgrade pip 

COPY . /BirdsEye
WORKDIR /BirdsEye
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3.8", "run_birdseye.py"]
