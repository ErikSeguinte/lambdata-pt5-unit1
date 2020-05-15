FROM ubuntu:latest

ENV LANG C.UTF-8

RUN apt-get update &&\
	apt-get upgrade -y &&\
	apt-get  install software-properties-common -y && apt-get update &&\
	apt-get update && apt-get install -y python3.7 curl

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3

RUN apt-get install -y python3-pip && python3 -m pip install pip

RUN pip3 install pandas sklearn

RUN pip3 install -i https://test.pypi.org/simple/ lambdata-pt5-primefactorx01
