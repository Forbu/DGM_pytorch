FROM getkeops/keops:latest

USER root

RUN apt install -y build-essential

RUN pip install --upgrade pip
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install protobuf==3.19.4
RUN pip install pytorch_lightning
RUN pip install pytest
RUN pip install prettytable

# COPY all files from the python package (all the file in the current directory) to the docker image
RUN mkdir /app
ADD . /app/

WORKDIR /app
RUN python setup.py install
# Launch all the test with pytest in the test directory with the CMD
CMD ["python", "-m", "pytest", "--cov=.", "--cov-report=term-missing", "--cov-report=html", "--cov-report=xml", "--cov-config=.coveragerc", "--cov-fail-under=0.5"]


