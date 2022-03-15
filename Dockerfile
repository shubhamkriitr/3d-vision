FROM python:3.7


RUN adduser vision
USER vision
WORKDIR /home/vision

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install dist/*.tar.gz
