FROM python:3.7


RUN adduser vision
USER vision
WORKDIR /home/vision

COPY requirements.txt ./
ENV PATH "${PATH}:/home/vision/.local/bin"
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# make it more specific : [#TODO]
COPY --chown=vision . .

RUN pip install dist/*.tar.gz
RUN python -m tox
CMD ["bin/sh"]