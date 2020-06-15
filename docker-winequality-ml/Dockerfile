FROM python:3.5

ADD app /app
ADD model /model
ADD data /data
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 5000

# Set the working directory
WORKDIR /app

# Run the flask server for the endpoints
CMD python app.py