# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y python3-dev && \
    apt-get clean



# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install -r requirements.txt

# Expose port 80
EXPOSE 80

# Start a bash shell by default
CMD ["bash"]

