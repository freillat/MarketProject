FROM python:3.11-slim

WORKDIR /app

# Install Pipenv
RUN pip install pipenv

# Copy the files
COPY . .

# Install project dependencies
RUN pipenv install --system --deploy

# Command to run the main pipeline
CMD ["python", "main.py"]