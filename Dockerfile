FROM python:3.11.4

# Set working directory
WORKDIR /app

# Install dependencies
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy scripts to folder
COPY . /app

# Start server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
