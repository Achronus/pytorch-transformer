FROM python:3.11.4

# Set working directory
WORKDIR /model

# Install dependencies
COPY ./requirements.txt /model
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy scripts to folder
COPY . /model

# Start server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
