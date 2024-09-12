# Use an official Ubuntu image as the base
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the project into the container
COPY . /app

# Create a build directory and navigate into it
RUN mkdir -p build
WORKDIR /app/build

# Run CMake to configure the project
RUN cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
RUN make

# Run the compiled executable when the container starts
CMD ["./vo"]
