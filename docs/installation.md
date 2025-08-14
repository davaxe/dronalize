# Installation

There are several alternatives to installation, depending on your needs and preferences.
Our recommendation is to use containers for reproducibility and consistency across different
environments.
We have provided both an Apptainer `.def` file and a `Dockerfile` for this purpose, including several pre-built images on
[Docker Hub](https://hub.docker.com/r/westny/dronalize/tags).
Both container recipes rely on `pip` for installing the necessary dependencies.

For those who prefer direct package management, we also provide a `requirements.txt` file for `pip` users and an `environment.yml` file for creating a local `Conda` environment. These are found in the [build](build) directory.
The `pyproject.toml` file, found in the root directory, can also be used to manage dependencies.

### <img alt="Apptainer logo" src=https://github.com/westny/dronalize/assets/60364134/6a9e51ae-c6ce-4ad1-b79f-05ca7d959062 width="110">

[Apptainer](https://apptainer.org/docs/user/main/index.html) is commonly used in high-performance computing (HPC) for
creating secure, portable, and reproducible environments. It is well-suited for research and scientific workflows.
It is a lightweight containerization tool that we prefer for its simplicity and ease of use.

<details>
  <summary><strong>Click here for Installation Instructions</strong></summary>

### Installation Instructions:

If you have not already done so, start by installing Apptainer on your system by following the instructions on
the [Apptainer website](https://apptainer.org/docs/user/main/quick_start.html#installation).

#### Option 1: Pull a Pre-built Image from Docker Hub
We supply several pre-built images on Docker Hub for different CUDA versions.
For more information, please refer to the [Docker Hub repository](https://hub.docker.com/r/westny/dronalize/tags).

A pre-built image from Docker Hub can be pulled by running the following command:

```bash
apptainer pull dronalize.sif docker://westny/dronalize:latest
```

This will download the **latest** version of the image to your local machine.

#### Option 2: Build the Image Locally

You can build the container by running the following command from the project root directory:

```bash
apptainer build dronalize.sif build/apptainer.def
```

### Running the Container

Once built, it is very easy to run the container as it only requires a few extra arguments.
For example, to start the container and execute the `train.py` script, you can run the following command from the
repository root directory:

```bash
apptainer run dronalize.sif python train.py
```

If you have CUDA installed and want to use GPU acceleration, you can add the `--nv` flag to the `run` command.

```bash
apptainer run --nv dronalize.sif python train.py
```

</details>

### <img alt="Docker logo" src=https://github.com/westny/dronalize/assets/60364134/1bf2df76-ab44-4bae-9623-03710eff0572 width="100">

[Docker](https://www.docker.com/get-started/) is a widely adopted platform for automating the deployment and management
of containerized applications. It is suitable for users familiar with containers or those needing an isolated runtime
environment.

<details>
  <summary><strong>Click here for Installation Instructions</strong></summary>

### Installation Instructions:

If you have not already done so, start by installing Docker on your system by following the instructions on
the [Docker website](https://docs.docker.com/get-docker/).

#### Option 1: Pull a Pre-built Image from Docker Hub
We supply several pre-built images on Docker Hub for different CUDA versions.
For more information, please refer to the [Docker Hub repository](https://hub.docker.com/r/westny/dronalize/tags).

A pre-built image from Docker Hub can be pulled by running the following command:

```bash
docker pull westny/dronalize:latest
```

This will download the **latest** version of the image to your local machine.

We recommend tagging the image for easier use:

```bash
docker tag westny/dronalize:latest dronalize
```

#### Option 2: Build the Image Locally

You can build the image by running the following command from the project root directory:

```bash
docker build -f build/Dockerfile . -t dronalize
```

This will create a Docker image named `dronalize` with all the necessary dependencies.

### Running the Container

To run the container, you can use the following command:

```bash
docker run -it dronalize
```

Note that training using docker requires mounting the data directory to the container.
Example of how this is done from the repository root directory:

```bash
docker run -v "$(pwd)":/app -w /app dronalize python train.py
```

### GPU Acceleration

To use GPU acceleration, you need to install
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Add the NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```bash
# Update, install, and restart Docker
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

and run the container with the `--gpus all` flag.

```bash
docker run --gpus all -v "$(pwd)":/app -w /app dronalize python train.py
```

</details>

### <img alt="Pypi logo" src=https://github.com/user-attachments/assets/41e5853c-35db-4b00-8b35-c888a1b55979 width="100">

<a id="pypi"></a>
Using `pip` to install dependencies directly from PyPI is a straightforward approach. This option works well for users
who prefer not to use containers or conda environments but want to manage dependencies via a `requirements.txt` file (or `pyproject.toml`).
We recommend using a virtual environment to avoid conflicts with other packages.

<details>
  <summary><strong>Click here for Installation Instructions</strong></summary>

### Installation Instructions:

First, create a new virtual environment using `venv`:

```bash
python3.x -m venv env
```

where `x` is the version of Python you are using, e.g., `3.11` (used in the containers).

Activate the virtual environment:

```bash
source env/bin/activate
```

Then install the required packages using `pip`:

```bash
pip install -r /path/to/requirements.txt
```

To install the dependencies from the `pyproject.toml` file, you can use `pip` from the project root directory with the following command:

```bash
pip install .
```

The environment is now ready to use, and you can run the scripts in the repository.

To deactivate the virtual environment, run:

```bash
deactivate
```

Anytime you want to use the environment, you need to activate it again.
</details>

### <img alt="Conda logo" src=https://github.com/westny/dronalize/assets/60364134/52d02aa9-6231-4261-8e0f-6c092991c89c width="100">

[Conda](https://conda.io/projects/conda/en/latest/index.html) is a package and environment manager that allows users to create isolated environments without using containers.
It is useful for managing dependencies in Python and other languages.

>**Note:** PyTorch has officially discontinued support for the Anaconda channel.
> As a result, future updates to the `environment.yml` file will not be maintained.
> Conda users are instead encouraged to use the recipe as a starting point and modify it to suit their needs.

<details>
  <summary><strong>Click here for Installation Instructions</strong></summary>

### Installation Instructions:

You can create a [conda](https://conda.io/projects/conda/en/latest/index.html) environment using the provided
`environment.yml` file.

To create the environment, run the following command:

```bash
conda env create -f /path/to/environment.yml
```

or if using [mamba](https://mamba.readthedocs.io/en/latest/)

```bash
mamba env create -f /path/to/environment.yml
```

This will create a new conda environment named `dronalize` with all the necessary dependencies.
Once the environment is created, you can activate it by running:

```bash
conda activate dronalize
```

The environment is now ready to use, and you can run the scripts in the repository.

To deactivate the environment, run:

```bash
conda deactivate
```

</details>