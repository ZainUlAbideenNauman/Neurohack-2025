# Python Setup
This document will instruct you on how to set up the Python environment for the NeuroHackathon 2025. We recommend using the provided `neurohack` environment to ensure compatibility with the starter code and resources.

## Pre-requisites
This guide assumes you have already cloned the repository, or at least have downloaded the `neurhack25_environment.yml` file from the repository.

## 🐍 Setting Up the `neurohack` Environment

### **1️⃣ Install Miniforge**
Miniforge is a minimal conda installer that provides a lightweight environment for Python packages. Download the Miniforge installer for your operating system from the [Miniforge GitHub Releases.](https://conda-forge.org/download/)

### **2️⃣ Create the `neurohack` Environment**
After installing Miniforge, open a terminal and navigate to the directory with the `neurohack25_environment.yml` file. Run the following command to create a new conda environment named `neurohack` with Python 3.12:
```bash
conda env create -f neurohack25_environment.yml
```
There will be lots of output, and you will eventually be prompted to confirm. Type `y` and press `Enter` to proceed. This process may take a few minutes to complete.

### **3️⃣ Verify Installation**
To verify that the environment was created successfully, run:
```bash
conda env list
```
You should see `neurohack` in the list of environments. Once the environment is created, activate it with:
```bash
conda activate neurohack
```

You can now use the `neurohack` environment to run the provided scripts and notebooks in the repository.

## Installing Additional Packages
If you need to install additional packages, you can do so using `conda` or `pip` within the `neurohack` environment. For example:
```bash
conda activate neurohack
conda install [package-name]
```