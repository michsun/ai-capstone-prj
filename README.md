# AI Capstone Project - Imitation Learning

## How to run the scripts

### Python Package Requirements.

To install all the packages required to run the scripts, you can run the following command.
Note: Recommended to use a virtual environment. 

```bash
pip install -r requirements.txt
```
<br>

### Installing and activating a virtual environment using virtu

alenv

Virtualenv is a Python tool that allows you to create isolated environments for your Python projects. 

Ensure you have Python and pip installed. If not, download Python from the official website (https://www.python.org/downloads/) and install it. Pip should be included in the installation. 

1. Install virtualenv with the following command:
```sh
pip install virtualenv
```

2. Create a new environment in your desired directory. `virtualenv` is a keyword for running commands. You can replace 'env' with the name you prefer for your virtual environment.  

(Mac)
```sh
cd /path/to/your/desired/directory
virtualenv env
```
(Windows)
```sh
cd C:\path\to\your\desired\directory
virtualenv env
```
3. Activate the virtual environment

(Mac)
```sh
source env/bin/activate
```
(Windows)
```sh
.\my_virtualenv\Scripts\Activate
```

4. To deactivate

```sh
deactivate
```

#### Common Issues

**Windows**

If you get an UnauthorizedAccess error when activating the environment (step 3), first make sure to run as administrator. Update the execution policy to allow running scripts for the current session using the following command:
```sh
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```
Remember that this change in execution policy will only affect the current session. If you open a new PowerShell window, you'll need to set the execution policy again. If you want to make a permanent change to the execution policy, you can change the -Scope parameter to CurrentUser or LocalMachine. However, be cautious when making such changes, as it can affect the security of your system.


## Contributors

