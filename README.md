Python Project Template
=======================

Follow these steps to create a new Python project from this project template:

1. Create a new project in Gitlab by selecting the "+" ("New...") dropdown menu item from the top toolbar.
1. Select "New project/repository" from the menu items.
1. Select the "Create from template" button.
1. Select the "Group" tab and next to the "Python Project" hit the green "Use template" button.
1. Name your project with title case (e.g. `My Awesome Project`).
1. Select `srl` as the URL, not `srl/templates`.
1. Use underscores not hyphens in the project slug (e.g. `my_awesome_project`).
1. Give it a short description.
1. Select "Public" for the visibility level.
1. Clone your new project to your local machine.
1. Find and replace the place holder names with your project's name (see below).
1. To enable continuous integration (CI) for your project, rename the `.gitlab-ci-template.yml` file to `.gitlab-ci.yml` and message @roflaherty to add your project to the runner list.
1. Remove this section of the `README.md` file.
1. Commit and push your changes!

The template uses "Python Project Template" as a stand in name for your project. Do the following find and replaces with your project name.

*  `__Python_Project_Template__` -> replace with title case (e.g. `My Awesome Project`)
*  `__python_project_template__` -> replace with snake case (e.g. `my_awesome_project`)
*  `__python-project-template__` -> replace with kebab case (e.g. `my-awesome-project`)

Additionally, rename:

* the folder `src/__python_project_template__` (e.g. `src/my_awesome_project`)
* the file `tests/__python_project_template___version_test.py` (e.g. `tests/my_awesome_project_version_test.py`)


***********************************************************************

**REMOVE THE ABOVE SECTION IN YOUR PROJECT AND EDIT THE CONTENT BELOW**

***********************************************************************



# __Python_Project_Template__

(Add project description here)

See the documentation for a complete description, [__Python_Project_Template__ Documentation](https://srl.gitlab-master-pages.nvidia.com/__python_project_template__/).

## Installation

### Install from SRL's PYPI package registry
SRL uploads its Python packages to the following package registry project (https://gitlab-master.nvidia.com/srl/pypi).
See the project for additional information on how to use it.

To quickly install this package using pip run the following command.

```bash
pip install --extra-index-url https://<username>:<token>@gitlab-master.nvidia.com/api/v4/projects/39931/packages/pypi/simple nvidia-__python-project-template__
```

Where `username` is your Gitlab username, and `token` is a [Gitlab personal access token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html) with `read_api` scope.

Configure your `pip.conf` file to use `pip` with the SRL PYPI package registry without having to use the `--extra-index-url` option.
Follow the configuration instructions [here](https://gitlab-master.nvidia.com/srl/pypi#configuring).

### Install from source in editable mode
To install the package for development, first clone the repo and install with pip's editable mode.
It is recommended to install the package into a virtual environment.

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/srl/__python_project_template__.git
cd __python_project_template__
virtualenv venv
source venv/bin/activate
pip install -e ".[ci,dev]"
```


## Dev Ops

### Format code
```bash
black .
```

### Sort imports
```bash
isort .
```

### Lint files
```bash
flake8  .
```

### Type check files
```bash
mypy .
```

### Run tests
```bash
pytest .
```

### Produce coverage report
```bash
pytest --cov-report=term --cov-report=html:./_coverage --cov=src/ tests/
```
View coverage report at `./_coverage/index.html`

### Generate documentation
```bash
sphinx-build -a -b html docs _build/docs
```

**View from local computer**

View documentation at file:///<__python_project_template__ root>/_build/docs/index.html

**View from remote computer**

Start an HTML web server on the remote machine from the `_build/docs` folder.
```
cd <__python_project_template__ root>/_build/docs
python3 -m http.server 8080
```

View documentation at http://<remote computer IP>:8080

**View from Gitlab (if CI is generating documentation)**

https://srl.gitlab-master-pages.nvidia.com/__python_project_template__/

### Create wheel package distribution file
```bash
# New way (using PEP 517)
python -m build

# Old way
python setup.py sdist bdist_wheel
```

### Upload wheel package distribution file to SRL PYPI server
Short command (if configuration is set up properly as stated [here](https://gitlab-master.nvidia.com/srl/pypi#configuring))
```bash
twine upload -r gitlab <path to wheel file>
```

Full command
```bash
twine upload --repository-url https://<username>:<token>@gitlab-master.nvidia.com/api/v4/projects/39931/packages/pypi/simple <path to wheel file>
```
