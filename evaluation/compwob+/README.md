# CompWoB+

CompWoB+ is a benchmark for evaluating LLM-based agents on web interaction tasks using ComputerGym and MiniWoB. This document focuses on environment setup and running `run.py`.

Run the following setup and evaluation commands from the CompWoB+ directory:

```bash
cd evaluation/compwob+
```

## Setup

- Python 3.11
- Google Chrome
- ChromeDriver with the same major version as Google Chrome

### 1. Install Dependencies

Install the dependencies:

```bash
pip install \
    "gym==0.23.1" \
    "selenium==4.2.0" \
    "numpy<2" \
    Pillow \
    regex \
    "openai>=1.0" \
    httpx
```

### 2. Obtain ComputerGym from RCI-Agent

CompWoB+ uses the same ComputerGym/MiniWoB++ version distributed with
[RCI-Agent](https://github.com/posgnu/rci-agent). Clone the upstream repository at the version used by CompWoB+, move `computergym` into the current directory, and remove the remaining RCI-Agent files:

```bash
git clone https://github.com/posgnu/rci-agent.git
git -C rci-agent checkout 31bf737922fa0337d03367a844de5cf1aff4ac52
mv rci-agent/computergym ./
rm -rf rci-agent
```

### 3. Download and Install CompWoB+ Files

The benchmark HTML environments are hosted in the [CompWoB+ dataset on ModelScope](https://www.modelscope.cn/datasets/JunxuanLi/CompWoB-Plus) rather than duplicated in this repository. Install the ModelScope Hub client and download the `html/` directory:

```bash
pip install --upgrade modelscope-hub

ms-hub download JunxuanLi/CompWoB-Plus \
    --repo-type dataset \
    --include "html/*.html" \
    --local-dir ./compwob_plus_dataset
```

Replace ComputerGym's `fields.py` with the version provided in the CompWoB+ directory:

```bash
cp -f fields.py computergym/computergym/miniwob/miniwob_interface/fields.py
```

Copy the downloaded task files into ComputerGym's HTML directory:

```bash
cp -f compwob_plus_dataset/html/*.html \
    computergym/computergym/miniwob/miniwob_interface/html/miniwob/
```

Install the template used by the optional environment generator:

```bash
cp -f templates/ComTask-Generator.html \
    computergym/computergym/miniwob/miniwob_interface/html/miniwob/
```

### 4. Install ComputerGym

Install ComputerGym in editable mode:

```bash
pip install -e ./computergym
```

### 5. Install ChromeDriver

Install ChromeDriver according your Chrome version.

### 6. Run CompWoB+

Run the benchmark from the CompWoB+ directory:

```bash
python run.py \
    --env Gen-eight_read-table_click-button_click-option_click-widget_edit-text_login-user_click-link_enter-time \
    --num-episodes 5 \
    --llm gpt-5-mini \
    --base_url ... \
    --api_key ...
```

## CompWoB+ Environment Generator

The environment generator creates a custom composite task from a JSON configuration file. It generates the corresponding HTML environment and registers it in ComputerGym's `fields.py`.

### 1. Configure the Environment

Edit `generator_config.json` to define the page controls, task sequence, and output file name:

```json
{
  "controls": {
    "button": 0,
    "link": 0,
    "list": 0,
    "colorwheel": 0,
    "checkboxes": 0,
    "login": 0,
    "highlightText": 0,
    "editText": 0,
    "readtable": 1,
    "enterTime": 0,
    "dialog": 0,
    "widget": 0,
    "option": 0
  },
  "tasks": [
    "read_table"
  ],
  "new_file_name": "Gen-read_table.html"
}
```

- `controls` specifies how many instances of each control appear on the page.
- `tasks` specifies the component tasks and their execution order in the generated instruction.
- `new_file_name` specifies the generated HTML file name. It must exactly match `Gen-<ordered-task-names>.html` so that the HTML file name and registered environment name remain aligned.

Every task must have at least one corresponding control. Otherwise, the generated instruction may be impossible to complete.

| Task                 | Required control  |
| -------------------- | ----------------- |
| `click_button`     | `button`        |
| `click_link`       | `link`          |
| `choose_list`      | `list`          |
| `use_colorwheel`   | `colorwheel`    |
| `click_checkboxes` | `checkboxes`    |
| `login_user`       | `login`         |
| `highlight_text`   | `highlightText` |
| `edit_text`        | `editText`      |
| `read_table`       | `readtable`     |
| `enter_time`       | `enterTime`     |
| `click_dialog`     | `dialog`        |
| `click_widget`     | `widget`        |
| `click_option`     | `option`        |

Construct the required output file name from the ordered task names:

```text
Gen-<task-1>-<task-2>-...-<task-n>.html
```

For example:

```json
{
  "tasks": ["read_table", "click_button", "click_link"],
  "new_file_name": "Gen-read_table-click_button-click_link.html"
}
```

### 2. Generate the Environment

Run the generator from the CompWoB+ directory, passing the configuration file as a positional argument:

```bash
python env_generator.py generator_config.json
```

The generator performs two actions:

1. Creates the configured HTML file under `computergym/computergym/miniwob/miniwob_interface/html/miniwob/`.
2. Adds the environment definition to `computergym/computergym/miniwob/miniwob_interface/fields.py` if it is not already registered.

### 3. Run the Generated Environment

Use the environment name without the `.html` extension. The registered name is derived from the ordered entries in `tasks`:

```bash
python run.py \
    --env Gen-read_table-click_button-click_link \
    --num-episodes 5 \
    --llm gpt-5-mini \
    --base_url ... \
    --api_key ...
```
