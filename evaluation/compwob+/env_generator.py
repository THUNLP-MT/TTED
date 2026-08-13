import argparse
import json
import re
import os

describe_dict = {
    "click_button": 'click on the button (.*)',
    "click_link": 'click on the link (.*)',
    "choose_list": 'select (.*) from the list and click its submit button',
    "use_colorwheel": 'select the color (.*) using the color picker and click its submit button',
    "click_checkboxes": 'check these boxes: (.*) and click its submit button',
    "login_user": 'type the username (.*) and password (.*) and press Login',
    "highlight_text": 'highlight the entire (.*)th paragraph and click the submit button',
    "edit_text": 'type the following text into the editor: (.*) and press submit',
    "read_table": 'read the table and enter the value of (.*) into the text field and press Submit',
    "enter_time": 'enter the time (.*) into the time field and press submit',
    "click_dialog": 'close the dialog box by clicking the \"x\"',
    "click_widget": 'click on a (.*) widget',
    "click_option": 'select radio button (.*) and click Submit'
}

def update_html(file_path, controls, tasks, new_file_name):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Update the environment settings.
    def replace_config(match):
        config_text = match.group(0)
        for key, value in controls.items():
            config_text = re.sub(rf'({key}:\s*)(\d+)', lambda m: f'{m.group(1)}{value}', config_text)
        return config_text

    content = re.sub(r'var config = \{.*?\};', replace_config, content, flags=re.DOTALL)

    # Update the task list.
    tasks_str = ',\n      '.join([f'"{task}"' for task in tasks])
    content = re.sub(r'var tasks = \[.*?\];', f'var tasks = [{tasks_str}];', content, flags=re.DOTALL)


    # Build the output path.
    task_names = '-'.join(tasks)
    output_path = os.path.join(os.path.dirname(file_path), new_file_name)

    # Save the generated HTML file.
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Generated HTML file: {new_file_name}")

    # Build the task description.
    description_parts = [describe_dict[task] for task in tasks if task in describe_dict]
    description = ', and then '.join(description_parts)
    first_letter = description[0].upper()
    rest_letter = description[1:]
    description = first_letter + rest_letter
    target_count = description.count('(.*)')
    targets = ', '.join([f"\'target{i + 1}\'" for i in range(target_count)])

    # Update fields.py.
    fields_path = os.path.join('computergym/computergym/miniwob/miniwob_interface', 'fields.py')
    generated_env_name = f"Gen-{task_names}"
    if os.path.exists(fields_path):
        with open(fields_path, 'r', encoding='utf-8') as f:
            fields_content = f.readlines()
        if any(generated_env_name in line for line in fields_content):
            print(f"{generated_env_name} already exists; no changes were made.")
        else:
            # Insert the new environment definition.
            new_entry = f"_add(\n    '{generated_env_name}',\n    r'{description}', [{targets}]\n)\n"
            fields_content.insert(268, new_entry)
            with open(fields_path, 'w', encoding='utf-8') as f:
                f.writelines(fields_content)
            print(f"Updated {fields_path}")


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)

    required_keys = {'controls', 'tasks', 'new_file_name'}
    missing_keys = required_keys - config.keys()
    if missing_keys:
        missing = ', '.join(sorted(missing_keys))
        raise ValueError(f"Missing required config field(s): {missing}")

    if not isinstance(config['controls'], dict):
        raise ValueError("'controls' must be an object")
    if not isinstance(config['tasks'], list) or not all(
            isinstance(task, str) for task in config['tasks']):
        raise ValueError("'tasks' must be a list of strings")
    if not config['tasks']:
        raise ValueError("'tasks' must not be empty")
    if not isinstance(config['new_file_name'], str) or not config['new_file_name']:
        raise ValueError("'new_file_name' must be a non-empty string")
    if os.path.basename(config['new_file_name']) != config['new_file_name']:
        raise ValueError("'new_file_name' must be a file name, not a path")

    expected_file_name = f"Gen-{'-'.join(config['tasks'])}.html"
    if config['new_file_name'] != expected_file_name:
        raise ValueError(
            f"'new_file_name' must be '{expected_file_name}' for the configured tasks"
        )

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Generate a CompWoB+ task from a JSON configuration file.'
    )
    parser.add_argument('config', help='Path to the generator JSON config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # The source template is fixed; task settings come from the config file.
    file_path = 'computergym/computergym/miniwob/miniwob_interface/html/miniwob/ComTask-Generator.html'
    update_html(
        file_path,
        config['controls'],
        config['tasks'],
        config['new_file_name'],
    )


if __name__ == '__main__':
    main()
