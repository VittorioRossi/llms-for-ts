from jinja2 import Template
from pathlib import Path

def load_template(template_name: str) -> Template:
    template_path = Path(__file__).resolve().parent / f"{template_name}.j2"
    if not template_path.exists():
        raise FileNotFoundError(f"Template {template_name} not found")

    with open(template_path, 'r') as file:
        return Template(file.read())