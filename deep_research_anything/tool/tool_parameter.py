
class ToolParameter:
    def __init__(
        self, name, code, description, optional=False, value=None, parameter_type=str
    ):
        self.name = name
        self.code = code
        self.description = description
        self.optional = optional
        self.value = value
        self.parameter_type = parameter_type

    def set_value(self, value):
        self.value = value

    def get_prompt(self):
        return f"""- Parameter Name: {self.name}
- Parameter Code: {self.code}
- Parameter Description: {self.description}
- Parameter Type: {self.parameter_type}
- Parameter Optional: {self.optional}
"""

    def is_ready(self):
        if self.optional:
            return True
        else:
            return self.value is not None

    def reset(self):
        self.value = None
