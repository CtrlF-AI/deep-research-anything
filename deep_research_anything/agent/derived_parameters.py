from deep_research_anything.agent.parameter import Parameter


class EnumParameter(Parameter):
    def __init__(
        self,
        name,
        code,
        description,
        optional=False,
        value=None,
        parameter_type=str,
        enum_values=None,
    ):
        self.enum_values = enum_values
        super().__init__(
            name=name,
            code=code,
            description=description,
            optional=optional,
            value=value,
            parameter_type=parameter_type,
        )

    def get_enum_values(self):
        return self.enum_values

    def check_value(self, value):
        if value not in self.enum_values:
            raise ValueError(
                f"Invalid value: {value}. Must be one of: {self.enum_values}"
            )
        return value

    def get_prompt(self):
        return (
            super().get_prompt()
            + f"- Please select one of the following options: {self.enum_values}"
        )
