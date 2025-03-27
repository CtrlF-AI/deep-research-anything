class AgentRegistry:
    _agents = {}

    @classmethod
    def register(cls, agent_instance):
        cls._agents[agent_instance.code] = agent_instance

    @classmethod
    def get_agent(cls, code):
        return cls._agents.get(code)

    @classmethod
    def get_all_agents(cls):
        return list(cls._agents.values())
