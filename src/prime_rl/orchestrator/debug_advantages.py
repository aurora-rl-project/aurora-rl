from prime_rl.orchestrator.advantage import AdvantageInputs, AdvantageOutputs


def constant_advantage(inputs: AdvantageInputs, value: float = 1.0) -> AdvantageOutputs:
    return AdvantageOutputs(advantages=[value for _ in inputs.rollouts])
