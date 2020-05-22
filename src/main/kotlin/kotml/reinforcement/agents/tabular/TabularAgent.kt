package kotml.reinforcement.agents.tabular

import kotml.reinforcement.agents.RLAgent
import kotml.reinforcement.policies.discrete.DiscreteBehaviorPolicy

abstract class TabularAgent(
    behaviorPolicy: DiscreteBehaviorPolicy
) : RLAgent<Int, Int>(behaviorPolicy)
