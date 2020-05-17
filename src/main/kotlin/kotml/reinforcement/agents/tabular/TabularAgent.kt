package kotml.reinforcement.agents.tabular

import kotml.reinforcement.agents.RLAgent
import kotml.reinforcement.policies.tabular.TabularPolicy

abstract class TabularAgent(
    val behaviorPolicy: TabularPolicy
) : RLAgent<Int, Int>()
