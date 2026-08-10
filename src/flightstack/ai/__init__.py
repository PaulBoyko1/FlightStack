"""FlightStack's optional learned-racing interfaces and reference environment."""

from flightstack.ai.actions import ACTION_SCHEMA_VERSION, action_to_command, normalized_action
from flightstack.ai.config import (
    EnvironmentConfig,
    ObservationConfig,
    RacingAIConfig,
    RewardConfig,
    load_racing_ai_config,
)
from flightstack.ai.environment import FlightStackRaceEnv, make_gymnasium_env
from flightstack.ai.errors import (
    OptionalTrainingDependencyError,
    PolicyNotTrainedError,
    PolicySchemaError,
)
from flightstack.ai.observation import (
    OBSERVATION_COMPONENTS,
    OBSERVATION_DIMENSION,
    OBSERVATION_SCHEMA_VERSION,
    build_observation,
)
from flightstack.ai.policy import (
    LearnedPolicyPilot,
    PolicyMetadata,
    load_policy_metadata,
)
from flightstack.ai.reward import RewardTerms, race_reward
from flightstack.ai.vector import ReferenceVectorEnv

__all__ = [
    "ACTION_SCHEMA_VERSION",
    "EnvironmentConfig",
    "FlightStackRaceEnv",
    "LearnedPolicyPilot",
    "OBSERVATION_COMPONENTS",
    "OBSERVATION_DIMENSION",
    "OBSERVATION_SCHEMA_VERSION",
    "ObservationConfig",
    "OptionalTrainingDependencyError",
    "PolicyMetadata",
    "PolicyNotTrainedError",
    "PolicySchemaError",
    "RacingAIConfig",
    "ReferenceVectorEnv",
    "RewardConfig",
    "RewardTerms",
    "action_to_command",
    "build_observation",
    "load_policy_metadata",
    "load_racing_ai_config",
    "make_gymnasium_env",
    "normalized_action",
    "race_reward",
]
