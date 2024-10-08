import numpy as np

class ExperienceBuffer:
    def __init__(self, states, actions, rewards,advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)
        h5file['experience'].create_dataset('advantages', data=self.advantages)

class ExperienceCollector:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_estimated_values = []
    
    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []
    
    def record_decision(self, state, action, estimated_value=0):
        self.current_episode_states.append(state)
        self.current_episode_actions.append(action)
        self.current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward):
        num_states = len(self.current_episode_states)
        self.states += self.current_episode_states
        self.actions += self.current_episode_actions
        self.rewards += [reward for _ in range(num_states)]

        for i in range(num_states):
            advantage = reward - self.current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_estimated_values = []
    
    def to_buffer(self):
        return ExperienceBuffer(states=np.array(self.states),
                                actions=np.array(self.actions),
                                rewards=np.array(self.rewards))

def load_experience(h5file):
    return ExperienceBuffer(
            states=np.array(h5file['experience']['states']),
            actions=np.array(h5file['experience']['action']),
            rewards=np.array(h5file['experience']['rewards'])
            )

def combine_experience(collectors) -> ExperienceBuffer:
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    combined_advantages = np.concatenate([np.array(c.advantages) for c in collectors])

    return ExperienceBuffer(combined_states,
                            combined_actions,
                            combined_rewards,
                            combined_advantages)


class ZeroExperienceBuffer:
    def __init__(self, states, visit_counts, rewards):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('visit_counts', data=self.visit_counts)
        h5file['experience'].create_dataset('rewards', data=self.rewards)


class ZeroExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self.current_episode_states = []
        self.current_episode_visit_counts = []

    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_visit_counts = []
    
    def record_decision(self, state, visit_counts):
        self.current_episode_states.append(state)
        self.current_episode_visit_counts.append(visit_counts)
    
    def complete_episode(self, reward):
        num_states = len(self.current_episode_states)
        self.states += self.current_episode_states
        self.visit_counts += self.current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]

        self.begin_episode()


def combine_zero_experience(collectors: list[ZeroExperienceCollector]) -> ZeroExperienceBuffer:
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    return ZeroExperienceBuffer(combined_states,
                                combined_visit_counts,
                                combined_rewards)

def load_zero_experience(h5file):
    return ZeroExperienceBuffer(
            states=np.array(h5file['experience']['states']),
            visit_counts=np.array(h5file['experience']['action']),
            rewards=np.array(h5file['experience']['rewards'])
            )