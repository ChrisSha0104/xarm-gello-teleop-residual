import torch

class ActionNormalizer:
    def __init__(self, action_low, action_high):
        """
        Args:
            action_low (torch.Tensor or np.ndarray): minimum values for each action dimension
            action_high (torch.Tensor or np.ndarray): maximum values for each action dimension
        """
        self.action_low = action_low
        self.action_high = action_high

    def denormalize(self, action):
        """
        Maps action from [-1, 1] to [action_low, action_high].
        Args:
            action (torch.Tensor): normalized action in [-1, 1]
        Returns:
            torch.Tensor: real-world action
        """
        real_action = 0.5 * (action + 1) * (self.action_high - self.action_low) + self.action_low
        return real_action

    def normalize(self, action):
        """
        Maps action from [action_low, action_high] to [-1, 1].
        Only needed if you need to *normalize* external inputs.
        """
        normalized_action = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
        return normalized_action
