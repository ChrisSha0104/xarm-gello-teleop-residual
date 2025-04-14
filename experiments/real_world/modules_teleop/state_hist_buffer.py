import torch

class HistoryBuffer:
    def __init__(self, num_envs: int, history_length: int = 10, state_dim: int = 10):
        """
        Initialize a history buffer as a preallocated tensor of shape (num_envs, history_length, state_dim).
        
        Args:
            num_envs (int): Number of environments.
            history_length (int): Maximum number of states to store (default 10).
            state_dim (int): Dimensionality of each state (default 10).
        """
        self.num_envs = num_envs
        self.history_length = history_length
        self.state_dim = state_dim

        # Preallocate a buffer: shape (num_envs, history_length, state_dim)
        self.buffer = torch.zeros(num_envs, history_length, state_dim)
        # Pointer to next insertion per environment: shape (num_envs,)
        self.ptr = torch.zeros(num_envs, dtype=torch.long)
        # Count of inserted items per environment: shape (num_envs,)
        self.count = torch.zeros(num_envs, dtype=torch.long)

    def append(self, obs: torch.Tensor):
        """
        Append new observations to the buffer for each environment.
        If the buffer is full for an environment, the oldest observation is overwritten.
        
        Args:
            obs (torch.Tensor): New observations of shape (num_envs, state_dim).
        """
        # Create indices for environments: shape (num_envs,)
        env_indices = torch.arange(self.num_envs)
        # Insert the new observation at the current pointer location for each environment.
        self.buffer[env_indices, self.ptr] = obs
        # Update pointer in circular fashion.
        self.ptr = (self.ptr + 1) % self.history_length
        # Update count per environment; cap the count at history_length.
        self.count = torch.clamp(self.count + 1, max=self.history_length)

    def is_full(self) -> torch.Tensor:
        """
        Check whether the history buffer is full for each environment.
        
        Returns:
            torch.Tensor: Boolean tensor of shape (num_envs,) where True indicates that
                          the corresponding environment's buffer is full.
        """
        return self.count == self.history_length

    def get_history(self) -> torch.Tensor:
        """
        Retrieve the history for each environment in time order (oldest to newest).
        This rearranges the circular buffer so that the oldest observation comes first.
        
        Returns:
            torch.Tensor: History tensor of shape (num_envs, history_length, state_dim)
        """
        # The pointer indicates where the next element will be inserted,
        # so it also indicates the position of the oldest element.
        idx = (self.ptr.unsqueeze(1) + torch.arange(self.history_length).unsqueeze(0)) % self.history_length
        ordered_history = torch.gather(self.buffer, 1, idx.unsqueeze(-1).expand(-1, -1, self.state_dim))
        return ordered_history

    def clear_envs(self, env_ids):
        """
        Clear the history for the specified environments by resetting their buffers, pointers, and counts.
        
        Args:
            env_ids (list or torch.Tensor): Indices of environments to clear.
        """
        # Convert env_ids to a tensor if it isn't already.
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long)
        # Reset the buffer for these environments.
        self.buffer[env_ids] = 0
        # Reset the insertion pointer and count.
        self.ptr[env_ids] = 0
        self.count[env_ids] = 0

    def get_oldest_obs(self) -> torch.Tensor:
        """
        Get a copy of the oldest observation from each environment's buffer without removing it.
        If the buffer is not yet full, the oldest observation is assumed to be at index 0.
        Otherwise, when the buffer is full, the oldest observation is at index indicated by the pointer.
        
        Returns:
            torch.Tensor: Oldest observations for each environment, shape (num_envs, state_dim).
        """
        # For each environment, if the count is less than history_length, then the oldest
        # observation is at index 0. Otherwise, the oldest is at index `ptr`.
        oldest_idx = torch.where(self.count < self.history_length, torch.zeros_like(self.ptr), self.ptr)
        env_indices = torch.arange(self.num_envs)
        oldest_obs = self.buffer[env_indices, oldest_idx]
        return oldest_obs