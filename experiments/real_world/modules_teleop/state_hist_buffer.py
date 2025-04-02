import numpy as np

class HistoryBuffer:
    def __init__(self, num_envs: int, history_length: int = 10, state_dim: int = 10):
        """
        Initialize a history buffer as a preallocated NumPy array of shape (num_envs, history_length, state_dim).
        
        Args:
            num_envs (int): Number of environments.
            history_length (int): Maximum number of states to store (default 10).
            state_dim (int): Dimensionality of each state (default 10).
        """
        self.num_envs = num_envs
        self.history_length = history_length
        self.state_dim = state_dim
        
        # Preallocate a buffer: shape (num_envs, history_length, state_dim)
        self.buffer = np.zeros((num_envs, history_length, state_dim), dtype=np.float32)
        # Pointer to next insertion per environment: shape (num_envs,)
        self.ptr = np.zeros(num_envs, dtype=np.int64)
        # Count of inserted items per environment: shape (num_envs,)
        self.count = np.zeros(num_envs, dtype=np.int64)

    def append(self, obs: np.ndarray):
        """
        Append new observations to the buffer for each environment.
        If the buffer is full for an environment, the oldest observation is overwritten.
        
        Args:
            obs (np.ndarray): New observations of shape (num_envs, state_dim).
        """
        env_indices = np.arange(self.num_envs)
        # Insert the new observation at the current pointer location for each environment.
        self.buffer[env_indices, self.ptr] = obs
        # Update pointer in circular fashion.
        self.ptr = (self.ptr + 1) % self.history_length
        # Update count per environment; cap the count at history_length.
        self.count = np.minimum(self.count + 1, self.history_length)

    def is_full(self) -> np.ndarray:
        """
        Check whether the history buffer is full for each environment.
        
        Returns:
            np.ndarray: Boolean array of shape (num_envs,) where True indicates that
                        the corresponding environment's buffer is full.
        """
        return self.count == self.history_length

    def get_history(self) -> np.ndarray:
        """
        Retrieve the history for each environment in time order (oldest to newest).
        This rearranges the circular buffer so that the oldest observation comes first.
        
        Returns:
            np.ndarray: History array of shape (num_envs, history_length, state_dim)
        """
        # The pointer indicates where the next element will be inserted,
        # so it also indicates the position of the oldest element.
        idx = (self.ptr[:, None] + np.arange(self.history_length)) % self.history_length
        # Use advanced indexing to reorder the buffer along the time dimension.
        ordered_history = self.buffer[np.arange(self.num_envs)[:, None], idx]
        return ordered_history

    def clear_envs(self, env_ids):
        """
        Clear the history for the specified environments by resetting their buffers, pointers, and counts.
        
        Args:
            env_ids (list or np.ndarray): Indices of environments to clear.
        """
        if not isinstance(env_ids, np.ndarray):
            env_ids = np.array(env_ids, dtype=np.int64)
        self.buffer[env_ids] = 0
        self.ptr[env_ids] = 0
        self.count[env_ids] = 0

    def get_oldest_obs(self) -> np.ndarray:
        """
        Get a copy of the oldest observation from each environment's buffer without removing it.
        If the buffer is not yet full, the oldest observation is assumed to be at index 0.
        Otherwise, when the buffer is full, the oldest observation is at the index indicated by the pointer.
        
        Returns:
            np.ndarray: Oldest observations for each environment, shape (num_envs, state_dim).
        """
        # For each environment, if the count is less than history_length, then the oldest
        # observation is at index 0. Otherwise, the oldest is at the pointer.
        oldest_idx = np.where(self.count < self.history_length, 0, self.ptr)
        oldest_obs = self.buffer[np.arange(self.num_envs), oldest_idx]
        return oldest_obs

# === Example usage ===
if __name__ == "__main__":
    # Create a history buffer for 3 environments, each with a history of 5 states of dimension 4.
    hb = HistoryBuffer(num_envs=3, history_length=5, state_dim=4)

    # Append some dummy observations.
    for t in range(7):
        obs = np.full((3, 4), t, dtype=np.float32)
        hb.append(obs)
        print(f"After step {t+1}:")
        print("Buffer:")
        print(hb.buffer)
        print("History (ordered):")
        print(hb.get_history())
        print("Oldest observation:")
        print(hb.get_oldest_obs())
        print("-" * 40)
