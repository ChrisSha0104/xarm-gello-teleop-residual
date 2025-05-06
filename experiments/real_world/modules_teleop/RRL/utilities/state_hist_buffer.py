import torch

class HistoryBuffer:
    def __init__(self,
                 num_envs: int,
                 hist_length: int,
                 state_dim: int,
                 num_samples: int,
                 sample_spacing: int,
                 device: str = 'cuda:0'):
        """
        A circular buffer storing the last `hist_length` observations per environment.

        Args:
            num_envs (int): Number of parallel environments.
            hist_length (int): Maximum history length per environment.
            state_dim (int): Dimensionality of each observation.
            num_samples (int): Number of samples to retrieve in get_history.
            sample_spacing (int): Step spacing between samples.
            device (str): Torch device for storage.
        """
        self.num_envs = num_envs
        self.hist_length = hist_length
        self.state_dim = state_dim
        self.num_samples = num_samples
        self.sample_spacing = sample_spacing
        self.device = device

        assert hist_length >= (num_samples - 1) * sample_spacing + 1, (
            f"hist_length must be at least {(num_samples - 1) * sample_spacing + 1} "
            f"to support num_samples={num_samples}, sample_spacing={sample_spacing}")

        # buffer shape: (num_envs, hist_length, state_dim)
        self.buffer = torch.zeros(num_envs, hist_length, state_dim, device=device)
        # pointer for next write per env
        self.ptr = torch.zeros(num_envs, dtype=torch.long, device=device)

    def append(self, obs: torch.Tensor):
        """
        Append a new observation for each environment.

        Args:
            obs (torch.Tensor): Shape (num_envs, state_dim).
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        # write obs to current pointers
        self.buffer[env_ids, self.ptr] = obs
        # advance pointers
        self.ptr = (self.ptr + 1) % self.hist_length

    def get_history(self) -> torch.Tensor:
        """
        Get the most recent `num_samples` observations spaced by `sample_spacing`.
        Returns a flat tensor of shape (num_envs, num_samples * state_dim).
        Order: [obs_t, obs_t-m, obs_t-2m, ...]
        """
        # compute offsets: [0, sample_spacing, 2*sample_spacing, ...]
        offsets = torch.arange(self.num_samples, device=self.device) * self.sample_spacing
        # last-written index per env: ptr - 1 mod hist_length
        last_idx = (self.ptr - 1) % self.hist_length
        # compute indices for each sample: shape (num_envs, num_samples)
        idx = (last_idx.unsqueeze(1) - offsets.unsqueeze(0)) % self.hist_length
        env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        # gather and flatten
        samples = self.buffer[env_ids, idx]  # (num_envs, num_samples, state_dim)
        return samples.reshape(self.num_envs, self.num_samples * self.state_dim)

    def clear_envs(self, env_ids):
        """
        Clear buffer contents for given environments.

        Args:
            env_ids (list or torch.Tensor): Environment indices to clear.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        # zero out buffer and reset pointers
        self.buffer[env_ids] = 0
        self.ptr[env_ids] = 0

    def initialize(self, initial_obs: torch.Tensor, env_ids=None):
        """
        Fill history with a given observation for select environments.

        Args:
            initial_obs (torch.Tensor): Shape (state_dim,) or (N, state_dim).
            env_ids (list or torch.Tensor, optional): Envs to init; defaults to all.
        """
        # determine target envs
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        # broadcast if needed
        if initial_obs.dim() == 1:
            assert initial_obs.shape[0] == self.state_dim, (
                f"Expected shape ({self.state_dim},), got {initial_obs.shape}")
            initial_obs = initial_obs.unsqueeze(0).expand(len(env_ids), -1)
        assert initial_obs.shape == (len(env_ids), self.state_dim), (
            f"Expected shape ({len(env_ids)}, {self.state_dim}), got {initial_obs.shape}")

        # fill buffer and reset pointers
        self.buffer[env_ids] = initial_obs.unsqueeze(1).expand(-1, self.hist_length, -1)
        self.ptr[env_ids] = 0
