import torch

class Normalizer:
    def __init__(self, r_mean, g_mean, b_mean, r_std, g_std, b_std):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean
        self.r_std = r_std
        self.g_std = g_std
        self.b_std = b_std

    def __call__(self, batch):
        batch[:, 0] = (batch[:, 0] - self.r_mean) / self.r_std
        batch[:, 1] = (batch[:, 1] - self.g_mean) / self.g_std
        batch[:, 2] = (batch[:, 2] - self.b_mean) / self.b_std

class GaussianIterator:
    def __init__(self, batch_size, num_batches, transformers):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batch_idx = 0
        self.transformers = transformers

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        self.batch_idx += 1
        batch = torch.randn(self.batch_size, 3, 32, 32) + 0.5
        batch = torch.clamp(batch, 0, 1)

        # Run in-place transformers on the batch, such as normalization
        for t in self.transformers:
            t(batch)

        return batch, None

class UniformIterator:
    def __init__(self, batch_size, num_batches, transformers):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batch_idx = 0
        self.transformers = transformers

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        self.batch_idx += 1
        batch = torch.rand(self.batch_size, 3, 32, 32)

        # Run in-place transformers on the batch, such as normalization
        for t in self.transformers:
            t(batch)

        return batch, None

class GeneratingLoader:
    def __init__(self, batch_size, num_batches, transformers):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.transformers = transformers
    
    def __len__(self):
        return self.num_batches

class GaussianLoader(GeneratingLoader):
    def __init__(self, batch_size, num_batches, transformers):
        super(GaussianLoader, self).__init__(batch_size, num_batches, transformers)
    
    def __iter__(self):
        return GaussianIterator(self.batch_size, self.num_batches, self.transformers)

class UniformLoader(GeneratingLoader):
    def __init__(self, batch_size, num_batches, transformers):
        super(UniformLoader, self).__init__(batch_size, num_batches, transformers)
    
    def __iter__(self):
        return UniformIterator(self.batch_size, self.num_batches, self.transformers)
