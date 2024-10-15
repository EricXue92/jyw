from ignite.metrics import Metric
import torch

class UncertaintyMetric(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(UncertaintyMetric, self).__init__(output_transform=output_transform)
        self._uncertainties = []

    def reset(self):
        """Reset the accumulated uncertainties."""
        self._uncertainties = []

    def update(self, output):
        """Accumulate the uncertainty from each batch."""
        _, _, _, uncertainty = output
        self._uncertainties.append(torch.tensor(uncertainty))

    def compute(self):
        """Return the average uncertainty across all batches."""
        # Stack all uncertainties and compute their mean
        if len(self._uncertainties) == 0:
            raise ValueError("Uncertainty metric must have at least one example before it can be computed.")

        # Compute the mean uncertainty over all batches
        all_uncertainties = torch.concat(self._uncertainties, dim=0)
        return all_uncertainties.mean(dim=1).squeeze().cpu().numpy()

