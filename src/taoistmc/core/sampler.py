import numpy as np
import scipy.interpolate as interp

class Sampler:
    """
    An optimized CDF sampler with improved performance and cleaner interface.
    """

    def __init__(self, x, y):
        """
        Initialize the sampler with input distribution.
        
        Args:
            x: Input values
            y: Probabilities corresponding to x values
        """
        self.x_vals = np.asarray(x)
        self.y_vals = np.asarray(y)
        self._normalize()
        self._build_cdf()
        
    def _normalize(self):
        """Normalize the probability distribution."""
        self.pdf = self.y_vals / np.sum(self.y_vals)
        
    def _build_cdf(self):
        """Build cumulative distribution function."""
        self.cdf = np.cumsum(self.pdf)
        
    def sample(self, size=1):
        """
        Generate random samples from the distribution.
        
        Args:
            size: Number of samples to generate
            
        Returns:
            Array of samples
        """
        rand_vals = np.random.random(size)
        # Vectorized search for faster performance
        indices = np.searchsorted(self.cdf, rand_vals)
        indices = np.clip(indices, 0, len(self.x_vals)-1)
        return self.x_vals[indices]


class HistogramSampler(Sampler):
    """
    Specialized sampler for histogram data with optional oversampling.
    """
    
    def __init__(self, bin_edges, counts, oversampling_factor=10, spline=False):
        """
        Initialize histogram sampler.
        
        Args:
            bin_edges: Edges of histogram bins
            counts: Counts in each bin
            oversampling_factor: How many points to generate per bin
            spline: Whether to use spline interpolation
        """
        self.spline = spline
        self.os_factor = oversampling_factor
        x, y = self._prepare_histogram(bin_edges, counts)
        super().__init__(x, y)
    
    def _prepare_histogram(self, bin_edges, counts):
        """Prepare oversampled histogram data."""
        dx = bin_edges[1] - bin_edges[0]
        x = np.linspace(
            bin_edges[0], 
            bin_edges[-1], 
            num=len(bin_edges)*self.os_factor - (self.os_factor-1),
            endpoint=True
        )
        y = np.zeros_like(x)
        
        if self.spline:
            # Use cubic spline interpolation
            mid_points = bin_edges[:-1] + dx/2
            mid_points = np.concatenate(([mid_points[0]-3*dx/2], mid_points, [mid_points[-1]+3*dx/2]))
            extended_counts = np.concatenate(([0], counts, [0]))
            
            spline = interp.interp1d(
                mid_points, 
                extended_counts, 
                kind='cubic', 
                fill_value=0, 
                bounds_error=False
            )
            y = spline(x)
            y[y < 0] = 0  # Ensure no negative probabilities
        else:
            # Basic oversampling
            bin_indices = np.digitize(x, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, len(counts)-1)
            y = counts[bin_indices]
            
        return x, y