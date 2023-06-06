import gc
import numpy as np
from skimage.util.shape import view_as_windows
from skimage.measure import block_reduce

from pca import myPCA

class Saab:
    def __init__(self, kernel_size, stride, pooling_size=None, pooling_type=np.max):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling_size = pooling_size
        self.pooling_type = pooling_type
        self.dataset_mean = None
        self.dataset_std = None
        self.features_mean = None
        self.kernels = None
        self.energies = None
        self.pca = None

    # Convert responses to patches representation
    def window_process(self, samples, kernel_size, stride):
        '''
        Create patches
        :param samples: [num_samples, feature_height, feature_width, feature_channel]
        :param kernel_size: int i.e. patch size
        :param stride: int
        :return patches: flattened, [num_samples, output_h, output_w, feature_channel*kernel_size^2]
        '''
        n, h, w, c = samples.shape
        output_h = (h - kernel_size)//stride + 1
        output_w = (w - kernel_size)//stride + 1
        patches = view_as_windows(samples, (1, kernel_size, kernel_size, c), step=(1, stride, stride, c))
        patches = patches.reshape(n, output_h, output_w, c*kernel_size*kernel_size)
        return patches
    
    def remove_mean(self, features, axis):
        '''
        Remove the dataset mean.
        :param features [num_samples,...]
        :param axis the axis to compute mean
        
        '''
        feature_mean = np.mean(features, axis=axis, keepdims=True)
        feature_remove_mean = features-feature_mean
        return feature_remove_mean,feature_mean

    def pooling(self, X):
        return block_reduce(X, (1, self.pooling_size, self.pooling_size, 1), self.pooling_type)

    def calc_mean_std(self, X):
        self.dataset_mean = X.mean(axis=(0, 1, 2))
        self.dataset_std = np.std(X, axis=(0 , 1, 2))

    def fit(self, X):
        X = X.astype("float32")
        
        # Normalize
        self.calc_mean_std(X)
        X -= self.dataset_mean
        X /= self.dataset_std

        # Create patches
        sample_patches = self.window_process(X, self.kernel_size, self.stride)
        h = sample_patches.shape[1]
        w = sample_patches.shape[2]
        sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])
        
        if self.pooling_size is not None:
            if h % self.pooling_size != 0 or w % self.pooling_size != 0:
                print("[Saab] Warning: pooling operation can not equally divide the features!")
        
        # Remove feature mean
        sample_patches, self.features_mean = self.remove_mean(sample_patches, axis=0)

        # Remove patch mean
        ac, dc = self.remove_mean(sample_patches, axis=1)

        # Compute PCA kernel
        self.pca = myPCA()
        self.pca.fit(ac)
        kernels = self.pca.eigenvectors
        eigenvalues = self.pca.eigenvalues

        # Add DC kernel
        num_channels = sample_patches.shape[-1]
        largest_ev = [np.var(dc*np.sqrt(num_channels), ddof=1)]
        dc_kernel = 1/np.sqrt(num_channels)*np.ones((1, num_channels))
        self.kernels = np.concatenate((dc_kernel, kernels[:-1]), axis=0)
        eigenvalues = np.concatenate((largest_ev, eigenvalues[:-1]), axis=0)
        self.energies = eigenvalues/np.sum(eigenvalues)

        return self

    def _transform_batch(self, X, n_channels=-1, channel=-1, channels=[]):
        X = X.astype("float32")

        # Normalize
        X -= self.dataset_mean
        X /= self.dataset_std

        num_samples = X.shape[0]
        
        # Create patches
        sample_patches = self.window_process(X, self.kernel_size, self.stride)

        del X
        gc.collect()

        h = sample_patches.shape[1]
        w = sample_patches.shape[2]
        sample_patches = sample_patches.reshape([-1, sample_patches.shape[-1]])
        sample_patches -= self.features_mean

        if len(channels) != 0:
            transformed = np.matmul(sample_patches, np.transpose(self.kernels[channels]))
        elif channel != -1:
            transformed = np.matmul(sample_patches, np.transpose(self.kernels[channel]))
        else:
            transformed = np.matmul(sample_patches, np.transpose(self.kernels[:n_channels]))
        transformed = transformed.reshape(num_samples, h, w, -1)
        if self.pooling_size != None:
            transformed = self.pooling(transformed)
        return transformed

    def transform(self, X, n_channels=-1, channel=-1, channels=[], batch_size=100000):
        print("[Saab] Input shape", X.shape)
        X = X.astype("float32")
        num_samples, h, w, c = X.shape
        output_h = (h - self.kernel_size)//self.stride + 1
        output_w = (w - self.kernel_size)//self.stride + 1

        if self.pooling_size != None:
            if output_h % self.pooling_size != 0 or output_w % self.pooling_size != 0:
                print("[Saab] Warning: pooling operation can not equally divide the features!")
        
        if len(channels) != 0:
            n_channels = len(channels)
        elif channel != -1:
            n_channels = 1
        elif n_channels == -1:
            n_channels = len(self.energies)

        if num_samples < batch_size:
            output = self._transform_batch(X, n_channels, channel, channels)
            if self.pooling_size != None:
                output = self.pooling(output)
            print("[Saab] Output shape", output.shape)
            return output
        else:
            if self.pooling_size != None:
                output_h = int(output_h/self.pooling_size)
                output_w = int(output_w/self.pooling_size)
                if output_h % self.pooling_size != 0:
                    output_h += 1
                if output_w % self.pooling_size != 0:
                    output_w += 1
            output = np.zeros((num_samples, output_h, output_w, n_channels), dtype="float32")
            for i in range(num_samples//batch_size):
                print("[Saab] Batch", i, "from", i*batch_size , "to", (i+1)*batch_size)
                out = self._transform_batch(X[i*batch_size:(i+1)*batch_size], n_channels, channel, channels)
                output[i*batch_size:(i+1)*batch_size] = out
            if num_samples % batch_size != 0:
                print("[Saab] Batch", num_samples//batch_size, "from", (num_samples//batch_size)*batch_size , "to", num_samples)
                out = self._transform_batch(X[(num_samples//batch_size)*batch_size:num_samples], n_channels, channel, channels)
                output[(num_samples//batch_size)*batch_size:num_samples] = out
            print("[Saab] Output shape", output.shape)
            return output

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_sample_image

    china = load_sample_image('china.jpg')
    print(china.shape)
    china = china[:400,:,:]
    china = view_as_windows(china, (16, 16, 3), step=(16, 16, 3)).reshape(1000, 16, 16, 3)
    china = np.repeat(china, 1000, axis=0)
    print(china.shape)

    saab = Saab(kernel_size=2, stride=2)
    saab.fit(china)
    transformed = saab.transform(china, channel=2)
    print(transformed.shape)
    print(saab.energies)
    plt.plot(saab.energies)
    plt.show()

    plt.imshow(china[500])
    plt.show()
    plt.imshow(transformed[500,:,:,0], cmap="gray")
    plt.show()