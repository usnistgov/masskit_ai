import torch
from torch.overrides import TorchFunctionMode

# see https://github.com/pytorch/pytorch/issues/82296
# this code is used to set a default device for pytorch, replicating
# the solution to be used in future versions of pytorch

_DEVICE_CONSTRUCTOR = {
    # standard ones
    torch.empty,
    torch.empty_strided,
    torch.empty_quantized,
    torch.ones,
    torch.arange,
    torch.bartlett_window,
    torch.blackman_window,
    torch.eye,
    torch.fft.fftfreq,
    torch.fft.rfftfreq,
    torch.full,
    torch.fill,
    torch.hamming_window,
    torch.hann_window,
    torch.kaiser_window,
    torch.linspace,
    torch.logspace,
    # torch.nested_tensor,
    # torch.normal,
    torch.ones,
    torch.rand,
    torch.randn,
    torch.randint,
    torch.randperm,
    torch.range,
    torch.sparse_coo_tensor,
    torch.sparse_compressed_tensor,
    torch.sparse_csr_tensor,
    torch.sparse_csc_tensor,
    torch.sparse_bsr_tensor,
    torch.sparse_bsc_tensor,
    torch.tril_indices,
    torch.triu_indices,
    torch.vander,
    torch.zeros,
    torch.asarray,
    # weird ones
    torch.tensor,
    torch.as_tensor,
}

class DeviceMode(TorchFunctionMode):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in _DEVICE_CONSTRUCTOR and torch.utils.data.get_worker_info() is None:
            if kwargs.get('device') is None:
                kwargs['device'] = self.device
            return func(*args, **kwargs)
        return func(*args, **kwargs)


def set_torch_config(torch_device=None) -> None:

    if not('_device' in globals()): 
        if torch_device is None:
            mps_available = False
            try:
                if torch.backends.mps.is_available():
                    mps_available = True
            except:
                pass

            if mps_available:
                device_str = "mps"
            elif torch.cuda.is_available():
                device_str = "cuda"
            else:
                device_str = "cpu"
        else:
            device_str = torch_device

        global _device
        _device = torch.device(device_str)
        # cuda can't be reinitialized in a forked process, so only do the decoration in the main thread
        # and avoid it in the data loaders
        if torch.utils.data.get_worker_info() is None:
            DeviceMode(_device).__enter__()

        # if _device.type == "mps":
        #     DeviceMode.push(_device).__enter__()
        # elif _device.type == "cuda":
        #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
        # else:
        #     torch.set_default_tensor_type(torch.FloatTensor)


set_torch_config()
