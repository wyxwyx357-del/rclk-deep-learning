"""Report whether this Python environment can actually use the NVIDIA GPU."""

import json
import torch


def main():
    cuda = torch.cuda.is_available()
    print(json.dumps({
        'torch': torch.__version__,
        'torch_cuda_build': torch.version.cuda,
        'cuda_available': cuda,
        'device': torch.cuda.get_device_name(0) if cuda else None,
        'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory if cuda else None,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

