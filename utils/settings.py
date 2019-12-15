IMAGE_ROOT = './sample_data/cat2mask/'
IMAGE_LIST_TEXT = './sample_data/cat2mask/label.txt'
BATCH_SIZE = 1
EPOCH_COUNT = 40
INPUT_SIZE = (256, 256)
INPUT_CHANNEL = 3
OUTPUT_CHANNEL = 3
TORCH_DEVICE = 'cpu' # 'cpu' or 'cuda'
LOSS_FUNCTION_NAME = 'torch.nn.MSELoss'
OPTIMIZER_NAME = 'torch.optim.RMSprop'
OPTIMIZER_PARAMS = {'lr': 1e-8, 'weight_decay': 1e-8}
UNET_NAME = 'VanillaUNet'
