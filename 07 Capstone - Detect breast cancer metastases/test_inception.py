from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import sys
import csv
# ----------
import cnn
import utils

# --- get model name ---
training_name = sys.argv[1]

# --- define transformations ---
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(cnn.INCEPTION_INPUT_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(cnn.INCEPTION_MEAN, cnn.INCEPTION_STD)
    ]),
}

# --- load test datasets ---
test_x, test_y = utils.get_matrixes(phase='test',
                                    transform=data_transforms['test'])
test_data = utils.PCamDataset(test_x, test_y)
test_loader = DataLoader(test_data, batch_size=200, shuffle=False,
                         num_workers=3)
test_data_size = len(test_data)

# two GPUs available - one model will run per GPU
if training_name == 'inception_1':
    device = torch.device('cuda:0')
else:
    device = torch.device('cuda:1')

# --- initialize inception network and load weights from saved model ----
inception = cnn.initialize_model(cnn.OUTPUT_SIZE, device)
inception.load_state_dict(torch.load('{}.pth'.format(training_name)))
inception.eval()

# --- run on test dataset ---
predictions = []
with torch.no_grad():
    for x, y in tqdm(test_loader, desc='test', unit='batches'):
        x = x.to(device)
        y = y.long().reshape(1, -1).squeeze().to(device)
        outputs = inception(x)
        values, _ = torch.max(outputs.data, 1)
        predictions.extend(list(zip(outputs.data[:, 1].cpu().numpy(),
                                    y.data.cpu().numpy())))

# --- write results ---
with open('test_{}.csv'.format(training_name), mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['prediction', 'label'])
    for pred, label in predictions:
        writer.writerow([pred, label])
