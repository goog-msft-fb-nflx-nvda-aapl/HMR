import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse

# Define ResNet and related classes (unchanged from the original)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define ResNet, SingleInputRegressor, and related functions (unchanged from the original)
class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

def _resnet(arch, block, layers, in_channels, pretrained=False, progress=True, **kwargs):
    model = ResNet(block, layers, in_channels, **kwargs)
    return model

def resnet18(in_channels, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], in_channels, pretrained, progress, **kwargs)

class SingleInputRegressor(nn.Module):
    def __init__(self,
                 resnet_in_channels=1,
                 resnet_layers=18):
        super(SingleInputRegressor, self).__init__()

        if resnet_layers == 18:
            self.image_encoder = resnet18(in_channels=resnet_in_channels,
                                          pretrained=False)

    def forward(self, input):
        return self.image_encoder(input)

def load_straps_resnet_encoder(checkpoint_path, device, in_channels=18):
    """
    Load ResNet-18 encoder from STRAPS model checkpoint.
    
    Args:
        checkpoint_path (str): Path to the STRAPS model checkpoint
        device (torch.device): Device to load the model on
        in_channels (int): Number of input channels
    
    Returns:
        nn.Module: ResNet-18 encoder without the fully connected layer
    """
    full_model = SingleInputRegressor(resnet_in_channels=in_channels, resnet_layers=18)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder_state_dict = {}
    for k, v in checkpoint['best_model_state_dict'].items():
        if k.startswith('image_encoder.'):
            stripped_key = k.replace('image_encoder.', '')
            if not stripped_key.startswith('fc.'):
                encoder_state_dict[stripped_key] = v

    full_model.image_encoder.fc = nn.Identity()
    full_model.image_encoder.load_state_dict(encoder_state_dict, strict=False)
    return full_model.image_encoder

def generate_embeddings(device, proxy_path, checkpoint_path, output_dir, batch_size=16):
    # Load the proxy representations
    print("Loading proxy representations...")
    proxy_representations = np.load(proxy_path)  # Shape: (311, 18, 256, 256)
    
    # Convert to tensor and create DataLoader
    proxy_tensor = torch.tensor(proxy_representations, dtype=torch.float32)
    dataset = TensorDataset(proxy_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained encoder
    print("Loading encoder...")
    encoder = load_straps_resnet_encoder(checkpoint_path, device, in_channels=18)
    encoder.to(device)
    encoder.eval()

    embeddings = []
    print("Generating embeddings...")
    with torch.no_grad():
        for batch_num, (proxy_batch,) in enumerate(tqdm(dataloader)):
            proxy_batch = proxy_batch.to(device)  # Move to GPU/CPU
            embedding_batch = encoder(proxy_batch).cpu().numpy()  # Shape: (batch_size, 512)
            embeddings.append(embedding_batch)

    # Concatenate all batches into a single array
    embeddings = np.vstack(embeddings)  # Shape: (311, 512)

    # Save the embeddings
    output_path = os.path.join(output_dir, "proxy_embeddings.npy")
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 512-dimensional embeddings from proxy representations.")
    parser.add_argument("--proxy_path", type=str, required=True, help="Path to the proxy representations file (.npy).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the encoder checkpoint file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated embeddings.")
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_embeddings(device, args.proxy_path, args.checkpoint_path, args.output_dir)
