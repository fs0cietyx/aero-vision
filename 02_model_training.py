"""
Phase 2: Model Training
Implements a 9-channel modified ResNet34-UNet for cloud removal.
Incorporates temporal synthesis padding and L1+Edge Loss constraint.
"""
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, ResNet34_Weights

class ISROCloudDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as hf:
            self.cloudy_data = torch.tensor(hf['cloudy'][:], dtype=torch.float32)
            self.clear_data = torch.tensor(hf['clear'][:], dtype=torch.float32)
        self.length = self.cloudy_data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cloudy = self.cloudy_data[idx]
        clear = self.clear_data[idx]

        # Standard Top-Of-Atmosphere reflectance scaling
        cloudy = cloudy / 10000.0
        clear = clear / 10000.0

        mask = torch.zeros((1, 256, 256), dtype=torch.float32)
        temporal = cloudy.clone()
        return torch.cat([cloudy, mask, temporal], dim=0), clear

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class AdvancedCloudUNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(9, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=False, padding_mode='reflect')
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = old_conv.weight.clone()
            nn.init.kaiming_normal_(self.conv1.weight[:, 3:, :, :], mode="fan_out", nonlinearity="relu")

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64,  64)
        self.dec1 = DecoderBlock(64,  64,  64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        return self.final_up(d1)

def compute_edge_loss(pred, target):
    pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    return torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedCloudUNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    l1_loss_fn = nn.L1Loss()

    h5_file = "isro_training_data.h5"
    dataset = ISROCloudDataset(h5_file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    epochs = 25
    print("Initiating production training sequence...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            pixel_loss = l1_loss_fn(outputs, targets)
            edge_loss = compute_edge_loss(outputs, targets)
            loss = pixel_loss + 0.1 * edge_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Combined Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"isro_model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
