import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=[256, 512, 1024], **kwargs):
        super().__init__()
        self.enc_conv1 = nn.Sequential(
            self.conv_block(in_channels, hidden_channels[0], kernel_size=3, stride=1, padding=1),
            self.conv_block(hidden_channels[0], hidden_channels[0], kernel_size=3, stride=1, padding=1),
        )

        self.enc_conv2 = nn.Sequential(
            self.conv_block(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=1, padding=1),
            self.conv_block(hidden_channels[1], hidden_channels[1], kernel_size=3, stride=1, padding=1),
        )

        self.enc_conv3 = nn.Sequential(
            self.conv_block(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=1, padding=1),
            self.conv_block(hidden_channels[2], hidden_channels[2], kernel_size=3, stride=1, padding=1),
        )
        
        self.bottleneck = nn.Sequential(
            self.conv_block(hidden_channels[2], hidden_channels[2], kernel_size=3, stride=1, padding=1),
        )

        self.dropout = nn.Dropout2d(p=kwargs.pop('dropout', 0.2))
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.upconv3 = nn.ConvTranspose2d(hidden_channels[2], hidden_channels[2], kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            self.conv_block(2*hidden_channels[2], hidden_channels[2], kernel_size=3, stride=1, padding=1),
            self.conv_block(hidden_channels[2], hidden_channels[1], kernel_size=3, stride=1, padding=1),
        )

        self.upconv2 = nn.ConvTranspose2d(hidden_channels[1], hidden_channels[1], kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            self.conv_block(2*hidden_channels[1], hidden_channels[1], kernel_size=3, stride=1, padding=1),
            self.conv_block(hidden_channels[1], hidden_channels[0], kernel_size=3, stride=1, padding=1),
        )
        
        self.upconv1 = nn.ConvTranspose2d(hidden_channels[0], hidden_channels[0], kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            self.conv_block(2*hidden_channels[0], hidden_channels[0], kernel_size=3, stride=1, padding=1),
            self.conv_block(hidden_channels[0], hidden_channels[0], kernel_size=3, stride=1, padding=1),
        )

        self.out_conv = nn.Conv2d(hidden_channels[0], out_channels, kernel_size=1, stride=1)

        # Apply He initialization
        self._initialize_weights()

    def forward(self, x):
        # downsampling (i.e., encoding) part
        enc_conv1 = self.enc_conv1(x)
        enc_conv2 = self.enc_conv2(self.pool(enc_conv1))
        enc_conv3 = self.enc_conv3(self.pool(enc_conv2))

        # bottleneck (i.e., middle) part
        bottleneck = self.bottleneck(self.dropout(self.pool(enc_conv3)))  # Let's try adding Dropout (to avoid overfitting)

        # upsampling (i.e., decoding) part
        upconv3 = self.upconv3(bottleneck)
        dec_conv3 = self.dec_conv3(torch.cat([upconv3, enc_conv3], 1))
        upconv2 = self.upconv2(dec_conv3)
        dec_conv2 = self.dec_conv2(torch.cat([upconv2, enc_conv2], 1))
        upconv1 = self.upconv1(dec_conv2)
        dec_conv1 = self.dec_conv1(torch.cat([upconv1, enc_conv1], 1))

        # output part
        return self.out_conv(dec_conv1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),   # BatchNorm after ReLU
        )

    def _initialize_weights(self):
        """
        Initialize the weights of convolutional layers using He initialization (Kaiming initialization).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
