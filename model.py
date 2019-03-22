# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
FACE_FEATURES = 1024


class DDAH(nn.Module):
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('ConvTranspose2d') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)

    def __init__(self, hash_bits, CLASS_NUM):
        super(DAH, self).__init__()

        self.spatial_features_1 = nn.Sequential(
            nn.Conv2d(3, 32, 2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.spatial_features_2 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.spatial_features_3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True))

        self.upscales_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(True)
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.upscales_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(True)
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.upscales_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(True)
        )
        self.bn3 = nn.BatchNorm2d(16)

        self.upscales_4 = nn.Sequential(
            nn.Conv2d(16, 1, 1)
        )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.global_avgpooling_3 = nn.Sequential(
            nn.Linear(128*3*3, 512),
            nn.ReLU(True),
            nn.Linear(512, 128)
        )

        self.global_avgpooling_4 = nn.Sequential(
            nn.Linear(256*2*2, 512),
            nn.ReLU(True),
            nn.Linear(512, 256)
        )

        self.face_features_layer = nn.Sequential(
            nn.Linear(2176, FACE_FEATURES),
            nn.BatchNorm1d(FACE_FEATURES),
            nn.ReLU(True)
        )

        self.hash_layer = nn.Sequential(
            nn.Linear(FACE_FEATURES, hash_bits),
            nn.BatchNorm1d(hash_bits)
        )

        self.classifier = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hash_bits, CLASS_NUM),
            nn.LogSoftmax(dim=1)
        )
        self.apply(self.weights_init)

    def forward(self, x):
        attention_mask_16 = self.spatial_features_1(x)
        attention_mask_8 = self.spatial_features_2(attention_mask_16)
        attention_mask_4 = self.spatial_features_3(attention_mask_8)
        attention_mask = self.fc(attention_mask_4)
        attention_mask = self.upscales_1(attention_mask)
        attention_mask = self.bn1(attention_mask + attention_mask_8)
        attention_mask = self.upscales_2(attention_mask)
        attention_mask = self.bn2(attention_mask + attention_mask_16)
        attention_mask = self.upscales_3(attention_mask)
        attention_mask = self.upscales_4(attention_mask)

        # spatial normalization
        spatial_attention_min, _ = torch.min(attention_mask.view(x.size(0), -1), dim=1)
        spatial_attention_max, _ = torch.max(attention_mask.view(x.size(0), -1), dim=1)

        spatial_attention_min = spatial_attention_min.reshape(x.size(0), 1, 1, 1)
        spatial_attention_max = spatial_attention_max.reshape(x.size(0), 1, 1, 1)
        attention_mask = (attention_mask - spatial_attention_min)/(spatial_attention_max - spatial_attention_min)

        x_with_spatial_attention = x * attention_mask
        
        features_3 = self.features(x_with_spatial_attention)
        features_4 = self.conv4(features_3)

        channel_attention_3 = self.global_avgpooling_3(features_3.view(features_3.size(0), -1))
        channel_attention_4 = self.global_avgpooling_4(features_4.view(features_4.size(0), -1))

        # channel normalization
        channel_attention_3_min, _ = torch.min(channel_attention_3, dim=1)
        channel_attention_3_max, _ = torch.max(channel_attention_3, dim=1)
        channel_attention_3_min = channel_attention_3_min.unsqueeze(1)
        channel_attention_3_max = channel_attention_3_max.unsqueeze(1)
        channel_attention_3 = (channel_attention_3 - channel_attention_3_min)/(channel_attention_3_max - channel_attention_3_min)
        channel_attention_4_min, _ = torch.min(channel_attention_4, dim=1)
        channel_attention_4_max, _ = torch.max(channel_attention_4, dim=1)
        channel_attention_4_min = channel_attention_4_min.unsqueeze(1)
        channel_attention_4_max = channel_attention_4_max.unsqueeze(1)
        channel_attention_4 = (channel_attention_4 - channel_attention_4_min)/(channel_attention_4_max - channel_attention_4_min)
        features_3_with_attention = features_3 * channel_attention_3.reshape(channel_attention_3.size(0), -1, 1, 1)
        features_4_with_attention = features_4 * channel_attention_4.reshape(channel_attention_4.size(0), -1, 1, 1)
       
        features_a = torch.cat([features_3_with_attention.view(features_3_with_attention.size(0), -1), features_4_with_attention.view(features_4_with_attention.size(0), -1)], -1)
        features_a = self.face_features_layer(features_a)
        hash_a = self.hash_layer(features_a)
        cls_a = self.classifier(hash_a)

        features_3_without_attention = self.features(x)
        features_4_without_attention = self.conv4(features_3_without_attention)

        channel_attention_3 = self.global_avgpooling_3(features_3_without_attention.view(features_3_without_attention.size(0), -1))
        channel_attention_4 = self.global_avgpooling_4(features_4_without_attention.view(features_4_without_attention.size(0), -1))
        
        # channel normalization
        channel_attention_3_min, _ = torch.min(channel_attention_3, dim=1)
        channel_attention_3_max, _ = torch.max(channel_attention_3, dim=1)
        channel_attention_3_min = channel_attention_3_min.unsqueeze(1)
        channel_attention_3_max = channel_attention_3_max.unsqueeze(1)
        channel_attention_3 = (channel_attention_3 - channel_attention_3_min) / (
                    channel_attention_3_max - channel_attention_3_min)
        channel_attention_4_min, _ = torch.min(channel_attention_4, dim=1)
        channel_attention_4_max, _ = torch.max(channel_attention_4, dim=1)
        channel_attention_4_min = channel_attention_4_min.unsqueeze(1)
        channel_attention_4_max = channel_attention_4_max.unsqueeze(1)
        channel_attention_4 = (channel_attention_4 - channel_attention_4_min) / (
                    channel_attention_4_max - channel_attention_4_min)

        features_3_with_attention = features_3_without_attention * channel_attention_3.reshape(channel_attention_3.size(0), -1, 1, 1)
        features_4_with_attention = features_4_without_attention * channel_attention_4.reshape(channel_attention_4.size(0), -1, 1, 1)

        features = torch.cat([features_3_with_attention.view(features_3_with_attention.size(0), -1),
                              features_4_with_attention.view(features_4_with_attention.size(0), -1)], -1)
        features = self.face_features_layer(features)
        hash_b = self.hash_layer(features)
        cls_b = self.classifier(hash_b)

        return hash_a, cls_a, hash_b, cls_b


if __name__ == '__main__':
    fake_data = torch.randn(2, 3, 32, 32)
    net = DDAH(43,9)
    print(net)
    net(fake_data)
