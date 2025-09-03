import jittor as jt
import jittor.nn as nn

import sys
sys.path.append("./models")
from jimm import seresnext101_32x8d, tf_efficientnet_b3_ns, seresnext50_32x4d


class GeM(nn.Module):
    """Generalized Mean Pooling layer"""
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = jt.ones(1) * p
        self.eps = eps
        self.avg_pool = None  # Lazy initialization

    def execute(self, x):
        bs, ch, h, w = x.shape

        # Initialize pooling layer on first run
        if self.avg_pool is None:
            self.avg_pool = nn.AvgPool2d(kernel_size=(h, w))

        x = jt.clamp(x, min_v=self.eps).pow(self.p)
        x = self.avg_pool(x)
        x = x.pow(1.0 / self.p)
        return x.reshape(bs, ch)


class FeatureExtractor(nn.Module):
    """Abstract base class for feature extractors"""
    def __init__(self, pretrain=True):
        super().__init__()
        self.pretrain = pretrain

    def forward_features(self, x):
        """Return intermediate and final features"""
        raise NotImplementedError("Subclasses must implement forward_features")

    def get_embedding_size(self):
        """Return the size of the final feature embedding"""
        raise NotImplementedError("Subclasses must implement get_embedding_size")


class SEResNeXtExtractor(FeatureExtractor):
    """Feature extractor for SEResNeXt models"""
    def __init__(self, model_name, pretrain=True):
        super().__init__(pretrain)
        self.model_name = model_name

        if self.model_name == "seresnext101_32x8d":
            self.backbone = seresnext101_32x8d(pretrained=pretrain, num_classes=0)
        elif self.model_name == "seresnext50_32x4d":
            self.backbone = seresnext50_32x4d(pretrained=pretrain, num_classes=0)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.embedding_size = self._determine_embedding_size()

    def _determine_embedding_size(self):
        """Determine embedding size based on backbone architecture"""
        if hasattr(self.backbone, 'num_features'):
            print(self.backbone.num_features)
            return self.backbone.num_features
        elif hasattr(self.backbone, 'fc'):
            print(self.backbone.fc.in_features)
            return self.backbone.fc.in_features
        return 2048

    def get_embedding_size(self):
        return self.embedding_size

    def forward_features(self, x):
        """Extract intermediate and final features"""
        # Standard ResNet-like architecture
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        return x3, x4


class CancerModel(nn.Module):
    """Breast Cancer Classification Model with flexible backbone"""

    POOLING_OPTIONS = {
        'gem': GeM,
        'avg': nn.AdaptiveAvgPool2d,
        'max': nn.AdaptiveMaxPool2d
    }

    def __init__(self, model_name, model_name2, num_classes, pretrain=True,
                pooling_type='gem', dropout_rate=0.2):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrain = pretrain
        self.dropout_rate = dropout_rate

        if "seresnext" in model_name:
            self.feature_extractor = SEResNeXtExtractor(model_name, pretrain)
        else:
            raise NotImplementedError(f"Backbone {model_name} not implemented")

        if 'tf_efficientnet_b3_ns' in model_name2:
            self.backbone2 = tf_efficientnet_b3_ns(num_classes=num_classes, pretrained=pretrain)
            backbone2_num_features = 1536

        if pooling_type == 'gem':
            self.pool = GeM(p=3, eps=1e-6)
        else:
            self.pool = self.POOLING_OPTIONS[pooling_type](1)

        emb_size = self.feature_extractor.get_embedding_size() + backbone2_num_features

        # Optional multi-layer head
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, emb_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_size//2, num_classes)
        )

    def execute(self, x):
        """Forward pass with optional intermediate features"""
        x1 = self.backbone2.conv_stem(x)
        x1 = self.backbone2.bn1(x1)
        x1 = self.backbone2.act1(x1)
        x1 = self.backbone2.blocks(x1)
        x1 = self.backbone2.conv_head(x1)
        x1 = self.backbone2.bn2(x1)
        x1 = self.backbone2.act2(x1)

        _, final_features = self.feature_extractor.forward_features(x)
        # print(final_features.shape)

        pooled = self.pool(final_features)
        pooled1 = self.pool(x1)
        # print(pooled1.shape)
        # print(pooled.shape)

        if pooled.ndim > 2:
            pooled = pooled.flatten(1)
            pooled1 = pooled1.flatten(1)
        # print(pooled.shape)

        pooled_combine = jt.cat([pooled, pooled1], dim=1)
        # print(pooled_combine.shape)

        logit = self.classifier(pooled_combine)
        # print(logit.shape)
        return logit

    def get_feature_extractor(self):
        """Get the feature extractor module for feature extraction"""
        return self.feature_extractor



if __name__ == '__main__':
    model = CancerModel(
        model_name="seresnext50_32x4d",
        model_name2="tf_efficientnet_b3_ns",
        num_classes=6,
        pretrain=False,
        pooling_type='gem',
        dropout_rate=0.4
    )
    model.eval()

    batch_size = 4
    height, width = 448, 448
    dummy_input = jt.randn(batch_size, 3, height, width)

    print("="*50)
    print("Input Shape:", dummy_input.shape)
    print("Input Type:", type(dummy_input))
    print("="*50)

    try:
        with jt.no_grad():
            output = model(dummy_input)
        print("\nForward Pass Successful!")
        print("Output Shape:", output.shape)
        print("Output Sample:\n", output[0])
    except Exception as e:
        print("\nForward Pass Failed!")
        print("Error:", str(e))


    try:
        feature_extractor = model.get_feature_extractor()
        x3, x4 = feature_extractor.forward_features(dummy_input)
        print("\nFeature Extraction Successful!")
        print("Intermediate Feature (x3) Shape:", x3.shape)
        print("Final Feature (x4) Shape:", x4.shape)
    except Exception as e:
        print("\nFeature Extraction Failed!")
        print("Error:", str(e))


    try:
        pooled = model.pool(x4)
        print("\nPooling Layer Check:")
        print("Before Pooling:", x4.shape)
        print("After Pooling:", pooled.shape)
        print("Pooled Tensor ndim:", pooled.ndim)
    except Exception as e:
        print("\nPooling Check Failed!")
        print("Error:", str(e))