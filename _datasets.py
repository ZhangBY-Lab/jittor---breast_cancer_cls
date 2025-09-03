import os
import numpy as np
from PIL import Image
from jittor.dataset import Dataset
import jittor as jt
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List, Dict
import pandas as pd
import albumentations as A
import cv2


# ============== Label Smoothing ==============
class LabelSmoothingStrategy(ABC):
    """Abstract base class for label smoothing strategies."""
    
    @abstractmethod
    def __call__(self, label: int, num_classes: int) -> np.ndarray:
        """Generate smoothed label vector."""
        pass


class UniformSmoothing(LabelSmoothingStrategy):
    """Uniform label smoothing across all classes."""
    
    def __init__(self, smooth_factor: float = 0.1):
        self.smooth_factor = smooth_factor
    
    def __call__(self, label: int, num_classes: int) -> np.ndarray:
        one_hot = np.zeros(num_classes)
        one_hot[label] = 1
        smooth_val = self.smooth_factor / (num_classes - 1)
        return one_hot * (1.0 - self.smooth_factor) + (1.0 - one_hot) * smooth_val


class ConfidenceBasedSmoothing(LabelSmoothingStrategy):
    """Adaptive smoothing based on model confidence."""
    
    def __init__(self, base_factor: float = 0.1, confidence_threshold: float = 0.9):
        self.base_factor = base_factor
        self.confidence_threshold = confidence_threshold
    
    def __call__(self, label: int, num_classes: int, confidence: float = None) -> np.ndarray:
        # If confidence is not provided, use base smoothing
        if confidence is None:
            return UniformSmoothing(self.base_factor)(label, num_classes)
        
        # Adaptive smoothing based on confidence
        smoothing_factor = self.base_factor * (1 - min(1.0, confidence/self.confidence_threshold))
        return UniformSmoothing(smoothing_factor)(label, num_classes)


class CurriculumSmoothing(LabelSmoothingStrategy):
    """Curriculum learning based smoothing that decreases over epochs."""
    
    def __init__(self, max_smooth: float = 0.15, min_smooth: float = 0.01,
                total_epochs: int = 100):
        self.max_smooth = max_smooth
        self.min_smooth = min_smooth
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch
    
    def __call__(self, label: int, num_classes: int) -> np.ndarray:
        # Linearly decrease smoothing factor
        progress = min(1.0, self.current_epoch / self.total_epochs)
        smooth_factor = self.max_smooth - (self.max_smooth - self.min_smooth) * progress
        return UniformSmoothing(smooth_factor)(label, num_classes)


# ============== Image Transformation Pipeline ==============
class ImageProcessor:
    """Unified image processing pipeline with mode-specific transformations."""
    
    def __init__(self, transform, mode):
        """
        Initialize processor.
        
        Args:
            transform: Transformation functions
            mode: Processing mode (train/val/test)
        """
        self.mode = mode
        
        self.aug_transform = transform
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply transformations based on mode."""
        r, g, b = image, image, image
        image = Image.merge("RGB", (r, g, b))

        if self.mode == "train":
            img_arr = np.array(image)
            img_arr = self.aug_transform["train"](image=img_arr)["image"]
            return img_arr
        elif self.mode == "val":
            img_arr = np.array(image)
            img_arr = self.aug_transform["val"](image=img_arr)["image"]
            return img_arr


# ============== Dataset Class ==============
class ImageFolder(Dataset):
    """Enhanced image dataset with flexible smoothing strategies."""
    
    def __init__(self, root: str, 
                df: Optional[pd.DataFrame] = None, 
                transform: Optional[Union[Tuple, callable]] = None, 
                mode: str = "train", 
                num_classes: int = 6,
                smoothing_strategy: CurriculumSmoothing = None,
                **kwargs):
        """
        Initialize advanced image dataset.
        
        Args:
            root: Root directory of images
            df: DataFrame with filename and class info
            transform: Image transformations
            mode: Dataset mode (train/val/test)
            num_classes: Number of target classes
            smoothing_strategy: Label smoothing strategy
        """
        super().__init__(**kwargs)
        self.root = root
        self.mode = mode
        self.num_classes = num_classes

        if mode == "train":
            if smoothing_strategy is None:
                self.smoothing_strategy = CurriculumSmoothing()
            else:
                self.smoothing_strategy = smoothing_strategy
        else:
            self.smoothing_strategy = None

        if df is not None:
            self.data_dir = df[["filename", "class"]].values.tolist()
        else:
            self.data_dir = [(f, None) for f in sorted(os.listdir(root))]
        self.total_len = len(self.data_dir)
      
        self.image_processor = ImageProcessor(transform, mode) if transform else None
    
    def set_epoch(self, epoch: int):
        """Update epoch for curriculum-based strategies."""
        if hasattr(self.smoothing_strategy, 'set_epoch'):
            self.smoothing_strategy.set_epoch(epoch)

    def __getitem__(self, idx: int) -> Tuple[jt.Var, Union[jt.Var, str]]:
        """Get single data sample with advanced processing."""
        filename, label = self.data_dir[idx]
        image_path = os.path.join(self.root, filename)

        image = Image.open(image_path).convert('L')

        if self.image_processor:
            image = self.image_processor(image)

        image = np.transpose(image, [2,0,1])

        if label is not None:
            if self.smoothing_strategy:
                one_hot = self.smoothing_strategy(label, self.num_classes)
            else:
                one_hot = np.zeros(self.num_classes)
                one_hot[label] = 1
            return jt.array(image), jt.array(one_hot)
        
        return jt.array(image), filename


def build_transform():
    data_transforms = {
        "train": A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            A.RandomCrop(height=448, width=448, p=1.0),

            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1,
                               scale_limit=0.15,
                               rotate_limit=60,
                               p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT,
                                   value=0, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT,
                                 value=0, normalized=True, p=0.5),
            ], p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], p=1.0),
        "val": A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            A.CenterCrop(height=448, width=448, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], p=1.)
    }
    return data_transforms
