from typing import Any
import torch
import torch.jit
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np

class TiledPrediction:
    def __init__(self, tile_size, model_path, overlap, device, input_channels, output_channels):
        self.tile_size = tile_size
        self.model_path = model_path
        self.overlap = overlap
        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model = torch.jit.load(model_path).to(device)
        self.model.eval()
        self.blend_weights = self.make_blend_mask()

    def make_blend_mask(self):
        blend_weights = torch.ones((self.output_channels,) + self.tile_size)
        for i in range(self.overlap[0]):
            blend_weights[:, :, i] *= (i / self.overlap[0])
            blend_weights[:, :, -(i+1)] *= (i / self.overlap[0])
        for i in range(self.overlap[1]):
            blend_weights[:, i, :] *= (i / self.overlap[1])
            blend_weights[:, -(i+1), :] *= (i / self.overlap[1])
        return blend_weights.to(self.device)

    def sliding_window(self, image, stepSize):
        for y in range(0, image.shape[2], stepSize[1]):
            for x in range(0, image.shape[3], stepSize[0]):
                yield (x, y, image[:,:, y:y + self.tile_size[1], x:x + self.tile_size[0]])

    def apply_model(self, tile):
        with torch.no_grad():
            result = self.model(tile)
        return result

    def tile_and_apply(self, img : np.ndarray):
        img = ToTensor()(img)
        assert img.shape[0] == self.input_channels, "Input image does not match model input channels"
        img = img.unsqueeze(0).to(self.device)
        
        step_size = [ts - o for ts, o in zip(self.tile_size, self.overlap)]
        output_img = torch.zeros((self.output_channels, img.shape[2], img.shape[3])).to(self.device)
        blend_map = torch.zeros((self.output_channels, img.shape[2], img.shape[3])).to(self.device)

        for (x, y, window) in self.sliding_window(img, step_size):
            window_w = window.shape[3]
            window_h = window.shape[2]

            if window_w < self.tile_size[0] or window_h < self.tile_size[1]:
                window = torch.nn.functional.pad(window, (0, self.tile_size[0] - window_w, 0, self.tile_size[1] - window_h), "constant", 0)

            result = self.apply_model(window)
            
            output_img[:, y:y+window_h, x:x+window_w] += self.blend_weights[:,:window_h,:window_w] * result.squeeze().cpu()[:, :window_h, :window_w]
            blend_map[:, y:y+window_h, x:x+window_w] += self.blend_weights[:,:window_h, :window_w]

        # Blending the overlapping parts
        output_img = output_img / blend_map
        return output_img.cpu().numpy()

    def __call__(self, img : np.ndarray) -> Any:
        return self.tile_and_apply(img)
