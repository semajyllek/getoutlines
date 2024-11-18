import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from .processor import ObjectOutline

class ResultVisualizer:
    @staticmethod
    def visualize_detection(
        image: np.ndarray,
        binary_mask: np.ndarray,
        outline: ObjectOutline,
        save_path: Path = None
    ):
        plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Binary mask
        plt.subplot(142)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Detection Mask')
        plt.axis('off')
        
        # Outline
        outline_img = np.zeros((
            outline.metadata["original_height"],
            outline.metadata["original_width"],
            3
        ), dtype=np.uint8)
        
        # Draw outline
        vertices = np.array(outline.vertices, dtype=np.int32)
        cv2.polylines(outline_img, [vertices], True, (0, 255, 0), 2)
        
        plt.subplot(143)
        plt.imshow(outline_img)
        plt.title('Extracted Outline')
        plt.axis('off')
        
        # Anchor points
        anchor_img = outline_img.copy()
        for point in outline.anchor_points:
            cv2.circle(
                anchor_img,
                (int(point[0]), int(point[1])),
                5,
                (255, 0, 0),
                -1
            )
            
        plt.subplot(144)
        plt.imshow(anchor_img)
        plt.title('Outline with Anchor Points')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()