import typer
from pathlib import Path
from typing import Optional
import json
from .core.detector import SAMDetector
from .core.processor import OutlineProcessor
from .core.visualizer import ResultVisualizer
from .utils.image_utils import load_image, save_json

app = typer.Typer()

@app.command()
def extract(
    image_path: str,
    output_dir: str,
    model_path: str = "sam_vit_h_4b8939.pth",
    visualize: bool = True,
    target_width: int = 600,
    target_height: int = 400
):
    """Extract object outline from an image."""
    # Initialize components
    detector = SAMDetector(model_path=model_path)
    detector.initialize()
    
    processor = OutlineProcessor(
        target_width=target_width,
        target_height=target_height
    )
    
    # Process image
    image = load_image(image_path)
    binary_mask, _ = detector.detect(image)
    outline = processor.extract_outline(binary_mask)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(image_path).stem
    json_path = output_path / f"{base_name}_outline.json"
    
    save_json(outline, json_path)
    
    if visualize:
        viz_path = output_path / f"{base_name}_visualization.png"
        ResultVisualizer.visualize_detection(
            image,
            binary_mask,
            outline,
            viz_path
        )
    
    typer.echo(f"Results saved to {output_dir}")

if __name__ == "__main__":
    app()