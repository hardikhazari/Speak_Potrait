import os
import requests
import torch
from io import BytesIO
from PIL import Image

try:
    from carvekit.ml.files.models_loc import download_all
    from carvekit.web.schemas.config import MLConfig
    from carvekit.web.utils.init_utils import init_interface
except ImportError:
    print("WARNING: CarveKit not found. Please install carvekit-colab.")

class BackgroundReplacementModel:
    """
    Handles background removal and replacement using CarveKit.
    Defaults to Tracer-B7 network which is optimized for general object cutouts.
    """
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_carvekit()

    def _initialize_carvekit(self):
        """Initializes the CarveKit inference pipeline with optimal settings for soft edges."""
        download_all()  # Ensures pre-trained weights are present
        
        config = MLConfig(
            segmentation_network="tracer_b7",
            preprocessing_method="none",
            postprocessing_method="fba", # FBA matting for smooth hair/edge blending
            seg_mask_size=640,
            trimap_dilation=30,
            trimap_erosion=5,
            device=self.device
        )
        self.interface = init_interface(config)

    def download_image(self, url: str) -> Image.Image:
        """Helper method to securely download an image to memory."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    def replace_background(self, foreground_url: str, background_url: str, output_path: str = "result_composite.png") -> str:
        """
        Removes the background from the foreground image and composites it over the background image.
        
        Args:
            foreground_url: URL of the subject image.
            background_url: URL of the desired background image.
            output_path: Path to save the final composite.
            
        Returns:
            The path to the saved composite image.
        """
        # Download images
        foreground_img = self.download_image(foreground_url)
        background_img = self.download_image(background_url).convert("RGBA")

        # CarveKit requires a local file path for inference
        temp_fg_path = "temp_foreground.jpg"
        foreground_img.save(temp_fg_path)

        try:
            # Perform background removal
            result_images = self.interface([temp_fg_path])
            foreground_no_bg = result_images[0].convert("RGBA")

            # Match background size to foreground
            background_img = background_img.resize(foreground_no_bg.size)

            # Composite the images
            final_image = Image.alpha_composite(background_img, foreground_no_bg)
            final_image.save(output_path)
            
            return output_path
        finally:
            if os.path.exists(temp_fg_path):
                os.remove(temp_fg_path)

if __name__ == "__main__":
    # Example usage
    carver = BackgroundReplacementModel()
    
    fg_url = "https://example.com/foreground.jpg"
    bg_url = "https://example.com/background.jpg"
    
    try:
        out_path = carver.replace_background(fg_url, bg_url, "composite_output.png")
        print(f"Background successfully replaced. Saved to {out_path}")
    except Exception as e:
        print(f"Skipped example due to invalid URLs or missing deps: {e}")
