import os
import sys
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from PIL import Image

# Import from the local SAM repository
# Ensure that the SAM repository is cloned and in the python path
try:
    from datasets.augmentations import AgeTransformer
    from utils.common import tensor2im
    from models.psp import pSp
except ImportError:
    print("WARNING: SAM repository modules not found. Ensure the yuval-alaluf/SAM repository is cloned in your workspace.")

class AgeTransformationModel:
    """
    Handles the initialization and inference of the StyleGAN-based Age Transformation model.
    Requires the SAM model weights (e.g., sam_ffhq_aging.pt) and the dlib face landmarks predictor.
    """
    def __init__(self, model_path: str, landmarks_predictor_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.landmarks_predictor_path = landmarks_predictor_path
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.net = self._load_model()
    
    def _load_model(self):
        """Loads the pre-trained SAM pSp model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
            
        ckpt = torch.load(self.model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = self.model_path
        opts = Namespace(**opts)
        
        net = pSp(opts)
        net.eval()
        net.to(self.device)
        return net

    def align_face(self, image_path: str):
        """Aligns the face in the image using dlib 68-point landmarks."""
        import dlib
        from scripts.align_all_parallel import align_face as align_face_script
        
        if not os.path.exists(self.landmarks_predictor_path):
            raise FileNotFoundError(f"Dlib landmarks predictor not found at {self.landmarks_predictor_path}")
            
        predictor = dlib.shape_predictor(self.landmarks_predictor_path)
        aligned_image = align_face_script(filepath=image_path, predictor=predictor)
        return aligned_image

    def transform_age(self, input_image_path: str, target_age: int, output_dir: str = "outputs") -> str:
        """
        Transforms the age of the person in the input image.
        
        Args:
            input_image_path: Path to the source image.
            target_age: The target age for the transformation.
            output_dir: Directory to save the transformed image.
            
        Returns:
            The path to the generated output image.
        """
        aligned_image = self.align_face(input_image_path)
        input_tensor = self.transform(aligned_image)
        
        age_transformer = AgeTransformer(target_age=target_age)
        
        with torch.no_grad():
            input_image_age = age_transformer(input_tensor.cpu()).to(self.device)
            input_image_age = input_image_age.unsqueeze(0)  # Add batch dimension
            
            result_tensor = self.net(input_image_age.float(), randomize_noise=False, resize=False)[0]
            result_image = tensor2im(result_tensor)
            
        os.makedirs(output_dir, exist_ok=True)
        filename = f"age_{target_age}_{os.path.basename(input_image_path)}"
        output_path = os.path.join(output_dir, filename)
        
        result_image.save(output_path)
        return output_path

if __name__ == "__main__":
    # Example usage (ensure models are downloaded before running)
    model = AgeTransformationModel(
        model_path="./pretrained_models/sam_ffhq_aging.pt",
        landmarks_predictor_path="./shape_predictor_68_face_landmarks.dat"
    )
    result_path = model.transform_age("sample_input.jpg", target_age=50)
    print(f"Age transformation complete. Saved to: {result_path}")
