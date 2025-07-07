import os
import torch
import torchaudio

try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
except ImportError:
    print("WARNING: Zonos repository not found. Ensure Zyphra/Zonos is cloned and in your PYTHONPATH.")

class ZonosTTSModel:
    """
    Text-to-Speech generation using the Zonos transformer model.
    Requires a reference audio file to extract the speaker embedding.
    """
    def __init__(self, reference_audio_path: str, model_id: str = "Zyphra/Zonos-v0.1-transformer"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.reference_audio_path = reference_audio_path
        
        # Reproducibility
        torch.manual_seed(421)
        
        self.model = self._load_model()
        self.speaker_embedding = self._extract_speaker_embedding()

    def _load_model(self):
        """Loads the Zonos model to the appropriate device."""
        print(f"Loading Zonos model '{self.model_id}' to {self.device}...")
        return Zonos.from_pretrained(self.model_id, device=self.device)

    def _extract_speaker_embedding(self):
        """Extracts the speaker embedding from the provided reference audio."""
        if not os.path.exists(self.reference_audio_path):
            raise FileNotFoundError(f"Reference audio not found at {self.reference_audio_path}")
            
        print("Extracting speaker embedding...")
        wav, sampling_rate = torchaudio.load(self.reference_audio_path)
        return self.model.make_speaker_embedding(wav, sampling_rate)

    def generate_speech(self, text: str, language: str = "en-us", speed: float = 1.0, 
                        pitch: float = 20.0, output_path: str = "tts_output.wav") -> str:
        """
        Generates speech audio from text using the extracted speaker conditioning.
        
        Args:
            text: The input text to synthesize.
            language: Language code (e.g., 'en-us', 'hi').
            speed: Speech speed modifier.
            pitch: Pitch modifier.
            output_path: Path to save the synthesized .wav file.
            
        Returns:
            The path to the generated output audio file.
        """
        # Create the conditioning dictionary
        # Note: Emotion vector can also be passed if supported by the conditioning pipeline
        cond_dict = make_cond_dict(
            text=text,
            speaker=self.speaker_embedding,
            language=language
            # Note: speed and pitch might require specific handling depending on the Zonos version.
        )
        conditioning = self.model.prepare_conditioning(cond_dict)

        # Autoregressive generation
        with torch.no_grad():
            codes = self.model.generate(conditioning)
            wavs = self.model.autoencoder.decode(codes).cpu()

        # Save to disk
        torchaudio.save(output_path, wavs[0], self.model.autoencoder.sampling_rate)
        return output_path

if __name__ == "__main__":
    # Example usage
    # Requires a valid reference.mp3 file to exist
    try:
        tts = ZonosTTSModel(reference_audio_path="assets/reference.mp3")
        output = tts.generate_speech("Hello world, this is a test of the Zonos TTS pipeline.")
        print(f"Speech generated successfully: {output}")
    except FileNotFoundError as e:
        print(e)
