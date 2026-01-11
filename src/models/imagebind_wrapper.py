"""ImageBind wrapper for easy embedding extraction."""

import sys
from pathlib import Path
from typing import Union, Optional, Dict

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from PIL import Image
from torchvision import transforms

# Add ImageBind to path
_imagebind_path = Path(__file__).parent / "imagebind"
if str(_imagebind_path) not in sys.path:
    sys.path.insert(0, str(_imagebind_path))

from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.models.multimodal_preprocessors import SimpleTokenizer


class ImageBindWrapper:
    """Wrapper for ImageBind model with simplified embedding extraction."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize ImageBind wrapper and load pretrained model.
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load pretrained model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing transform
        self.image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        
        # Audio preprocessing parameters
        self.audio_num_mel_bins = 128
        self.audio_target_length = 204
        self.audio_sample_rate = 16000
        self.audio_clip_duration = 2
        self.audio_mean = -4.268
        self.audio_std = 9.138
    
    def _preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Preprocess image input."""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                image = Image.open(f).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        image_tensor = self.image_transform(image).to(self.device)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def _preprocess_audio(
        self, 
        audio: Union[str, Path, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess audio input.
        
        Supports:
        - File path (str or Path)
        - NumPy array (waveform, shape: [samples] or [channels, samples])
        - PyTorch tensor (waveform, shape: [samples] or [channels, samples])
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            waveform, sr = torchaudio.load(str(audio))
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            # Assume default sample rate if not provided
            sr = self.audio_sample_rate
        elif isinstance(audio, torch.Tensor):
            waveform = audio.float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            sr = self.audio_sample_rate
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")
        
        # Resample if needed
        if sr != self.audio_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.audio_sample_rate
            )
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or truncate to clip_duration
        target_samples = int(self.audio_clip_duration * self.audio_sample_rate)
        current_samples = waveform.shape[1]
        
        if current_samples < target_samples:
            # Pad with zeros
            padding = target_samples - current_samples
            waveform = F.pad(waveform, (0, padding), mode='constant', value=0)
        elif current_samples > target_samples:
            # Truncate to target length
            waveform = waveform[:, :target_samples]
        
        # Convert to mel spectrogram
        waveform -= waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.audio_sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.audio_num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=10,  # DEFAULT_AUDIO_FRAME_SHIFT_MS
        )
        
        # Convert to [mel_bins, num_frames] shape
        fbank = fbank.transpose(0, 1)
        
        # Pad to target_length
        n_frames = fbank.size(1)
        p = self.audio_target_length - n_frames
        if p > 0:
            fbank = F.pad(fbank, (0, p), mode='constant', value=0)
        elif p < 0:
            fbank = fbank[:, :self.audio_target_length]
        
        # Convert to [1, mel_bins, num_frames] shape
        fbank = fbank.unsqueeze(0)
        
        # Normalize
        normalize = transforms.Normalize(mean=self.audio_mean, std=self.audio_std)
        fbank = normalize(fbank).to(self.device)
        
        # Add batch dimension: [1, 1, mel_bins, num_frames]
        return fbank.unsqueeze(0)
    
    def _preprocess_text(self, text: Union[str, list]) -> torch.Tensor:
        """Preprocess text input."""
        if isinstance(text, str):
            text = [text]
        
        # Use ImageBind's text preprocessing
        tokenizer = SimpleTokenizer(
            bpe_path=imagebind_data.return_bpe_path()
        )
        tokens = [tokenizer(t).unsqueeze(0).to(self.device) for t in text]
        tokens = torch.cat(tokens, dim=0)
        return tokens
    
    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Normalize embedding to shape [1, D] with L2 normalization.
        
        Args:
            embedding: Embedding tensor of any shape
            
        Returns:
            Normalized embedding tensor of shape [1, D]
        """
        # Flatten to [N, D] if needed
        if embedding.ndim > 2:
            # Handle multi-clip audio/video outputs by averaging
            if embedding.ndim >= 3:
                embedding = embedding.mean(dim=1)  # Average over clips
            embedding = embedding.view(-1, embedding.shape[-1])
        
        # Take first sample if batch size > 1
        if embedding.shape[0] > 1:
            embedding = embedding[0:1]
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def get_embeddings(
        self,
        image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        audio: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        text: Optional[Union[str, list]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get normalized embeddings for provided modalities.
        
        Args:
            image: Image input (path, PIL Image, or numpy array)
            audio: Audio input (path, numpy array, or torch tensor)
            text: Text input (string or list of strings)
        
        Returns:
            Dictionary with keys 'image', 'audio', 'text' containing normalized
            embeddings of shape [1, D]. Keys are only present if corresponding
            input was provided.
        """
        inputs = {}
        
        # Preprocess and add inputs
        if image is not None:
            image_tensor = self._preprocess_image(image)
            inputs[ModalityType.VISION] = image_tensor
        
        if audio is not None:
            audio_tensor = self._preprocess_audio(audio)
            inputs[ModalityType.AUDIO] = audio_tensor
        
        if text is not None:
            text_tensor = self._preprocess_text(text)
            inputs[ModalityType.TEXT] = text_tensor
        
        if not inputs:
            raise ValueError("At least one modality (image, audio, or text) must be provided")
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(inputs)
        
        # Normalize and format outputs
        result = {}
        for modality_key, embedding in embeddings.items():
            normalized_embedding = self._normalize_embedding(embedding)
            
            # Map modality keys to simple names
            if modality_key == ModalityType.VISION:
                result['image'] = normalized_embedding
            elif modality_key == ModalityType.AUDIO:
                result['audio'] = normalized_embedding
            elif modality_key == ModalityType.TEXT:
                result['text'] = normalized_embedding
        
        return result

