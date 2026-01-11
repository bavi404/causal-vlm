"""Music-AVQA dataset loader."""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MusicAVQADataset(Dataset):
    """
    Music-AVQA dataset loader.
    
    Supports both JSON and CSV annotation files.
    Expected format:
    - JSON: List of dicts with keys: image_path, audio_path, question, answer
    - CSV: Columns: image_path, audio_path, question, answer
    """
    
    def __init__(
        self,
        annotations_file: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        image_key: str = "image_path",
        audio_key: str = "audio_path",
        question_key: str = "question",
        answer_key: str = "answer",
    ):
        """
        Initialize Music-AVQA dataset.
        
        Args:
            annotations_file: Path to JSON or CSV annotations file
            data_root: Root directory for data files (if paths in annotations are relative)
            image_key: Key/column name for image paths
            audio_key: Key/column name for audio paths
            question_key: Key/column name for questions
            answer_key: Key/column name for answers
        """
        annotations_file = Path(annotations_file)
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        self.data_root = Path(data_root) if data_root else None
        self.image_key = image_key
        self.audio_key = audio_key
        self.question_key = question_key
        self.answer_key = answer_key
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        
        # Filter out invalid entries
        self.annotations = [
            ann for ann in self.annotations
            if self._validate_entry(ann)
        ]
    
    def _load_annotations(self, file_path: Path) -> List[Dict]:
        """Load annotations from JSON or CSV file."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both list of dicts and dict with list
                if isinstance(data, dict):
                    # Assume it's a dict with a list under some key
                    for key in ['data', 'annotations', 'items']:
                        if key in data:
                            return data[key]
                    raise ValueError(f"Could not find data list in JSON. Keys: {list(data.keys())}")
                return data
        
        elif suffix == '.csv':
            annotations = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    annotations.append(dict(row))
            return annotations
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")
    
    def _validate_entry(self, entry: Dict) -> bool:
        """Validate that entry has required keys."""
        required_keys = [self.image_key, self.audio_key, self.question_key, self.answer_key]
        return all(key in entry and entry[key] for key in required_keys)
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve file path, optionally prepending data_root."""
        path = Path(path)
        if path.is_absolute():
            return path
        if self.data_root:
            return self.data_root / path
        return path
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, List[str]]]:
        """
        Get a single dataset item.
        
        Returns:
            Dictionary with keys: image_paths, audio_paths, question, answer
        """
        entry = self.annotations[idx]
        
        # Handle both single paths and lists
        image_path = entry[self.image_key]
        audio_path = entry[self.audio_key]
        
        # Convert to lists for consistency
        if isinstance(image_path, str):
            image_paths = [str(self._resolve_path(image_path))]
        else:
            image_paths = [str(self._resolve_path(p)) for p in image_path]
        
        if isinstance(audio_path, str):
            audio_paths = [str(self._resolve_path(audio_path))]
        else:
            audio_paths = [str(self._resolve_path(p)) for p in audio_path]
        
        return {
            "image_paths": image_paths,
            "audio_paths": audio_paths,
            "question": entry[self.question_key],
            "answer": entry[self.answer_key],
        }


def compute_and_save_embeddings(
    dataset: MusicAVQADataset,
    model,
    output_dir: Union[str, Path],
    batch_size: int = 32,
    device: str = "cuda",
    image_embedding_key: str = "image",
    audio_embedding_key: str = "audio",
):
    """
    Compute embeddings for all images and audio in dataset and save to disk.
    
    Args:
        dataset: MusicAVQADataset instance
        model: Model with get_embeddings method (e.g., ImageBindWrapper)
        output_dir: Directory to save .pt files
        batch_size: Batch size for processing
        device: Device to run model on
        image_embedding_key: Key for image embeddings in model output
        audio_embedding_key: Key for audio embeddings in model output
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    if hasattr(model, 'device'):
        device = model.device
    
    image_embeddings = []
    audio_embeddings = []
    
    print(f"Computing embeddings for {len(dataset)} samples...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing"):
            item = dataset[idx]
            
            # Get image embedding
            if item["image_paths"]:
                img_emb = model.get_embeddings(image=item["image_paths"][0])
                image_embeddings.append(img_emb[image_embedding_key].cpu())
            else:
                image_embeddings.append(None)
            
            # Get audio embedding
            if item["audio_paths"]:
                audio_emb = model.get_embeddings(audio=item["audio_paths"][0])
                audio_embeddings.append(audio_emb[audio_embedding_key].cpu())
            else:
                audio_embeddings.append(None)
    
    # Save embeddings
    image_path = output_dir / "image_embeddings.pt"
    audio_path = output_dir / "audio_embeddings.pt"
    
    torch.save(image_embeddings, image_path)
    torch.save(audio_embeddings, audio_path)
    
    print(f"Saved image embeddings to {image_path}")
    print(f"Saved audio embeddings to {audio_path}")
    
    return image_embeddings, audio_embeddings


def load_embeddings(embeddings_dir: Union[str, Path]):
    """
    Load precomputed embeddings from disk.
    
    Args:
        embeddings_dir: Directory containing image_embeddings.pt and audio_embeddings.pt
    
    Returns:
        Tuple of (image_embeddings, audio_embeddings) lists
    """
    embeddings_dir = Path(embeddings_dir)
    image_path = embeddings_dir / "image_embeddings.pt"
    audio_path = embeddings_dir / "audio_embeddings.pt"
    
    image_embeddings = torch.load(image_path, map_location='cpu')
    audio_embeddings = torch.load(audio_path, map_location='cpu')
    
    return image_embeddings, audio_embeddings


