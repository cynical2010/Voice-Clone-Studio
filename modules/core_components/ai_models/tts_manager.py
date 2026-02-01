"""
TTS Model Manager

Centralized management for all TTS models (Qwen3, VibeVoice, etc.)
"""

import torch
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional
from qwen_tts import Qwen3TTSModel

from .model_utils import (
    get_device, get_dtype, get_attention_implementation,
    check_model_available_locally, empty_cuda_cache, log_gpu_memory
)


class TTSManager:
    """Manages all TTS models with lazy loading and VRAM optimization."""
    
    def __init__(self, user_config: Dict = None, samples_dir: Path = None):
        """
        Initialize TTS Manager.
        
        Args:
            user_config: User configuration dict (with attention_mechanism, low_cpu_mem_usage, offline_mode)
            samples_dir: Path to samples directory for prompt caching
        """
        self.user_config = user_config or {}
        self.samples_dir = samples_dir or Path("samples")
        
        # Model cache
        self._qwen3_base_model = None
        self._qwen3_base_size = None
        self._qwen3_voice_design_model = None
        self._qwen3_custom_voice_model = None
        self._qwen3_custom_voice_size = None
        self._vibevoice_tts_model = None
        self._vibevoice_tts_size = None
        
        # Prompt cache
        self._voice_prompt_cache = {}
        self._last_loaded_model = None
    
    def _check_and_unload_if_different(self, model_id: str):
        """If switching to a different model, unload all."""
        if self._last_loaded_model is not None and self._last_loaded_model != model_id:
            print(f"📦 Switching from {self._last_loaded_model} to {model_id} - unloading all TTS models...")
            self.unload_all()
        self._last_loaded_model = model_id
    
    def _load_model_with_attention(self, model_class, model_name: str, **kwargs):
        """
        Load a HuggingFace model with best available attention mechanism.
        
        Returns:
            Tuple: (loaded_model, attention_mechanism_used)
        """
        offline_mode = self.user_config.get("offline_mode", False)
        
        # Check local availability
        local_path = check_model_available_locally(model_name)
        if local_path:
            print(f"Found local model: {local_path}")
            model_to_load = str(local_path)
        elif offline_mode:
            raise RuntimeError(
                f"❌ Offline mode enabled but model not available locally: {model_name}\n"
                f"To use offline mode, download the model first or disable offline mode in Settings."
            )
        else:
            model_to_load = model_name
        
        mechanisms = get_attention_implementation(
            self.user_config.get("attention_mechanism", "auto")
        )
        
        last_error = None
        for attn in mechanisms:
            try:
                model = model_class.from_pretrained(
                    model_to_load,
                    attn_implementation=attn,
                    trust_remote_code=True,
                    **kwargs
                )
                print(f"✓ Model loaded with {attn}")
                return model, attn
            except Exception as e:
                error_msg = str(e).lower()
                last_error = e
                
                is_attn_error = any(
                    keyword in error_msg 
                    for keyword in ["flash", "attention", "sdpa", "not supported"]
                )
                
                if is_attn_error:
                    print(f"  {attn} not available, trying next option...")
                    continue
                else:
                    raise e
        
        raise RuntimeError(f"Failed to load model: {str(last_error)}")
    
    def get_qwen3_base(self, size: str = "1.7B"):
        """Load Qwen3 Base TTS model."""
        model_id = f"qwen3_base_{size}"
        self._check_and_unload_if_different(model_id)
        
        if self._qwen3_base_model is None:
            model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-Base"
            print(f"Loading {model_name}...")
            
            self._qwen3_base_model, _ = self._load_model_with_attention(
                Qwen3TTSModel,
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
            )
            self._qwen3_base_size = size
            print(f"Qwen3 Base TTS ({size}) loaded!")
        
        return self._qwen3_base_model
    
    def get_qwen3_voice_design(self):
        """Load Qwen3 VoiceDesign model (1.7B only)."""
        self._check_and_unload_if_different("qwen3_voice_design")
        
        if self._qwen3_voice_design_model is None:
            print("Loading Qwen3 VoiceDesign model (1.7B)...")
            
            self._qwen3_voice_design_model, _ = self._load_model_with_attention(
                Qwen3TTSModel,
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map="cuda:0",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
            )
            print("VoiceDesign model loaded!")
        
        return self._qwen3_voice_design_model
    
    def get_qwen3_custom_voice(self, size: str = "1.7B"):
        """Load Qwen3 CustomVoice model."""
        model_id = f"qwen3_custom_voice_{size}"
        self._check_and_unload_if_different(model_id)
        
        if self._qwen3_custom_voice_model is None:
            model_name = f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
            print(f"Loading {model_name}...")
            
            self._qwen3_custom_voice_model, _ = self._load_model_with_attention(
                Qwen3TTSModel,
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
            )
            self._qwen3_custom_voice_size = size
            print(f"CustomVoice model ({size}) loaded!")
        
        return self._qwen3_custom_voice_model
    
    def get_vibevoice_tts(self, size: str = "1.5B"):
        """Load VibeVoice TTS model."""
        model_id = f"vibevoice_tts_{size}"
        self._check_and_unload_if_different(model_id)
        
        if self._vibevoice_tts_model is None:
            print(f"Loading VibeVoice TTS ({size})...")
            try:
                from modules.vibevoice_tts.modular.modeling_vibevoice_inference import (
                    VibeVoiceForConditionalGenerationInference
                )
                import warnings
                
                # Map size to model path
                if size == "Large (4-bit)":
                    model_path = "FranckyB/VibeVoice-Large-4bit"
                    try:
                        import bitsandbytes
                    except ImportError:
                        raise ImportError(
                            "bitsandbytes required for 4-bit models. Install with: pip install bitsandbytes"
                        )
                else:
                    model_path = f"FranckyB/VibeVoice-{size}"
                
                import logging
                logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    
                    self._vibevoice_tts_model, _ = self._load_model_with_attention(
                        VibeVoiceForConditionalGenerationInference,
                        model_path,
                        dtype=torch.bfloat16,
                        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                        low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False)
                    )
                
                self._vibevoice_tts_size = size
                print(f"VibeVoice TTS ({size}) loaded!")
            
            except ImportError as e:
                print(f"❌ VibeVoice TTS not available: {e}")
                raise
            except Exception as e:
                print(f"❌ Error loading VibeVoice TTS: {e}")
                raise
        
        return self._vibevoice_tts_model
    
    def unload_all(self):
        """Unload all TTS models to free VRAM."""
        freed = []
        
        if self._qwen3_base_model is not None:
            del self._qwen3_base_model
            self._qwen3_base_model = None
            freed.append("Qwen3 Base")
        
        if self._qwen3_voice_design_model is not None:
            del self._qwen3_voice_design_model
            self._qwen3_voice_design_model = None
            freed.append("Qwen3 VoiceDesign")
        
        if self._qwen3_custom_voice_model is not None:
            del self._qwen3_custom_voice_model
            self._qwen3_custom_voice_model = None
            freed.append("Qwen3 CustomVoice")
        
        if self._vibevoice_tts_model is not None:
            del self._vibevoice_tts_model
            self._vibevoice_tts_model = None
            freed.append("VibeVoice TTS")
        
        if freed:
            empty_cuda_cache()
            print(f"🗑️ Unloaded TTS models: {', '.join(freed)}")
        
        return bool(freed)
    
    # Voice prompt caching
    def get_prompt_cache_path(self, sample_name: str, model_size: str = "1.7B") -> Path:
        """Get path to cached voice prompt."""
        return self.samples_dir / f"{sample_name}_{model_size}.pt"
    
    def compute_sample_hash(self, wav_path: str, ref_text: str) -> str:
        """Compute hash of sample to detect changes."""
        hasher = hashlib.md5()
        with open(wav_path, 'rb') as f:
            hasher.update(f.read())
        hasher.update(ref_text.encode('utf-8'))
        return hasher.hexdigest()
    
    def save_voice_prompt(self, sample_name: str, prompt_items, sample_hash: str, model_size: str = "1.7B") -> bool:
        """Save voice prompt to cache."""
        cache_path = self.get_prompt_cache_path(sample_name, model_size)
        try:
            # Move tensors to CPU
            if isinstance(prompt_items, dict):
                cpu_prompt = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in prompt_items.items()
                }
            elif isinstance(prompt_items, (list, tuple)):
                cpu_prompt = [
                    item.cpu() if isinstance(item, torch.Tensor) else item
                    for item in prompt_items
                ]
            else:
                cpu_prompt = prompt_items.cpu() if isinstance(prompt_items, torch.Tensor) else prompt_items
            
            cache_data = {
                'prompt': cpu_prompt,
                'hash': sample_hash,
                'version': '1.0'
            }
            torch.save(cache_data, cache_path)
            print(f"Saved voice prompt: {cache_path}")
            return True
        except Exception as e:
            print(f"Failed to save voice prompt: {e}")
            return False
    
    def load_voice_prompt(self, sample_name: str, expected_hash: str, model_size: str = "1.7B") -> Optional[dict]:
        """Load voice prompt from cache if valid."""
        cache_key = f"{sample_name}_{model_size}"
        
        # Check memory cache first
        if cache_key in self._voice_prompt_cache:
            cached = self._voice_prompt_cache[cache_key]
            if cached['hash'] == expected_hash:
                print(f"Using cached prompt: {sample_name}")
                return cached['prompt']
        
        # Check disk cache
        cache_path = self.get_prompt_cache_path(sample_name, model_size)
        if not cache_path.exists():
            return None
        
        try:
            cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
            
            if cache_data.get('hash') != expected_hash:
                print(f"Sample changed, cache invalidated: {sample_name}")
                return None
            
            # Move to device
            cached_prompt = cache_data['prompt']
            device = get_device()
            
            if isinstance(cached_prompt, dict):
                prompt_items = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in cached_prompt.items()
                }
            elif isinstance(cached_prompt, (list, tuple)):
                prompt_items = [
                    item.to(device) if isinstance(item, torch.Tensor) else item
                    for item in cached_prompt
                ]
            else:
                prompt_items = cached_prompt.to(device) if isinstance(cached_prompt, torch.Tensor) else cached_prompt
            
            # Store in memory cache
            self._voice_prompt_cache[cache_key] = {
                'prompt': prompt_items,
                'hash': expected_hash
            }
            
            print(f"Loaded voice prompt from cache: {cache_path}")
            return prompt_items
        
        except Exception as e:
            print(f"Failed to load voice prompt cache: {e}")
            return None


# Global singleton instance
_tts_manager = None


def get_tts_manager(user_config: Dict = None, samples_dir: Path = None) -> TTSManager:
    """Get or create the global TTS manager."""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = TTSManager(user_config, samples_dir)
    return _tts_manager
