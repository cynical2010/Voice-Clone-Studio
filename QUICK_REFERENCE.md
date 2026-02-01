# Quick Reference - AI Managers Integration

## 🚀 TL;DR

All model loading in Voice Clone Studio now goes through centralized AI Managers:
- **TTS Manager**: Handles all Qwen3 and VibeVoice TTS models
- **ASR Manager**: Handles all Whisper and VibeVoice ASR transcription

## 📍 Where Managers Are Initialized

**File**: `voice_clone_studio.py`
**Function**: `create_ui()` (line ~4020)

```python
def create_ui():
    """Create the Gradio interface."""
    
    # Initialize AI Model Managers (global)
    global _tts_manager, _asr_manager
    _tts_manager = get_tts_manager(_user_config, SAMPLES_DIR)
    _asr_manager = get_asr_manager(_user_config)
    # ... rest of UI creation
```

## 🔌 How to Use Managers

### In Main File Wrapper Functions

```python
# TTS Generation
model = _tts_manager.get_qwen3_base("1.7B")
model = _tts_manager.get_qwen3_custom_voice("1.7B")
model = _tts_manager.get_qwen3_voice_design()
model = _tts_manager.get_vibevoice_tts("1.5B")

# ASR Transcription
model = _asr_manager.get_whisper()
model = _asr_manager.get_vibevoice_asr()

# Check capabilities
if _asr_manager.whisper_available:
    # Whisper is available
```

## 📝 Updated Wrapper Functions

| Function | Old Pattern | New Pattern |
|----------|------------|-------------|
| `generate_audio()` | `get_tts_model()` | `_tts_manager.get_qwen3_base()` |
| `generate_voice_design()` | `get_voice_design_model()` | `_tts_manager.get_qwen3_voice_design()` |
| `generate_conversation()` | `get_custom_voice_model()` | `_tts_manager.get_qwen3_custom_voice()` |
| `generate_conversation_base()` | `get_tts_model()` | `_tts_manager.get_qwen3_base()` |
| `generate_vibevoice_longform()` | `get_vibevoice_tts_model()` | `_tts_manager.get_vibevoice_tts()` |
| `transcribe_audio()` | `get_whisper_model()` / `get_vibe_voice_model()` | `_asr_manager.get_whisper()` / `_asr_manager.get_vibevoice_asr()` |

## 💾 Manager Features

### TTSManager
- ✅ Lazy-loaded model instances
- ✅ Automatic VRAM cleanup on switches
- ✅ Voice prompt caching (MD5 hash validation)
- ✅ Attention mechanism selection
- ✅ Offline mode support
- ✅ Configuration-driven behavior

### ASRManager
- ✅ Whisper availability detection
- ✅ Graceful fallback to VibeVoice ASR
- ✅ Automatic model unloading
- ✅ Configuration-driven behavior

## 🧪 Testing the Integration

```bash
# Test imports
python -c "from modules.core_components.ai_models import get_tts_manager, get_asr_manager; print('✅ Imports OK')"

# Test manager initialization
python -c "
from modules.core_components.ai_models import get_tts_manager, get_asr_manager
from pathlib import Path

config = {}
tts_mgr = get_tts_manager(config, Path('samples'))
asr_mgr = get_asr_manager(config)
print(f'TTSManager: {tts_mgr}')
print(f'ASRManager: {asr_mgr}')
print(f'Whisper: {asr_mgr.whisper_available}')
"

# Test main file import
python -c "import voice_clone_studio; print('✅ Main file OK')"
```

## 🎯 Manager Methods Quick Guide

### TTS Manager

```python
# Get models
tts_mgr.get_qwen3_base(size)              # "0.6B" or "1.7B"
tts_mgr.get_qwen3_custom_voice(size)      # "0.6B" or "1.7B"
tts_mgr.get_qwen3_voice_design()          # Always 1.7B
tts_mgr.get_vibevoice_tts(size)           # "1.5B", "Large", "Large (4-bit)"

# Cache management
tts_mgr.compute_sample_hash(wav_path, ref_text)
tts_mgr.load_voice_prompt(cache_key)
tts_mgr.save_voice_prompt(cache_key, prompt_items)

# Cleanup
tts_mgr.unload_all()
```

### ASR Manager

```python
# Get models
asr_mgr.get_whisper()
asr_mgr.get_vibevoice_asr()

# Check capabilities
asr_mgr.whisper_available  # bool property

# Cleanup
asr_mgr.unload_all()
```

## 🔄 Model Switching Behavior

When calling manager methods:
1. **First time**: Model is downloaded/loaded
2. **Subsequent calls**: Same model instance returned (cached)
3. **Different model**: Previous model automatically unloaded first
4. **VRAM cleanup**: GPU cache cleared before loading new model

## 🌐 Offline Mode

```python
# Config setting
_user_config["offline_mode"] = True

# Effect
# - Models won't download from HuggingFace
# - Must already be in local models/ folder
# - Manager will fail gracefully if not found
```

## 🎓 Configuration Options

```python
_user_config = {
    # Attention mechanism: "flash_attention_2", "sdpa", "eager", or "auto"
    "attention_mechanism": "auto",
    
    # CPU memory optimization
    "low_cpu_mem_usage": False,
    
    # Work without internet
    "offline_mode": False,
    
    # Model storage location
    "models_folder": "models",
    "trained_models_folder": "models",
    "samples_folder": "samples",
}
```

## 🐛 Troubleshooting

### Manager not initializing
```python
# Make sure _user_config is loaded and SAMPLES_DIR exists
assert _user_config is not None
assert SAMPLES_DIR.exists()
```

### Model not found
```python
# Check online/offline mode
print(f"Offline mode: {_user_config.get('offline_mode', False)}")

# Try unloading and reloading
_tts_manager.unload_all()
model = _tts_manager.get_qwen3_base("1.7B")
```

### Whisper not available
```python
# Use fallback
if not _asr_manager.whisper_available:
    model = _asr_manager.get_vibevoice_asr()
```

## 📚 See Also

- [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) - Full technical summary
- [AI_MANAGERS_INTEGRATION_COMPLETE.md](AI_MANAGERS_INTEGRATION_COMPLETE.md) - Detailed guide
- [modules/core_components/ai_models/README.md](modules/core_components/ai_models/README.md) - Manager implementation

## ✅ Status

- **Integration**: Complete ✅
- **Testing**: Verified ✅
- **Production Ready**: Yes ✅
- **Breaking Changes**: None ✅

---

**Last Updated**: Integration Complete
**Tested**: All managers verified working
**Status**: Ready for production use
