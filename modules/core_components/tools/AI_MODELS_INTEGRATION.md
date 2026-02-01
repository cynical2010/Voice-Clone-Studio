# AI Models Integration Guide for Tools

## Overview

Tools should no longer directly call model loading functions. Instead, they receive AI managers through `shared_state`.

## Pattern

### Before (Old Way)
```python
# Scattered throughout tool code
from voice_clone_studio import get_tts_model, generate_audio

model = get_tts_model("1.7B")
audio = model.generate(text, ...)
```

### After (New Way)
```python
# In setup_events:
def setup_events(cls, components, shared_state):
    tts_manager = shared_state['tts_manager']
    asr_manager = shared_state['asr_manager']
    
    # Use managers
    model = tts_manager.get_qwen3_base("1.7B")
    audio = model.generate(...)
```

## What Goes in shared_state

The main `voice_clone_studio.py` will provide:

```python
shared_state = {
    # AI Managers
    'tts_manager': TTSManager instance,
    'asr_manager': ASRManager instance,
    
    # Config
    'user_config': dict,
    'active_emotions': dict,
    
    # Utilities
    'save_preference': func,
    'LANGUAGES': list,
    # ... etc
}
```

## Tool Integration Steps

1. **In `setup_events()`**, get managers from shared_state:
```python
tts_manager = shared_state.get('tts_manager')
asr_manager = shared_state.get('asr_manager')
```

2. **Replace model loading calls**:
```python
# Old
model = get_tts_model("1.7B")

# New
tts_manager = shared_state['tts_manager']
model = tts_manager.get_qwen3_base("1.7B")
```

3. **For audio generation**, keep using shared_state functions:
```python
generate_audio = shared_state['generate_audio']
result = generate_audio(...)
```

## Tools That Need Updates

### TTS Tools (Need tts_manager)
- [ ] voice_clone.py - Uses Qwen3 Base, VibeVoice TTS
- [ ] voice_presets.py - Uses Qwen3 CustomVoice
- [ ] conversation.py - Uses Qwen3 CustomVoice, VibeVoice TTS
- [ ] voice_design.py - Uses Qwen3 VoiceDesign

### ASR Tools (Need asr_manager)
- [ ] prep_samples.py - Uses Whisper, VibeVoice ASR
- [ ] finetune_dataset.py - Uses Whisper, VibeVoice ASR

### No Changes Needed
- [x] output_history.py - No AI models
- [x] settings.py - No AI models
- [x] train_model.py - No AI models
- [x] help.py - No AI models

## Example: Updating voice_clone.py

### Current Structure (simplified)
```python
class VoiceCloneTab(Tab):
    @classmethod
    def setup_events(cls, components, shared_state):
        def on_generate():
            # Calls shared_state['generate_audio']
            result = shared_state['generate_audio'](...)
```

### After Update
```python
class VoiceCloneTab(Tab):
    @classmethod
    def setup_events(cls, components, shared_state):
        tts_manager = shared_state.get('tts_manager')
        
        def on_generate():
            # Can now use: tts_manager.get_qwen3_base()
            # But still call generate_audio from shared_state
            result = shared_state['generate_audio'](...)
```

## Main File Integration

In `voice_clone_studio.py`, before calling tool setup:

```python
from modules.core_components.ai_models import get_tts_manager, get_asr_manager

def create_ui():
    tts_manager = get_tts_manager(_user_config, SAMPLES_DIR)
    asr_manager = get_asr_manager(_user_config)
    
    shared_state = {
        'tts_manager': tts_manager,
        'asr_manager': asr_manager,
        # ... other stuff
    }
    
    # Create tools
    tool_components = create_enabled_tools(shared_state)
    setup_tool_events(tool_components, shared_state)
```

## Notes

- Tools don't need to worry about model loading - just request from managers
- Managers handle VRAM optimization automatically
- Audio generation still goes through shared_state functions
- No need to update tool UI code - only event handlers

## Status

This guide documents the integration pattern. Tools will be updated systematically to follow this pattern.
