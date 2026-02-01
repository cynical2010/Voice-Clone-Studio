"""
Tab Modules Extraction

This file documents the tab extraction from voice_clone_studio.py.

# Extracted Tabs

## tab_voice_clone.py
**Source:** Lines 4113-4506
**Purpose:** Clone voices from samples using Qwen3-TTS or VibeVoice
**Features:**
- Sample selection and preview
- Voice cloning generation
- Qwen3 advanced parameters (emotion, sampling, tokens)
- VibeVoice advanced parameters (CFG, steps, sampling)
- Language toggle based on model
- Emotion preset management

## tab_voice_presets.py
**Source:** Lines 4507-4859
**Purpose:** Use pre-trained voices or custom trained models
**Features:**
- Premium speaker selection
- Trained model selection
- Style instructions for generation
- Model size selection
- Emotion presets (trained models only)
- Voice type toggling (premium vs trained)

## tab_conversation.py
**Source:** Lines 4860-5387
**Purpose:** Create multi-speaker conversations
**Features:**
- VibeVoice long-form generation
- Qwen Base multi-speaker with samples
- Qwen CustomVoice with preset speakers
- Model-specific UI toggling
- Pause controls for Qwen modes
- Advanced parameters per model
- Sample refresh for all modes

## tab_prep_samples.py
**Source:** Lines 5481-5716
**Purpose:** Prepare audio samples for voice cloning
**Features:**
- Browse and preview existing samples
- Load samples to editor
- Audio file upload and editing
- Audio processing (normalize, denoise, mono conversion)
- Transcription (Whisper or VibeVoice ASR)
- Save samples with metadata
- Delete and cache management

## tab_finetune_dataset.py
**Source:** Lines 5813-6077
**Purpose:** Manage and prepare finetuning datasets
**Features:**
- Dataset folder organization
- File browser and preview
- Audio trimming and processing
- Individual and batch transcription
- Transcript editing
- File management (save, delete)

## tab_train_model.py
**Source:** Lines 6078-6323
**Purpose:** Train custom voice models
**Features:**
- Dataset and reference audio selection
- Training parameter configuration
- Batch size, learning rate, epochs
- Model name validation (modal)
- Training status monitoring
- Reference audio preview

## tab_settings.py
**Source:** Lines 6368-end
**Purpose:** Configure global application settings
**Features:**
- Model loading options (CPU memory, attention mechanism)
- Offline mode toggle
- Audio notifications toggle
- Model download from HuggingFace
- Folder path configuration
- Settings auto-save

# Shared State Dependencies

All tabs require the following from shared_state:

## User Configuration
- `_user_config`: User preferences dictionary
- `_active_emotions`: Currently loaded emotions

## Helper Functions
- `save_preference(key, value)`: Save user preference
- `save_config(config)`: Save full config
- `get_sample_choices()`: Get available sample names
- `get_available_samples()`: Get detailed sample info
- `get_emotion_choices(emotions)`: Get emotion names
- `get_trained_models()`: Get trained model list

## UI Generators
- `create_qwen_advanced_params()`: Generate Qwen UI
- `create_vibevoice_advanced_params()`: Generate VibeVoice UI
- `create_emotion_intensity_slider()`: Generate emotion slider
- `apply_emotion_preset()`: Apply emotion to parameters

## Processing Functions
- `generate_audio()`: Generate speech from text
- `generate_custom_voice()`: Generate with custom voice
- `generate_with_trained_model()`: Generate with trained model
- `generate_conversation()`: Generate Qwen conversation
- `generate_conversation_base()`: Generate Qwen Base conversation
- `generate_vibevoice_longform()`: Generate VibeVoice conversation
- `transcribe_audio()`: Transcribe with Whisper/VibeVoice

## Modal Functions
- `show_input_modal_js()`: JavaScript for input modal
- `show_confirmation_modal_js()`: JavaScript for confirmation modal

## Trigger Components
- `input_trigger`: Hidden component that triggers on modal input
- `confirm_trigger`: Hidden component that triggers on modal confirmation
- `sample_dropdown`: Reference to Voice Clone tab's sample dropdown (for updates)

## Configuration Constants
- `LANGUAGES`: List of supported languages
- `VOICE_CLONE_OPTIONS`: Available voice clone models
- `CUSTOM_VOICE_SPEAKERS`: Premium speaker list
- `MODEL_SIZES_CUSTOM`: Model size options for CustomVoice
- `MODEL_SIZES_BASE`: Model size options for Qwen Base
- `MODEL_SIZES_VIBEVOICE`: Model size options for VibeVoice
- `DEFAULT_VOICE_CLONE_MODEL`: Default voice clone model
- `WHISPER_AVAILABLE`: Whether Whisper is available
- `DEEPFILTER_AVAILABLE`: Whether DeepFilterNet is available
- `DATASETS_DIR`: Path to datasets directory
- `SAMPLES_DIR`: Path to samples directory
- `OUTPUT_DIR`: Path to output directory

# Integration Pattern

Each tab module follows this pattern:

1. **create_tab(shared_state)** - Creates UI components, returns dict
2. **setup_events(components, shared_state)** - Wires up event handlers
3. **get_tab_class** - Exports the Tab class for registry

Example usage in main file:
```python
from modules.tabs.tab_voice_clone import get_tab_class as get_voice_clone_tab
VoiceCloneTab = get_voice_clone_tab()
components = VoiceCloneTab.create_tab(shared_state)
VoiceCloneTab.setup_events(components, shared_state)
```

# Notes for Integration

- All tabs store component references in dictionaries
- Event handlers are defined inline within setup_events()
- Shared functions are accessed via shared_state dict
- Modal callbacks use context prefixes to filter events
- Some tabs reference other tabs' components (cross-tab updates)
