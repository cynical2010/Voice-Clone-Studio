# Wrapper Functions Implementation Guide

## Current State

✅ **Fixed**: All wrapper functions now use managers directly, NOT importing from `voice_clone_studio.py`

✅ **Architecture**: 
- Global managers (`_tts_manager`, `_asr_manager`) initialized in `create_ui()`
- Wrapper functions access these globals
- All functions have TODO comments showing where to add manager calls
- Shared_state passes managers to tools if needed

❌ **TODO**: Actually implement the wrapper functions

## Wrapper Functions to Implement

### Generation Functions (Using tts_manager)

#### 1. `generate_audio()` (Line ~155)
**Uses**: `_tts_manager.get_qwen3_base_model()` or `_tts_manager.get_vibevoice_tts_model()`
**Needs**:
- Parse model_selection to determine engine (Qwen vs VibeVoice)
- Get sample audio from SAMPLES_DIR
- Call appropriate manager method for generation
- Return (audio_wav, status_message)

#### 2. `generate_voice_design()` (Line ~168)
**Uses**: `_tts_manager.get_qwen3_voice_design_model()`
**Needs**:
- Load voice design model via manager
- Generate voice from description
- Save to output if requested
- Return (audio_wav, status_message)

#### 3. `generate_conversation()` (Line ~178)
**Uses**: `_tts_manager.get_qwen3_custom_voice_model()`
**Needs**:
- Load custom voice model via manager
- Parse speaker for this model
- Apply pause controls
- Return (audio_wav, status_message)

#### 4. `generate_conversation_base()` (Line ~187)
**Uses**: `_tts_manager.get_qwen3_base_model()`
**Needs**:
- Load base model via manager
- Handle multiple voice samples
- Apply emotion if provided
- Apply pause controls
- Return (audio_wav, status_message)

#### 5. `generate_vibevoice_longform()` (Line ~198)
**Uses**: `_tts_manager.get_vibevoice_tts_model()`
**Needs**:
- Load VibeVoice model via manager
- Process multiple voice samples
- Apply CFG scale and num_steps
- Return (audio_wav, status_message)

### Transcription Functions (Using asr_manager)

#### 6. `transcribe_audio()` (Line ~208)
**Uses**: `_asr_manager.transcribe()` or similar
**Needs**:
- Determine which model to use (Whisper vs VibeVoice ASR)
- Load audio from path
- Transcribe using manager
- Return transcribed_text

#### 7. `batch_transcribe_folder()` (Line ~216)
**Uses**: `_asr_manager.transcribe()` in a loop
**Needs**:
- Iterate through audio files in folder
- Transcribe each (or skip if exists)
- Save transcripts
- Return status_message

### Sample Management Functions

#### 8-16. Sample Functions (Lines ~225-280)
These need direct implementation without manager calls:
- `get_sample_choices()` - List samples in SAMPLES_DIR
- `get_available_samples()` - Return {name, wav_path, ref_text} for each
- `load_existing_sample()` - Load audio/text from sample
- `refresh_samples()` - Rescan SAMPLES_DIR
- `delete_sample()` - Delete sample files
- `save_as_sample()` - Save audio to SAMPLES_DIR as sample
- `on_prep_audio_load()` - Process uploaded audio
- etc.

### Audio Processing Functions (Lines ~275+)

These are simple transformations using libraries:
- `normalize_audio()` - Normalize volume
- `convert_to_mono()` - Convert to mono
- `clean_audio()` - Noise reduction (using deepfilternet if available)

## Key Manager Methods Available

### TTS Manager
```python
_tts_manager.get_qwen3_base_model(model_size)
_tts_manager.get_qwen3_custom_voice_model(model_size)
_tts_manager.get_qwen3_voice_design_model()
_tts_manager.get_vibevoice_tts_model(model_size)
_tts_manager.unload_all()
_tts_manager.get_voice_prompt_cache(sample_name, model_size)
```

### ASR Manager
```python
_asr_manager.transcribe(audio_path, language, model_type)
_asr_manager.unload_all()
```

## Implementation Priority

1. **High Priority** (For app to work):
   - `generate_audio()` - Most used
   - `transcribe_audio()` - Text input needed
   - `get_sample_choices()` - Sample loading
   - `load_existing_sample()` - Sample browser

2. **Medium Priority**:
   - `generate_conversation()` 
   - `generate_conversation_base()`
   - `delete_sample()`
   - `save_as_sample()`

3. **Lower Priority**:
   - `generate_voice_design()`
   - Audio processing functions
   - Helper functions

## Testing Each Function

Once implemented, test like:
```python
from voice_clone_studio_minimal import create_ui, generate_audio

# This initializes managers
app, theme, custom_css, *_ = create_ui()

# Then wrapper functions are ready to use
audio, status = generate_audio("sample_name", "Hello world", "English", -1)
```

## Next Steps

1. Extract sample management code from voice_clone_studio.py
2. Implement in wrapper functions here
3. Extract generation code using manager methods
4. Replace all manager calls in wrapper functions
5. Delete old voice_clone_studio.py functions one by one as they're replaced
