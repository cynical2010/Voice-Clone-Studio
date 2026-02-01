"""
Voice Clone Studio - Main Application

Minimal orchestrator that loads modular tools and wires them together.
All tab implementations are in modules/core_components/tools/

ARCHITECTURE:
- Each tool is fully independent and self-contained
- Tools import get_tts_manager() / get_asr_manager() directly (singleton pattern)
- Tools implement their own generation logic, file I/O, progress updates
- This file only provides: directories, constants, shared utilities, modals
- No wrapper functions for generation - tools handle everything themselves

REFACTORED TOOLS (fully independent):
✅ Voice Design - calls tts_manager.generate_voice_design()
✅ Voice Clone - calls tts_manager.get_qwen3_custom_voice() / get_vibevoice_tts()
✅ Voice Presets - calls tts_manager.get_qwen3_custom_voice()
🔄 Conversation - TODO: needs 3 conversation handlers added
"""

import os
import sys
from pathlib import Path
import torch
import json
import random
import tempfile
from datetime import datetime

import gradio as gr

# Core imports
from modules.core_components import (
    CONFIRMATION_MODAL_CSS,
    CONFIRMATION_MODAL_HEAD,
    CONFIRMATION_MODAL_HTML,
    INPUT_MODAL_CSS,
    INPUT_MODAL_HEAD,
    INPUT_MODAL_HTML,
    CORE_EMOTIONS,
    show_confirmation_modal_js,
    show_input_modal_js,
    load_emotions_from_config,
    get_emotion_choices,
    calculate_emotion_values,
    handle_save_emotion,
    handle_delete_emotion
)

# UI components
from modules.core_components.ui_components import (
    create_qwen_advanced_params,
    create_vibevoice_advanced_params,
    create_emotion_intensity_slider,
    create_pause_controls
)

# AI Managers
from modules.core_components.ai_models import (
    get_tts_manager,
    get_asr_manager
)

# Modular tools
from modules.core_components.tools import (
    create_enabled_tabs,
    setup_tab_events
)

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# ============================================================================
# CONFIG & SETUP
# ============================================================================

CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    """Load user preferences from config file."""
    default_config = {
        "transcribe_model": "Whisper",
        "tts_base_size": "Large",
        "custom_voice_size": "Large",
        "language": "Auto",
        "whisper_language": "Auto-detect",
        "low_cpu_mem_usage": False,
        "attention_mechanism": "auto",
        "offline_mode": False,
        "browser_notifications": True,
        "samples_folder": "samples",
        "output_folder": "output",
        "datasets_folder": "datasets",
        "temp_folder": "temp",
        "models_folder": "models",
        "trained_models_folder": "models",
        "emotions": None
    }

    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                default_config.update(saved_config)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")

    # Initialize emotions if not present
    if not default_config.get("emotions"):
        default_config["emotions"] = dict(sorted(CORE_EMOTIONS.items(), key=lambda x: x[0].lower()))
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save initial emotions: {e}")

    return default_config

def save_config(config):
    """Save user preferences to config file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")

# Load config
_user_config = load_config()
_active_emotions = load_emotions_from_config(_user_config)

# Initialize directories
SAMPLES_DIR = Path(__file__).parent / _user_config.get("samples_folder", "samples")
OUTPUT_DIR = Path(__file__).parent / _user_config.get("output_folder", "output")
DATASETS_DIR = Path(__file__).parent / _user_config.get("datasets_folder", "datasets")
TEMP_DIR = Path(__file__).parent / _user_config.get("temp_folder", "temp")

for dir_path in [SAMPLES_DIR, OUTPUT_DIR, DATASETS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# CONSTANTS - Import from central location
# ============================================================================

from modules.core_components.constants import (
    MODEL_SIZES,
    MODEL_SIZES_BASE,
    MODEL_SIZES_CUSTOM,
    MODEL_SIZES_DESIGN,
    MODEL_SIZES_VIBEVOICE,
    VOICE_CLONE_OPTIONS,
    DEFAULT_VOICE_CLONE_MODEL,
    LANGUAGES,
    CUSTOM_VOICE_SPEAKERS,
    SUPPORTED_MODELS,
    SAMPLE_RATE,
    DEFAULT_CONFIG as DEFAULT_CONFIG_TEMPLATE,
    QWEN_GENERATION_DEFAULTS,
    VIBEVOICE_GENERATION_DEFAULTS,
)

# Check optional dependencies
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from df.enhance import enhance, init_df
    DEEPFILTER_AVAILABLE = True
except ImportError:
    DEEPFILTER_AVAILABLE = False

# ============================================================================
# GLOBAL MANAGERS - Tools access via shared_state
# ============================================================================
_tts_manager = None
_asr_manager = None

def play_completion_beep():
    """Play completion sound if enabled."""
    if not _user_config.get("browser_notifications", True):
        return
    try:
        import numpy as np
        sr = 22050
        duration = 0.2
        freq = 1000
        t = np.linspace(0, duration, int(sr * duration))
        wav = np.sin(2 * np.pi * freq * t) * 0.3
        print("\a", flush=True)
    except:
        print("\a", flush=True)

def save_preference(key, value):
    """Save a user preference."""
    _user_config[key] = value
    save_config(_user_config)

# ============================================================================
# ADDITIONAL HELPER FUNCTIONS - STUBS FOR NOW
# ============================================================================
# These also need to be implemented without importing from voice_clone_studio.py

def get_sample_choices():
    """Get list of available samples."""
    # TODO: Implement directly, not importing from main file
    return []

def get_available_samples():
    """Get detailed info about all available samples."""
    # TODO: Implement directly
    return []

def get_audio_duration(audio_path):
    """Get audio duration in seconds."""
    # TODO: Implement directly
    return 0

def format_time(seconds):
    """Format seconds to MM:SS."""
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"

def apply_emotion_preset(emotion, intensity):
    """Apply emotion preset and return adjusted parameters."""
    # TODO: Implement directly
    return 0.9, 1.0, 1.05

def get_prompt_cache_path(sample_name, model_size):
    """Get cache path for voice prompt."""
    # TODO: Implement directly
    return TEMP_DIR / f"{sample_name}_{model_size}_prompt.pt"

def load_existing_sample(sample_name):
    """Load existing sample audio, text, and info."""
    # TODO: Implement directly
    return None, "", ""

def refresh_samples():
    """Refresh sample list."""
    # TODO: Implement directly
    return []

def delete_sample(confirm_value, sample_name):
    """Delete a sample."""
    # TODO: Implement directly
    return "", [], []

def clear_sample_cache(sample_name):
    """Clear cached voice prompt for a sample."""
    # TODO: Implement directly
    return ""

def on_prep_audio_load(file_info):
    """Handle audio file load in prep tab."""
    # TODO: Implement directly
    return None, ""

def normalize_audio(audio):
    """Normalize audio."""
    # TODO: Implement directly
    return audio

def convert_to_mono(audio):
    """Convert audio to mono."""
    # TODO: Implement directly
    return audio

def clean_audio(audio):
    """Clean audio (noise reduction)."""
    # TODO: Implement directly
    return audio

def save_as_sample(audio, sample_name, text):
    """Save generated audio as a sample."""
    # TODO: Implement directly
    return ""

# ============================================================================
# UI CREATION
# ============================================================================

def create_ui():
    """Create the Gradio interface with modular tools."""

    # Initialize AI managers and make them available to wrapper functions
    global _tts_manager, _asr_manager
    _tts_manager = get_tts_manager(_user_config, SAMPLES_DIR)
    _asr_manager = get_asr_manager(_user_config)

    custom_css = """
    #confirm-trigger { display: none !important; }
    #input-trigger { display: none !important; }
    """

    with gr.Blocks(title="Voice Clone Studio", theme=theme) as app:
        # Modal HTML
        gr.HTML(CONFIRMATION_MODAL_HTML)
        gr.HTML(INPUT_MODAL_HTML)

        # Hidden triggers for modals
        confirm_trigger = gr.Textbox(label="Confirm Trigger", value="", elem_id="confirm-trigger")
        input_trigger = gr.Textbox(label="Input Trigger", value="", elem_id="input-trigger")

        # Header with unload button
        with gr.Row():
            with gr.Column(scale=20):
                gr.Markdown("""
                    # 🎙️ Voice Clone Studio
                    <p style="font-size: 0.9em; color: #ffffff; margin-top: -10px;">Powered by Qwen3-TTS, VibeVoice and Whisper</p>
                    """)

            with gr.Column(scale=1, min_width=180):
                unload_all_btn = gr.Button("Clear VRAM", size="sm", variant="secondary")
                unload_status = gr.Markdown(" ", visible=True)

        # ============================================================
        # BUILD SHARED STATE - everything tools need
        # ============================================================
        shared_state = {
            # AI Managers - tools import get_tts_manager() / get_asr_manager() directly
            # These are here for backwards compatibility with tools not yet refactored
            'tts_manager': _tts_manager,
            'asr_manager': _asr_manager,

            # Config & Emotions
            'user_config': _user_config,
            '_user_config': _user_config,
            '_active_emotions': _active_emotions,

            # Directories
            'OUTPUT_DIR': OUTPUT_DIR,
            'SAMPLES_DIR': SAMPLES_DIR,
            'DATASETS_DIR': DATASETS_DIR,
            'TEMP_DIR': TEMP_DIR,

            # Constants - Model sizes, languages, speakers
            'MODEL_SIZES': MODEL_SIZES,
            'MODEL_SIZES_BASE': MODEL_SIZES_BASE,
            'MODEL_SIZES_CUSTOM': MODEL_SIZES_CUSTOM,
            'MODEL_SIZES_DESIGN': MODEL_SIZES_DESIGN,
            'MODEL_SIZES_VIBEVOICE': MODEL_SIZES_VIBEVOICE,
            'VOICE_CLONE_OPTIONS': VOICE_CLONE_OPTIONS,
            'DEFAULT_VOICE_CLONE_MODEL': DEFAULT_VOICE_CLONE_MODEL,
            'LANGUAGES': LANGUAGES,
            'CUSTOM_VOICE_SPEAKERS': CUSTOM_VOICE_SPEAKERS,
            'WHISPER_AVAILABLE': WHISPER_AVAILABLE,
            'DEEPFILTER_AVAILABLE': DEEPFILTER_AVAILABLE,

            # UI component creators
            'create_qwen_advanced_params': create_qwen_advanced_params,
            'create_vibevoice_advanced_params': create_vibevoice_advanced_params,
            'create_emotion_intensity_slider': create_emotion_intensity_slider,
            'create_pause_controls': create_pause_controls,

            # Emotion management
            'get_emotion_choices': lambda: get_emotion_choices(_active_emotions),
            'apply_emotion_preset': calculate_emotion_values,
            'save_emotion_handler': handle_save_emotion,
            'delete_emotion_handler': handle_delete_emotion,

            # Core utilities
            'save_preference': save_preference,
            'play_completion_beep': play_completion_beep,

            # Modal triggers
            'confirm_trigger': confirm_trigger,
            'input_trigger': input_trigger,
            'show_confirmation_modal_js': show_confirmation_modal_js,
            'show_input_modal_js': show_input_modal_js,

            # ============================================================
            # SAMPLE MANAGEMENT - Shared utilities
            # ============================================================
            'get_sample_choices': get_sample_choices,
            'get_available_samples': get_available_samples,
            'load_existing_sample': load_existing_sample,
            'refresh_samples': refresh_samples,
            'delete_sample': delete_sample,
            'clear_sample_cache': clear_sample_cache,
            'on_prep_audio_load': on_prep_audio_load,
            'save_as_sample': save_as_sample,
            'get_prompt_cache_path': get_prompt_cache_path,
            'get_or_create_voice_prompt': lambda: None,  # TODO: Implement

            # ============================================================
            # AUDIO PROCESSING - Shared utilities
            # ============================================================
            'normalize_audio': normalize_audio,
            'convert_to_mono': convert_to_mono,
            'clean_audio': clean_audio,
            'get_audio_duration': get_audio_duration,
            'format_time': format_time,

            # ============================================================
            # TRAINING - Shared utilities
            # ============================================================
            'get_trained_models': lambda: [],  # TODO: Implement
            'preprocess_conversation_script': lambda x: x,  # TODO: Implement
            'extract_style_instructions': lambda x: (x, ""),  # TODO: Implement
        }

        # ============================================================
        # LOAD ALL MODULAR TOOLS
        # ============================================================
        with gr.Tabs():
            tab_components = create_enabled_tabs(shared_state)
            setup_tab_events(tab_components, shared_state)

        # Wire up unload button
        def on_unload_all():
            _tts_manager.unload_all()
            _asr_manager.unload_all()
            return "✓ All models unloaded. VRAM freed."

        unload_all_btn.click(
            on_unload_all,
            outputs=[unload_status]
        )

    return app, theme, custom_css, CONFIRMATION_MODAL_CSS, CONFIRMATION_MODAL_HEAD, INPUT_MODAL_CSS, INPUT_MODAL_HEAD


if __name__ == "__main__":
    theme = gr.themes.Base.load('modules/core_components/theme.json')
    app, theme, custom_css, modal_css, modal_head, input_css, input_head = create_ui()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=custom_css + modal_css + input_css,
        head=modal_head + input_head
    )
