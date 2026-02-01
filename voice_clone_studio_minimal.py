"""
Voice Clone Studio - Main Application

Minimal orchestrator that loads modular tools and wires them together.
All tab implementations are in modules/core_components/tools/
"""

import os
import sys
from pathlib import Path
import torch
import json
import random
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
# HELPER FUNCTIONS
# ============================================================================

# [These would be imported or defined - for now, we rely on tools to have what they need]
# Helper functions like get_sample_choices, transcribe_audio, etc. can be:
# 1. Defined here (as needed by tools)
# 2. Imported from tool modules
# 3. Passed through shared_state

def play_completion_beep():
    """Play completion sound if enabled."""
    if not _user_config.get("browser_notifications", True):
        return
    try:
        import soundfile as sf
        import numpy as np
        # Simple beep
        sr = 22050
        duration = 0.2
        freq = 1000
        t = np.linspace(0, duration, int(sr * duration))
        wav = np.sin(2 * np.pi * freq * t) * 0.3
        # Play via browser notification (or just print)
        print("\a", flush=True)
    except:
        print("\a", flush=True)

def save_preference(key, value):
    """Save a user preference."""
    _user_config[key] = value
    save_config(_user_config)

# ============================================================================
# UI CREATION
# ============================================================================

def create_ui():
    """Create the Gradio interface with modular tools."""

    # Initialize AI managers
    tts_manager = get_tts_manager(_user_config, SAMPLES_DIR)
    asr_manager = get_asr_manager(_user_config)

    # Load theme
    theme = gr.themes.Base.load('modules/core_components/theme.json')

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
            # AI Managers
            'tts_manager': tts_manager,
            'asr_manager': asr_manager,

            # Config
            'user_config': _user_config,
            '_user_config': _user_config,
            '_active_emotions': _active_emotions,

            # Directories
            'OUTPUT_DIR': OUTPUT_DIR,
            'SAMPLES_DIR': SAMPLES_DIR,
            'DATASETS_DIR': DATASETS_DIR,
            'TEMP_DIR': TEMP_DIR,

            # UI components
            'create_qwen_advanced_params': create_qwen_advanced_params,
            'create_vibevoice_advanced_params': create_vibevoice_advanced_params,
            'create_emotion_intensity_slider': create_emotion_intensity_slider,
            'create_pause_controls': create_pause_controls,

            # Emotion management
            'get_emotion_choices': lambda: get_emotion_choices(_active_emotions),
            'apply_emotion_preset': calculate_emotion_values,
            'save_emotion_handler': handle_save_emotion,
            'delete_emotion_handler': handle_delete_emotion,

            # Utilities
            'save_preference': save_preference,
            'play_completion_beep': play_completion_beep,

            # Modal triggers
            'confirm_trigger': confirm_trigger,
            'input_trigger': input_trigger,
            'show_confirmation_modal_js': show_confirmation_modal_js,
            'show_input_modal_js': show_input_modal_js,

            # TODO: Add other helpers as needed by tools
            # These would be functions like:
            # - get_sample_choices
            # - transcribe_audio
            # - generate_audio
            # - etc.
        }

        # ============================================================
        # LOAD ALL MODULAR TOOLS
        # ============================================================
        with gr.Tabs():
            tab_components = create_enabled_tabs(shared_state)
            setup_tab_events(tab_components, shared_state)

        # Wire up unload button
        def on_unload_all():
            tts_manager.unload_all()
            asr_manager.unload_all()
            return "✓ All models unloaded. VRAM freed."

        unload_all_btn.click(
            on_unload_all,
            outputs=[unload_status]
        )

    return app, theme, custom_css, CONFIRMATION_MODAL_CSS, CONFIRMATION_MODAL_HEAD, INPUT_MODAL_CSS, INPUT_MODAL_HEAD


if __name__ == "__main__":
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
