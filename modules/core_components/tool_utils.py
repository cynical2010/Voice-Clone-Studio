"""
Shared utilities for tool modules.

Provides helpers and utilities that tools can use.
"""

import json
from pathlib import Path
import markdown

# Config file path (relative to project root)
CONFIG_FILE = Path(__file__).parent.parent.parent / "config.json"

# Shared CSS for all tools
# - Hides trigger widgets used by modal system
# - Styles file list groups for prep_samples, finetune_dataset, etc.
SHARED_CSS = """
#confirm-trigger {
    display: none !important;
}
#input-trigger {
    display: none !important;
}
#finetune-files-group > div {
    display: grid !important;
}
#finetune-files-container {
    max-height: 400px;
    overflow-y: auto;
}
#finetune-files-group label {
    background: none !important;
    border: none !important;
    padding: 4px 8px !important;
    margin: 2px 0 !important;
    box-shadow: none !important;
}
#finetune-files-group label:hover {
    background: rgba(255, 255, 255, 0.05) !important;
}
#output-files-group > div {
    display: grid !important;
}
#output-files-container {
    max-height: 800px;
    overflow-y: auto;
}
#output-files-group label {
    background: none !important;
    border: none !important;
    padding: 4px 8px !important;
    margin: 2px 0 !important;
    box-shadow: none !important;
}
#output-files-group label:hover {
    background: rgba(255, 255, 255, 0.05) !important;
}
"""

# Alias for backward compatibility (if any tools imported this)
TRIGGER_HIDE_CSS = SHARED_CSS

def load_config():
    """Load user preferences from config file.
    
    Returns:
        dict: User configuration with defaults
    """
    default_config = {
        "transcribe_model": "Whisper",
        "tts_base_size": "Large",
        "custom_voice_size": "Large",
        "voice_clone_model": "Qwen3 - Large",
        "language": "Auto",
        "conv_pause_duration": 0.5,
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
        "emotions": None,
        "conv_model_type": "Qwen CustomVoice",
        "conv_model_size": "Large",
        "conv_base_model_size": "Large",
        "vibevoice_model_size": "Small",
        "conv_pause_linebreak": 0.5,
        "conv_pause_period": 0.3,
        "conv_pause_comma": 0.2,
        "conv_pause_question": 0.4,
        "conv_pause_hyphen": 0.15
    }
    
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                # Merge with defaults to handle new settings
                default_config.update(saved_config)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
    
    return default_config


def save_config(config):
    """Save user preferences to config file.
    
    Args:
        config: Dictionary of user preferences
    """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")


def save_preference(config, key, value):
    """Save a single preference to config file.
    
    Args:
        config: Current config dictionary (will be modified)
        key: Preference key
        value: Preference value
    """
    config[key] = value
    save_config(config)


def format_help_html(markdown_text, height="70vh"):
    """Convert markdown to HTML with scrollable container styling that matches Gradio components.

    Args:
        markdown_text: Markdown content to convert
        height: CSS height value (default: "70vh")
    """
    html_content = markdown.markdown(
        markdown_text,
        extensions=['fenced_code', 'tables', 'nl2br']
    )
    return f"""
    <div style="
        width: 100%;
        max-height: {height};
        overflow-y: auto;
        box-sizing: border-box;
        color: var(--block-label-text-color);
        font-size: var(--block-text-size);
        font-family: var(--font);
        line-height: 1.6;
    ">
        {html_content}
    </div>
    """
