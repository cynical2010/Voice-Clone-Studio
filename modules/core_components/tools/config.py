"""
Configuration schema for enabled/disabled tabs.

Add this to your config.json to control which tabs appear in the app.
"""

# Example config.json with tab settings
EXAMPLE_CONFIG = {
    "language": "Auto",
    "voice_clone_model": "Qwen3-TTS-12Hz-0.6B-Base",
    
    # Tab enable/disable settings
    "enabled_tabs": {
        # Generation tabs
        "Voice Clone": True,
        "Voice Presets": True,
        "Conversation": True,
        "Voice Design": True,
        
        # Preparation & utilities
        "Prep Samples": True,
        "Output History": False,
        
        # Training tabs (disabled by default for non-developers)
        "Finetune Dataset": False,
        "Train Model": False,
        
        # Information & configuration
        "Help Guide": True,
        "Settings": True
    },
    
    # ... other config options
}


# Tab categories for organization
TAB_CATEGORIES = {
    "generation": [
        "Voice Clone",
        "Voice Presets", 
        "Conversation",
        "Voice Design"
    ],
    "preparation": [
        "Prep Samples",
        "Output History"
    ],
    "training": [
        "Finetune Dataset",
        "Train Model"
    ],
    "utility": [
        "Help Guide",
        "Settings"
    ]
}


# Which tabs require which features
TAB_DEPENDENCIES = {
    "Voice Presets": ["Prep Samples"],        # Need samples to create presets
    "Finetune Dataset": ["Prep Samples"],     # Need samples to finetune
    "Train Model": ["Finetune Dataset"],      # Need dataset before training
    "Conversation": ["Voice Clone"],          # Conversation uses voice engine
}


# Suggested presets for different user types
USER_PROFILES = {
    "beginner": {
        "enabled_tabs": {
            "Voice Clone": True,
            "Voice Presets": True,
            "Conversation": True,
            "Voice Design": False,
            "Prep Samples": False,
            "Output History": False,
            "Finetune Dataset": False,
            "Train Model": False,
            "Help Guide": True,
            "Settings": True
        },
        "description": "Simple voice cloning from samples"
    },
    
    "creator": {
        "enabled_tabs": {
            "Voice Clone": True,
            "Voice Presets": True,
            "Conversation": True,
            "Voice Design": True,
            "Prep Samples": True,
            "Output History": True,
            "Finetune Dataset": False,
            "Train Model": False,
            "Help Guide": True,
            "Settings": True
        },
        "description": "Full voice creation workflow"
    },
    
    "developer": {
        "enabled_tabs": {
            "Voice Clone": True,
            "Voice Presets": True,
            "Conversation": True,
            "Voice Design": True,
            "Prep Samples": True,
            "Output History": True,
            "Finetune Dataset": True,
            "Train Model": True,
            "Help Guide": True,
            "Settings": True
        },
        "description": "All features including model training"
    }
}


def get_user_config_for_profile(profile_name: str) -> dict:
    """
    Get configuration for a user profile.
    
    Args:
        profile_name: "beginner", "creator", or "developer"
    
    Returns:
        Dict with enabled_tabs configuration
    """
    if profile_name not in USER_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}")
    
    return USER_PROFILES[profile_name]["enabled_tabs"]
