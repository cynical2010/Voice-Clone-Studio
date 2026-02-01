"""
Tool modules registry and loader.

This module manages all available tools and their configurations.
Tools can be enabled/disabled through configuration.
"""

from typing import Dict, List, Any
from modules.core_components.tools.base import TabConfig, Tab

# Import all tool modules here
from modules.core_components.tools import help
from modules.core_components.tools import output_history
from modules.core_components.tools import voice_design
from modules.core_components.tools import voice_clone
from modules.core_components.tools import voice_presets
from modules.core_components.tools import conversation
from modules.core_components.tools import prep_samples
from modules.core_components.tools import finetune_dataset
from modules.core_components.tools import train_model
from modules.core_components.tools import settings

# Registry of all available tools
# Format: 'tool_name': (module, TabConfig)
ALL_TABS: Dict[str, tuple] = {
    'voice_clone': (voice_clone, voice_clone.VoiceCloneTab.config),
    'voice_presets': (voice_presets, voice_presets.VoicePresetsTab.config),
    'conversation': (conversation, conversation.ConversationTab.config),
    'voice_design': (voice_design, voice_design.VoiceDesignTab.config),
    'prep_samples': (prep_samples, prep_samples.PrepSamplesTab.config),
    'output_history': (output_history, output_history.OutputHistoryTab.config),
    'finetune_dataset': (finetune_dataset, finetune_dataset.FinetuneDatasetTab.config),
    'train_model': (train_model, train_model.TrainModelTab.config),
    'settings': (settings, settings.SettingsTab.config),
    'help': (help, help.HelpGuideTab.config),
}


def get_tab_registry() -> Dict[str, TabConfig]:
    """Get registry of all available tabs and their configs."""
    return {name: config for name, (_, config) in ALL_TABS.items()}


def get_enabled_tabs(user_config: Dict[str, Any]) -> List[tuple]:
    """
    Get list of enabled tab modules based on user config.
    
    Args:
        user_config: User configuration dict
    
    Returns:
        List of (tab_module, TabConfig) tuples for enabled tabs
    """
    # Get tab settings from config (with defaults)
    tab_settings = user_config.get("enabled_tabs", {})
    
    enabled_tabs = []
    for name, (module, config) in ALL_TABS.items():
        # Default to enabled if not specified
        is_enabled = tab_settings.get(config.name, config.enabled)
        if is_enabled:
            enabled_tabs.append((module, config))
    
    return enabled_tabs


def save_tab_settings(user_config: Dict[str, Any], tab_name: str, enabled: bool):
    """
    Save tab enabled/disabled setting.
    
    Args:
        user_config: User configuration dict (will be modified)
        tab_name: Tab name
        enabled: Whether tab is enabled
    """
    if "enabled_tabs" not in user_config:
        user_config["enabled_tabs"] = {}
    user_config["enabled_tabs"][tab_name] = enabled


def create_enabled_tabs(shared_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Create UI for all enabled tabs.
    
    Args:
        shared_state: Shared globals (active_emotions, user_config, helpers)
    
    Returns:
        Dict mapping tab module to component references
    """
    user_config = shared_state.get('user_config', {})
    enabled_tabs = get_enabled_tabs(user_config)
    
    tab_components = {}
    for tab_module, config in enabled_tabs:
        try:
            # Create tab UI - use get_tab_class if available
            if hasattr(tab_module, 'get_tab_class'):
                tab_class = tab_module.get_tab_class()
            else:
                # Fallback: find first Tab subclass
                tab_class = None
                for attr_name in dir(tab_module):
                    attr = getattr(tab_module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Tab) and attr is not Tab:
                        tab_class = attr
                        break
                if not tab_class:
                    raise ValueError(f"No Tab class found in module {tab_module}")
            
            components = tab_class.create_tab(shared_state)
            tab_components[config.name] = {
                'module': tab_module,
                'config': config,
                'components': components,
                'tab_class': tab_class
            }
        except Exception as e:
            print(f"Warning: Failed to create tab '{config.name}': {e}")
    
    return tab_components


def setup_tab_events(tab_components: Dict[str, Dict[str, Any]], shared_state: Dict[str, Any]):
    """
    Set up event handlers for all tabs.
    
    Args:
        tab_components: Dictionary returned by create_enabled_tabs()
        shared_state: Shared globals
    """
    for tab_name, tab_info in tab_components.items():
        try:
            tab_class = tab_info['tab_class']
            components = tab_info['components']
            
            # Setup events
            tab_class.setup_events(components, shared_state)
        except Exception as e:
            print(f"Warning: Failed to setup events for tab '{tab_name}': {e}")


# Add this to modules so it can be accessed
__all__ = [
    'ALL_TABS',
    'get_tab_registry',
    'get_enabled_tabs',
    'save_tab_settings',
    'create_enabled_tabs',
    'setup_tab_events',
]
