"""
Tools Registry Reference

Quick reference for all modularized voice tools.
This is a documentation file (not imported anywhere).
"""

# Voice Clone Tool
# Location: modules/core_components/tools/voice_clone.py
# Class: VoiceCloneTab
# Export: get_tab_class = lambda: VoiceCloneTab
from modules.core_components.tools.voice_clone import get_tab_class as get_voice_clone_tab
VoiceCloneTab = get_voice_clone_tab()

# Voice Presets Tool
# Location: modules/core_components/tools/voice_presets.py
# Class: VoicePresetsTab
# Export: get_tab_class = lambda: VoicePresetsTab
from modules.core_components.tools.voice_presets import get_tab_class as get_voice_presets_tab
VoicePresetsTab = get_voice_presets_tab()

# Conversation Tool
# Location: modules/core_components/tools/conversation.py
# Class: ConversationTab
# Export: get_tab_class = lambda: ConversationTab
from modules.core_components.tools.conversation import get_tab_class as get_conversation_tab
ConversationTab = get_conversation_tab()

# Prep Samples Tool
# Location: modules/core_components/tools/prep_samples.py
# Class: PrepSamplesTab
# Export: get_tab_class = lambda: PrepSamplesTab
from modules.core_components.tools.prep_samples import get_tab_class as get_prep_samples_tab
PrepSamplesTab = get_prep_samples_tab()

# Finetune Dataset Tool
# Location: modules/core_components/tools/finetune_dataset.py
# Class: FinetuneDatasetTab
# Export: get_tab_class = lambda: FinetuneDatasetTab
from modules.core_components.tools.finetune_dataset import get_tab_class as get_finetune_dataset_tab
FinetuneDatasetTab = get_finetune_dataset_tab()

# Train Model Tab
# Location: modules/tabs/tab_train_model.py
# Class: TrainModelTab
# Export: get_tab_class = lambda: TrainModelTab
# Lines: 6078-6323
from modules.core_components.tools.tab_train_model import get_tab_class as get_train_model_tab
TrainModelTab = get_train_model_tab()

# Settings Tab
# Location: modules/tabs/tab_settings.py
# Class: SettingsTab
# Export: get_tab_class = lambda: SettingsTab
# Lines: 6368-end
from modules.core_components.tools.tab_settings import get_tab_class as get_settings_tab
SettingsTab = get_settings_tab()


# Example Usage
if __name__ == "__main__":
    """
    Example of how to use the tab classes in the main application.
    """
    
    # Import all tab classes
    from modules.core_components.tools.tab_voice_clone import get_tab_class as get_voice_clone_tab
    from modules.core_components.tools.tab_voice_presets import get_tab_class as get_voice_presets_tab
    from modules.core_components.tools.tab_conversation import get_tab_class as get_conversation_tab
    from modules.core_components.tools.tab_prep_samples import get_tab_class as get_prep_samples_tab
    from modules.core_components.tools.tab_finetune_dataset import get_tab_class as get_finetune_dataset_tab
    from modules.core_components.tools.tab_train_model import get_tab_class as get_train_model_tab
    from modules.core_components.tools.tab_settings import get_tab_class as get_settings_tab
    
    # Get tab classes
    VoiceCloneTab = get_voice_clone_tab()
    VoicePresetsTab = get_voice_presets_tab()
    ConversationTab = get_conversation_tab()
    PrepSamplesTab = get_prep_samples_tab()
    FinetuneDatasetTab = get_finetune_dataset_tab()
    TrainModelTab = get_train_model_tab()
    SettingsTab = get_settings_tab()
    
    # Build shared_state dictionary with all required items
    shared_state = {
        # ... all required functions and config from voice_clone_studio.py
    }
    
    # Create tabs
    voice_clone_components = VoiceCloneTab.create_tab(shared_state)
    voice_presets_components = VoicePresetsTab.create_tab(shared_state)
    conversation_components = ConversationTab.create_tab(shared_state)
    prep_samples_components = PrepSamplesTab.create_tab(shared_state)
    finetune_components = FinetuneDatasetTab.create_tab(shared_state)
    train_components = TrainModelTab.create_tab(shared_state)
    settings_components = SettingsTab.create_tab(shared_state)
    
    # Setup events
    VoiceCloneTab.setup_events(voice_clone_components, shared_state)
    VoicePresetsTab.setup_events(voice_presets_components, shared_state)
    ConversationTab.setup_events(conversation_components, shared_state)
    PrepSamplesTab.setup_events(prep_samples_components, shared_state)
    FinetuneDatasetTab.setup_events(finetune_components, shared_state)
    TrainModelTab.setup_events(train_components, shared_state)
    SettingsTab.setup_events(settings_components, shared_state)
