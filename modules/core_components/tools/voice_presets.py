"""
Voice Presets Tab

Use Qwen3-TTS pre-trained models or custom trained models with style control.
"""

# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
import soundfile as sf
import torch
import random
from datetime import datetime
from textwrap import dedent
from pathlib import Path

from modules.core_components.tools.base import Tab, TabConfig
from modules.core_components.tool_utils import format_help_html
from modules.core_components.ai_models.tts_manager import get_tts_manager


class VoicePresetsTab(Tab):
    """Voice Presets tab implementation."""

    config = TabConfig(
        name="Voice Presets",
        module_name="tab_voice_presets",
        description="Generate with preset voices or trained models",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tab(cls, shared_state):
        """Create Voice Presets tab UI."""
        components = {}

        # Get helper functions and config
        get_trained_models = shared_state['get_trained_models']
        create_qwen_advanced_params = shared_state['create_qwen_advanced_params']
        _user_config = shared_state['_user_config']
        _active_emotions = shared_state['_active_emotions']
        CUSTOM_VOICE_SPEAKERS = shared_state['CUSTOM_VOICE_SPEAKERS']
        MODEL_SIZES_CUSTOM = shared_state['MODEL_SIZES_CUSTOM']
        LANGUAGES = shared_state['LANGUAGES']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        generate_custom_voice = shared_state['generate_custom_voice']
        generate_with_trained_model = shared_state['generate_with_trained_model']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']

        with gr.TabItem("Voice Presets") as voice_presets_tab:
            components['voice_presets_tab'] = voice_presets_tab
            gr.Markdown("Use Qwen3-TTS pre-trained models or Custom Trained models with style control")

            with gr.Row():
                # Left - Speaker selection
                with gr.Column(scale=1):
                    gr.Markdown("### Select Voice Type")

                    components['voice_type_radio'] = gr.Radio(
                        choices=["Premium Speakers", "Trained Models"],
                        value="Premium Speakers",
                        label="Voice Source"
                    )

                    # Premium speakers dropdown
                    components['premium_section'] = gr.Column(visible=True)
                    with components['premium_section']:
                        speaker_choices = CUSTOM_VOICE_SPEAKERS
                        components['custom_speaker_dropdown'] = gr.Dropdown(
                            choices=speaker_choices,
                            label="Speaker",
                            info="Choose a premium voice"
                        )

                        components['custom_model_size'] = gr.Dropdown(
                            choices=MODEL_SIZES_CUSTOM,
                            value=_user_config.get("custom_voice_size", "Large"),
                            label="Model",
                            info="Small = faster, Large = better quality",
                            scale=1
                        )

                        premium_speaker_guide = dedent("""\
                            **Premium Speakers:**

                            | Speaker | Voice | Language |
                            |---------|-------|----------|
                            | Vivian | Bright young female    | 🇨🇳 Chinese |
                            | Serena | Warm gentle female    | 🇨🇳 Chinese |
                            | Uncle_Fu | Seasoned mellow male    | 🇨🇳 Chinese |
                            | Dylan | Youthful Beijing male    | 🇨🇳 Chinese |
                            | Eric | Lively Chengdu male    | 🇨🇳 Chinese |
                            | Ryan | Dynamic male | 🇺🇸 English    |
                            | Aiden | Sunny American male    | 🇺🇸 English |
                            | Ono_Anna | Playful female    | 🇯🇵 Japanese |
                            | Sohee | Warm female    | 🇰🇷 Korean |

                            *Each speaker works best in native language.*
                            """)

                        gr.HTML(
                            value=format_help_html(premium_speaker_guide, height="auto"),
                            container=True,
                            padding=True
                        )
                    # Trained models dropdown
                    components['trained_section'] = gr.Column(visible=False)
                    with components['trained_section']:
                        def get_initial_model_list():
                            """Get initial list of trained models for dropdown initialization."""
                            models = get_trained_models()
                            if not models:
                                return ["(No trained models found)"]
                            return ["(Select Model)"] + [m['display_name'] for m in models]

                        def refresh_trained_models():
                            """Refresh model list."""
                            models = get_trained_models()
                            if not models:
                                return gr.update(choices=["(No trained models found)"], value="(No trained models found)")
                            choices = ["(Select Model)"] + [m['display_name'] for m in models]
                            return gr.update(choices=choices, value="(Select Model)")

                        initial_choices = get_initial_model_list()
                        initial_value = initial_choices[0]

                        components['trained_model_dropdown'] = gr.Dropdown(
                            choices=initial_choices,
                            value=initial_value,
                            label="Trained Model",
                            info="Select your custom trained voice"
                        )

                        components['refresh_trained_btn'] = gr.Button("Refresh", size="sm")

                        trained_models_tip = dedent("""\
                        **Trained Models:**

                        Custom voices you've trained in the Train Model tab.
                        Models are listed as:
                        - "ModelName" for standalone models
                        - "ModelName - Epoch N" for checkpoint-based models

                        *Tip: Later epochs are usually better trained*
                        """)
                        gr.HTML(
                            value=format_help_html(trained_models_tip, height="auto"),
                            container=True,
                            padding=True,
                        )

                # Right - Generation
                with gr.Column(scale=3):
                    gr.Markdown("### Generate Speech")

                    components['custom_text_input'] = gr.Textbox(
                        label="Text to Generate",
                        placeholder="Enter the text you want spoken...",
                        lines=6
                    )

                    components['custom_instruct_input'] = gr.Textbox(
                        label="Style Instructions (Optional)",
                        placeholder="e.g., 'Speak with excitement' or 'Very sad and slow' or '用愤怒的语气说'",
                        lines=2,
                        info="Control emotion, tone, speed, etc."
                    )

                    with gr.Row():
                        components['custom_language'] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                            scale=2
                        )
                        components['custom_seed'] = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    # Qwen Advanced Parameters (visible for both modes)
                    custom_params = create_qwen_advanced_params(
                        emotions_dict=_active_emotions,
                        include_emotion=True,
                        initial_emotion=None,
                        initial_intensity=1.0,
                        visible=True
                    )

                    # Store emotion row references with correct names for visibility toggling
                    components['custom_emotion_row'] = custom_params['emotion_row'] if 'emotion_row' in custom_params else None
                    components['custom_emotion_buttons_row'] = custom_params['emotion_buttons_row'] if 'emotion_buttons_row' in custom_params else None

                    # Create alias references for backward compatibility
                    components['custom_emotion_preset'] = custom_params['emotion_preset']
                    components['custom_emotion_intensity'] = custom_params['emotion_intensity']
                    components['custom_save_emotion_btn'] = custom_params.get('save_emotion_btn')
                    components['custom_delete_emotion_btn'] = custom_params.get('delete_emotion_btn')
                    components['custom_emotion_save_name'] = custom_params.get('emotion_save_name')
                    components['custom_do_sample'] = custom_params['do_sample']
                    components['custom_temperature'] = custom_params['temperature']
                    components['custom_top_k'] = custom_params['top_k']
                    components['custom_top_p'] = custom_params['top_p']
                    components['custom_repetition_penalty'] = custom_params['repetition_penalty']
                    components['custom_max_new_tokens'] = custom_params['max_new_tokens']
                    components['custom_params'] = custom_params

                    components['custom_generate_btn'] = gr.Button("Generate Audio", variant="primary", size="lg")

                    components['custom_output_audio'] = gr.Audio(
                        label="Generated Audio",
                        type="filepath"
                    )
                    components['preset_status'] = gr.Textbox(label="Status", max_lines=5, interactive=False)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Presets tab events."""

        # Get helper functions and directories
        get_trained_models = shared_state['get_trained_models']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        OUTPUT_DIR = shared_state['OUTPUT_DIR']
        play_completion_beep = shared_state.get('play_completion_beep')
        user_config = shared_state.get('_user_config', {})

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()

        custom_params = components['custom_params']

        def generate_custom_voice_handler(text_to_generate, language, speaker, instruct, seed, model_size="1.7B",
                                          do_sample=True, temperature=0.9, top_k=50, top_p=1.0,
                                          repetition_penalty=1.05, max_new_tokens=2048, progress=gr.Progress()):
            """Generate audio using the CustomVoice model with premium speakers."""
            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate."

            if not speaker:
                return None, "❌ Please select a speaker."

            try:
                progress(0.1, desc=f"Loading CustomVoice model ({model_size})...")
                
                audio_data, sr = tts_manager.generate_custom_voice(
                    text=text_to_generate,
                    language=language,
                    speaker=speaker,
                    instruct=instruct,
                    model_size=model_size,
                    seed=int(seed) if seed is not None else -1,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens
                )

                progress(0.8, desc="Saving audio...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = OUTPUT_DIR / f"custom_{speaker}_{timestamp}.wav"

                sf.write(str(output_file), audio_data, sr)

                # Save metadata file
                metadata_file = output_file.with_suffix(".txt")
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: Custom Voice
                    Model: CustomVoice {model_size}
                    Speaker: {speaker}
                    Language: {language}
                    Seed: {seed}
                    Instruct: {instruct.strip() if instruct else ''}
                    Text: {text_to_generate.strip()}
                    """)
                metadata_file.write_text(metadata, encoding="utf-8")

                progress(1.0, desc="Done!")
                instruct_msg = f" with style: {instruct.strip()[:30]}..." if instruct and instruct.strip() else ""
                if play_completion_beep:
                    play_completion_beep()
                return str(output_file), f"Audio saved to: {output_file.name}\nSpeaker: {speaker}{instruct_msg}\nSeed: {seed} | {model_size}"

            except Exception as e:
                return None, f"❌ Error generating audio: {str(e)}"

        def generate_with_trained_model_handler(text_to_generate, language, speaker_name, checkpoint_path, instruct, seed,
                                                do_sample=True, temperature=0.9, top_k=50, top_p=1.0,
                                                repetition_penalty=1.05, max_new_tokens=2048, progress=gr.Progress()):
            """Generate audio using a trained custom voice model checkpoint."""
            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate."

            try:
                progress(0.1, desc=f"Loading trained model from {checkpoint_path}...")

                # Call tts_manager method
                audio_data, sr = tts_manager.generate_with_trained_model(
                    text=text_to_generate,
                    language=language,
                    speaker_name=speaker_name,
                    checkpoint_path=checkpoint_path,
                    instruct=instruct,
                    seed=int(seed) if seed is not None else -1,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    user_config=user_config
                )

                progress(0.8, desc="Saving audio...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = OUTPUT_DIR / f"trained_{speaker_name}_{timestamp}.wav"

                sf.write(str(output_file), audio_data, sr)

                # Save metadata file
                metadata_file = output_file.with_suffix(".txt")
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: Trained Model
                    Model: {checkpoint_path}
                    Speaker: {speaker_name}
                    Language: {language}
                    Seed: {seed}
                    Instruct: {instruct.strip() if instruct else ''}
                    Text: {text_to_generate.strip()}
                    """)
                metadata_file.write_text(metadata, encoding="utf-8")

                progress(1.0, desc="Done!")
                instruct_msg = f" with style: {instruct.strip()[:30]}..." if instruct and instruct.strip() else ""
                if play_completion_beep:
                    play_completion_beep()
                return str(output_file), f"Audio saved: {output_file.name}\nSpeaker: {speaker_name}{instruct_msg}\nSeed: {seed} | Trained Model"

            except Exception as e:
                return None, f"❌ Error generating audio: {str(e)}"

        def extract_speaker_name(selection):
            """Extract speaker name from dropdown selection."""
            if not selection:
                return None
            return selection.split(" - ")[0].split(" (")[0]

        def toggle_voice_type(voice_type):
            """Toggle between premium and trained model sections."""
            is_premium = voice_type == "Premium Speakers"

            if is_premium:
                # Return in order: premium_section, trained_section, instruct_input, emotion_row, emotion_buttons_row,
                # emotion_preset, emotion_intensity, temperature, top_p, repetition_penalty
                return (
                    gr.update(visible=True),   # premium_section
                    gr.update(visible=False),  # trained_section
                    gr.update(visible=True),   # instruct_input
                    gr.update(visible=False),  # emotion_row
                    gr.update(visible=False),  # emotion_buttons_row
                    gr.update(value=None),     # emotion_preset
                    gr.update(value=1.0),      # emotion_intensity
                    gr.update(value=0.9),      # temperature
                    gr.update(value=1.0),      # top_p
                    gr.update(value=1.05)      # repetition_penalty
                )
            else:
                return (
                    gr.update(visible=False),  # premium_section
                    gr.update(visible=True),   # trained_section
                    gr.update(visible=False),  # instruct_input
                    gr.update(visible=True),   # emotion_row
                    gr.update(visible=True),   # emotion_buttons_row
                    gr.update(),               # emotion_preset
                    gr.update(),               # emotion_intensity
                    gr.update(),               # temperature
                    gr.update(),               # top_p
                    gr.update()                # repetition_penalty
                )

        def generate_with_voice_type(text, lang, speaker_sel, instruct, seed, model_size, voice_type, premium_speaker, trained_model,
                                     do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens, progress=gr.Progress()):
            """Generate audio with either premium or trained voice."""

            if voice_type == "Premium Speakers":
                speaker = extract_speaker_name(premium_speaker)
                if not speaker:
                    return None, "❌ Please select a premium speaker"

                return generate_custom_voice_handler(
                    text, lang, speaker, instruct, seed,
                    "1.7B" if model_size == "Large" else "0.6B",
                    do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens,
                    progress
                )
            else:
                if not trained_model or trained_model in ["(No trained models found)", "(Select Model)"]:
                    return None, "❌ Please select a trained model or train one first"

                models = get_trained_models()
                model_path = None
                speaker_name = None
                for model in models:
                    if model['display_name'] == trained_model:
                        model_path = model['path']
                        speaker_name = model['speaker_name']
                        break

                if not model_path:
                    return None, f"❌ Model not found: {trained_model}"

                return generate_with_trained_model_handler(
                    text, lang, speaker_name, model_path, instruct, seed,
                    do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens,
                    progress
                )

        # Only wire events for components that exist (not None)
        if components.get('voice_type_radio') is not None:
            outputs = [
                components['premium_section'], components['trained_section'],
                components['custom_instruct_input'],
                components.get('custom_emotion_row'),
                components.get('custom_emotion_buttons_row'),
                components['custom_emotion_preset'], components['custom_emotion_intensity'],
                components['custom_temperature'], components['custom_top_p'], components['custom_repetition_penalty']
            ]
            # Filter out None values
            outputs = [o for o in outputs if o is not None]

            components['voice_type_radio'].change(
                toggle_voice_type,
                inputs=[components['voice_type_radio']],
                outputs=outputs
            )

        components['refresh_trained_btn'].click(
            lambda: (
                gr.update(choices=["(Select Model)"] + [m['display_name'] for m in get_trained_models()] if get_trained_models() else ["(No trained models found)"])
            ),
            outputs=[components['trained_model_dropdown']]
        )

        # Apply emotion preset to Custom Voice parameters
        if 'update_from_emotion' in components.get('custom_params', {}):
            components['custom_emotion_preset'].change(
                components['custom_params']['update_from_emotion'],
                inputs=[components['custom_emotion_preset'], components['custom_emotion_intensity']],
                outputs=[components['custom_temperature'], components['custom_top_p'], components['custom_repetition_penalty']]
            )

            components['custom_emotion_intensity'].change(
                components['custom_params']['update_from_emotion'],
                inputs=[components['custom_emotion_preset'], components['custom_emotion_intensity']],
                outputs=[components['custom_temperature'], components['custom_top_p'], components['custom_repetition_penalty']]
            )

        # Emotion management buttons
        components['custom_save_emotion_btn'].click(
            fn=None,
            inputs=[components['custom_emotion_preset']],
            outputs=None,
            js=show_input_modal_js(
                title="Save Emotion Preset",
                message="Enter a name for this emotion preset:",
                placeholder="e.g., Happy, Sad, Excited",
                context="custom_emotion_"
            )
        )

        def handle_custom_emotion_input(input_value, intensity, temp, rep_pen, top_p):
            """Process input modal submission for Voice Presets emotion save."""
            if not input_value or not input_value.startswith("custom_emotion_"):
                return gr.update(), gr.update()

            parts = input_value.split("_")
            if len(parts) >= 3:
                if parts[2] == "cancel":
                    return gr.update(), ""
                emotion_name = "_".join(parts[2:-1])
                return save_emotion_handler(emotion_name, intensity, temp, rep_pen, top_p)

            return gr.update(), gr.update()

        components['custom_delete_emotion_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Emotion Preset?",
                message="This will permanently delete this emotion preset from your configuration.",
                confirm_button_text="Delete",
                context="custom_emotion_"
            )
        )

        components['custom_generate_btn'].click(
            generate_with_voice_type,
            inputs=[
                components['custom_text_input'], components['custom_language'], components['custom_speaker_dropdown'],
                components['custom_instruct_input'], components['custom_seed'], components['custom_model_size'],
                components['voice_type_radio'], components['custom_speaker_dropdown'], components['trained_model_dropdown'],
                components['custom_do_sample'], components['custom_temperature'], components['custom_top_k'], components['custom_top_p'],
                components['custom_repetition_penalty'], components['custom_max_new_tokens']
            ],
            outputs=[components['custom_output_audio'], components['preset_status']]
        )

        def delete_custom_emotion_wrapper(confirm_value, emotion_name):
            """Only process if context matches custom_emotion_."""
            if not confirm_value or not confirm_value.startswith("custom_emotion_"):
                return gr.update(), gr.update()
            dropdown_update, status_msg, clear_trigger = delete_emotion_handler(confirm_value, emotion_name)
            return dropdown_update, status_msg

        confirm_trigger.change(
            delete_custom_emotion_wrapper,
            inputs=[confirm_trigger, components['custom_emotion_preset']],
            outputs=[components['custom_emotion_preset'], components['preset_status']]
        )

        input_trigger.change(
            handle_custom_emotion_input,
            inputs=[input_trigger, components['custom_emotion_intensity'], components['custom_temperature'], components['custom_repetition_penalty'], components['custom_top_p']],
            outputs=[components['custom_emotion_preset'], components['preset_status']]
        )

        # Refresh emotion dropdowns when tab is selected
        components['voice_presets_tab'].select(
            lambda: gr.update(choices=shared_state['get_emotion_choices'](shared_state['_active_emotions'])),
            outputs=[components['custom_emotion_preset']]
        )

        # Save preferences when settings change
        components['custom_model_size'].change(
            lambda x: save_preference("custom_voice_size", x),
            inputs=[components['custom_model_size']],
            outputs=[]
        )

        components['custom_language'].change(
            lambda x: save_preference("language", x),
            inputs=[components['custom_language']],
            outputs=[]
        )

# Export for tab registry
get_tab_class = lambda: VoicePresetsTab

if __name__ == "__main__":
    """Standalone testing of Voice Presets tool."""
    print("[*] Starting Voice Presets Tool - Standalone Mode")

    from pathlib import Path
    import sys

    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    from modules.core_components import (
        CORE_EMOTIONS,
        CONFIRMATION_MODAL_HTML,
        CONFIRMATION_MODAL_CSS,
        INPUT_MODAL_HTML,
        INPUT_MODAL_CSS,
        show_confirmation_modal_js,
        show_input_modal_js,
        handle_save_emotion,
        handle_delete_emotion
    )
    from modules.core_components.ui_components import create_qwen_advanced_params
    from modules.core_components.constants import (
        LANGUAGES,
        CUSTOM_VOICE_SPEAKERS,
        MODEL_SIZES_CUSTOM,
        QWEN_GENERATION_DEFAULTS
    )
    from modules.core_components.tool_utils import (
        load_config,
        save_preference as save_pref_to_file,
        format_help_html,
        TRIGGER_HIDE_CSS
    )

    # Load config with persistent settings
    user_config = load_config()
    active_emotions = user_config.get('emotions', CORE_EMOTIONS)

    # Simple mock for create_qwen_advanced_params if not available
    if not hasattr(create_qwen_advanced_params, '__call__'):
        def create_qwen_advanced_params(*args, **kwargs):
            return {}

    OUTPUT_DIR = project_root / "output"
    MODELS_DIR = project_root / "models"
    OUTPUT_DIR.mkdir(exist_ok=True)

    def get_trained_models():
        """Find trained model checkpoints."""
        models = []
        if MODELS_DIR.exists():
            for folder in MODELS_DIR.iterdir():
                if folder.is_dir():
                    for checkpoint in folder.glob("checkpoint-*"):
                        if checkpoint.is_dir():
                            models.append({
                                'display_name': f"{folder.name} - {checkpoint.name}",
                                'path': str(checkpoint),
                                'speaker_name': folder.name
                            })
        return models

    # Stub handlers for standalone testing (actual implementations in handlers section)
    def generate_custom_voice(*args, **kwargs):
        return None, "⚠️ Handler not yet connected in standalone mode"

    def generate_with_trained_model(*args, **kwargs):
        return None, "⚠️ Handler not yet connected in standalone mode"

    shared_state = {
        'get_trained_models': get_trained_models,
        'create_qwen_advanced_params': create_qwen_advanced_params,
        '_user_config': user_config,
        '_active_emotions': active_emotions,
        'CUSTOM_VOICE_SPEAKERS': CUSTOM_VOICE_SPEAKERS,
        'MODEL_SIZES_CUSTOM': MODEL_SIZES_CUSTOM,
        'LANGUAGES': LANGUAGES,
        'OUTPUT_DIR': OUTPUT_DIR,
        'show_input_modal_js': show_input_modal_js,
        'show_confirmation_modal_js': show_confirmation_modal_js,
        'generate_custom_voice': generate_custom_voice,
        'generate_with_trained_model': generate_with_trained_model,
        'save_emotion_handler': lambda name, intensity, temp, rep_pen, top_p: handle_save_emotion(name, intensity, temp, rep_pen, top_p, user_config, active_emotions),
        'delete_emotion_handler': lambda confirm_val, emotion_name: handle_delete_emotion(confirm_val, emotion_name, user_config, active_emotions),
        'save_preference': lambda k, v: save_pref_to_file(user_config, k, v),
        'play_completion_beep': lambda: print("[Beep] Complete!"),
        'get_emotion_choices': lambda emotions: sorted(emotions.keys(), key=str.lower),
        'confirm_trigger': None,
        'input_trigger': None}

    # Load custom theme
    theme = gr.themes.Base.load('modules/core_components/theme.json')

    print(f"[*] Output: {OUTPUT_DIR}")
    print(f"[*] Found {len(get_trained_models())} trained models")

    with gr.Blocks(title="Voice Presets - Standalone", head=CONFIRMATION_MODAL_CSS + INPUT_MODAL_CSS, css=TRIGGER_HIDE_CSS) as app:
        # Add modal HTML
        gr.HTML(CONFIRMATION_MODAL_HTML)
        gr.HTML(INPUT_MODAL_HTML)
        
        gr.Markdown("# 🎙️ Voice Presets Tool (Standalone Testing)")
        gr.Markdown("*Standalone mode with full modal support*")

        # Hidden trigger widgets - visible but hidden via CSS
        with gr.Row():
            confirm_trigger = gr.Textbox(label="Confirm Trigger", value="", elem_id="confirm-trigger")
            input_trigger = gr.Textbox(label="Input Trigger", value="", elem_id="input-trigger")
        shared_state['confirm_trigger'] = confirm_trigger
        shared_state['input_trigger'] = input_trigger

        components = VoicePresetsTab.create_tab(shared_state)
        VoicePresetsTab.setup_events(components, shared_state)

    print("\n✓ Voice Presets UI loaded successfully!")
    print("   Note: Generate buttons show stub messages - handlers pending implementation")
    print("[*] Launching on http://127.0.0.1:7863")
    app.launch(theme=theme, server_port=7863, server_name="127.0.0.1", share=False, inbrowser=False)
