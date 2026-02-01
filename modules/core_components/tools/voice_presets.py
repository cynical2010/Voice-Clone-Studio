"""
Voice Presets Tab

Use Qwen3-TTS pre-trained models or custom trained models with style control.
"""

import gradio as gr
from textwrap import dedent
from modules.core_components.tools.base import Tab, TabConfig
from modules.core_components.tab_utils import format_help_html


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
                            value=format_help_html(premium_speaker_guide),
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
                            value=format_help_html(trained_models_tip),
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
                        initial_emotion="",
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
        
        # Get helper functions
        get_trained_models = shared_state['get_trained_models']
        generate_custom_voice = shared_state['generate_custom_voice']
        generate_with_trained_model = shared_state['generate_with_trained_model']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        
        custom_params = components['custom_params']
        
        def extract_speaker_name(selection):
            """Extract speaker name from dropdown selection."""
            if not selection:
                return None
            return selection.split(" - ")[0].split(" (")[0]

        def toggle_voice_type(voice_type):
            """Toggle between premium and trained model sections."""
            is_premium = voice_type == "Premium Speakers"

            if is_premium:
                return {
                    components['premium_section']: gr.update(visible=True),
                    components['trained_section']: gr.update(visible=False),
                    components['custom_instruct_input']: gr.update(visible=True),
                    components['custom_emotion_row']: gr.update(visible=False),
                    components['custom_emotion_buttons_row']: gr.update(visible=False),
                    components['custom_emotion_preset']: gr.update(value=None),
                    components['custom_emotion_intensity']: gr.update(value=1.0),
                    components['custom_temperature']: gr.update(value=0.9),
                    components['custom_top_p']: gr.update(value=1.0),
                    components['custom_repetition_penalty']: gr.update(value=1.05)
                }
            else:
                return {
                    components['premium_section']: gr.update(visible=False),
                    components['trained_section']: gr.update(visible=True),
                    components['custom_instruct_input']: gr.update(visible=False),
                    components['custom_emotion_row']: gr.update(visible=True),
                    components['custom_emotion_buttons_row']: gr.update(visible=True),
                    components['custom_emotion_preset']: gr.update(),
                    components['custom_emotion_intensity']: gr.update(),
                    components['custom_temperature']: gr.update(),
                    components['custom_top_p']: gr.update(),
                    components['custom_repetition_penalty']: gr.update()
                }

        def generate_with_voice_type(text, lang, speaker_sel, instruct, seed, model_size, voice_type, premium_speaker, trained_model,
                                     do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens, progress=gr.Progress()):
            """Generate audio with either premium or trained voice."""

            if voice_type == "Premium Speakers":
                speaker = extract_speaker_name(premium_speaker)
                if not speaker:
                    return None, "❌ Please select a premium speaker"

                return generate_custom_voice(
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

                return generate_with_trained_model(
                    text, lang, speaker_name, model_path, instruct, seed,
                    do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens,
                    progress
                )

        components['voice_type_radio'].change(
            toggle_voice_type,
            inputs=[components['voice_type_radio']],
            outputs=[
                components['premium_section'], components['trained_section'],
                components['custom_instruct_input'], components['custom_emotion_row'], components['custom_emotion_buttons_row'],
                components['custom_emotion_preset'], components['custom_emotion_intensity'],
                components['custom_temperature'], components['custom_top_p'], components['custom_repetition_penalty']
            ]
        )

        components['refresh_trained_btn'].click(
            lambda: (
                get_trained_models(),
                gr.update(choices=["(Select Model)"] + [m['display_name'] for m in get_trained_models()] if get_trained_models() else ["(No trained models found)"])
            ),
            outputs=[components['trained_model_dropdown']]
        )

        # Apply emotion preset to Custom Voice parameters
        components['custom_emotion_preset'].change(
            custom_params['update_from_emotion'],
            inputs=[components['custom_emotion_preset'], components['custom_emotion_intensity']],
            outputs=[components['custom_temperature'], components['custom_top_p'], components['custom_repetition_penalty']]
        )

        components['custom_emotion_intensity'].change(
            custom_params['update_from_emotion'],
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


# Export for tab registry
get_tab_class = lambda: VoicePresetsTab
