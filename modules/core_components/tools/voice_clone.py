"""
Voice Clone Tab

Clone voices from samples using Qwen3-TTS or VibeVoice.
"""

import gradio as gr
from modules.core_components.tools.base import Tab, TabConfig


class VoiceCloneTab(Tab):
    """Voice Clone tab implementation."""
    
    config = TabConfig(
        name="Voice Clone",
        module_name="tab_voice_clone",
        description="Clone voices from voice samples",
        enabled=True,
        category="generation"
    )
    
    @classmethod
    def create_tab(cls, shared_state):
        """Create Voice Clone tab UI."""
        components = {}
        
        # Get helper functions and config
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        get_emotion_choices = shared_state['get_emotion_choices']
        apply_emotion_preset = shared_state['apply_emotion_preset']
        get_prompt_cache_path = shared_state['get_prompt_cache_path']
        LANGUAGES = shared_state['LANGUAGES']
        VOICE_CLONE_OPTIONS = shared_state['VOICE_CLONE_OPTIONS']
        DEFAULT_VOICE_CLONE_MODEL = shared_state['DEFAULT_VOICE_CLONE_MODEL']
        _user_config = shared_state['_user_config']
        _active_emotions = shared_state['_active_emotions']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        generate_audio = shared_state['generate_audio']
        refresh_samples = shared_state['refresh_samples']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        
        import soundfile as sf
        
        with gr.TabItem("Voice Clone") as voice_clone_tab:
            components['voice_clone_tab'] = voice_clone_tab
            gr.Markdown("Clone Voices from Samples, using Qwen3-TTS or VibeVoice")
            with gr.Row():
                # Left column - Sample selection (1/3 width)
                with gr.Column(scale=1):
                    gr.Markdown("### Voice Sample")

                    sample_choices = get_sample_choices()
                    components['sample_dropdown'] = gr.Dropdown(
                        choices=sample_choices,
                        value=sample_choices[0] if sample_choices else None,
                        label="Select Sample",
                        info="Manage samples in Prep Samples tab"
                    )

                    with gr.Row():
                        components['load_sample_btn'] = gr.Button("Load", size="sm")
                        components['refresh_samples_btn'] = gr.Button("Refresh", size="sm")

                    components['sample_audio'] = gr.Audio(
                        label="Sample Preview",
                        type="filepath",
                        interactive=False,
                        visible=True
                    )

                    components['sample_text'] = gr.Textbox(
                        label="Sample Text",
                        interactive=False,
                        max_lines=10
                    )

                    components['sample_info'] = gr.Textbox(
                        label="Info",
                        interactive=False,
                        max_lines=3
                    )

                # Right column - Generation (2/3 width)
                with gr.Column(scale=3):
                    gr.Markdown("### Generate Speech")

                    components['text_input'] = gr.Textbox(
                        label="Text to Generate",
                        placeholder="Enter the text you want to speak in the cloned voice...",
                        lines=6
                    )

                    # Language dropdown (hidden for VibeVoice models)
                    is_qwen_initial = "Qwen" in _user_config.get("voice_clone_model", DEFAULT_VOICE_CLONE_MODEL)
                    components['language_row'] = gr.Row(visible=is_qwen_initial)
                    with components['language_row']:
                        components['language_dropdown'] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                        )

                    with gr.Row():
                        components['clone_model_dropdown'] = gr.Dropdown(
                            choices=VOICE_CLONE_OPTIONS,
                            value=_user_config.get("voice_clone_model", DEFAULT_VOICE_CLONE_MODEL),
                            label="Engine & Model (Qwen3 or VibeVoice)",
                            scale=4
                        )
                        components['seed_input'] = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    # Qwen3 Advanced Parameters
                    is_qwen_initial = "Qwen" in _user_config.get("voice_clone_model", DEFAULT_VOICE_CLONE_MODEL)
                    components['qwen_params_accordion'] = gr.Accordion("Qwen3 Advanced Parameters", open=False, visible=is_qwen_initial)
                    with components['qwen_params_accordion']:

                        # Emotion preset dropdown
                        emotion_choices = get_emotion_choices(_active_emotions)
                        with gr.Row():
                            components['qwen_emotion_preset'] = gr.Dropdown(
                                choices=emotion_choices,
                                value=None,
                                label="🎭 Emotion Preset",
                                info="Quick presets that adjust parameters for different emotions",
                                scale=3
                            )
                            components['qwen_emotion_intensity'] = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Intensity",
                                info="Emotion strength (0=none, 2=extreme)",
                                scale=1
                            )

                        # Emotion management buttons
                        with gr.Row():
                            components['qwen_save_emotion_btn'] = gr.Button("Save", size="sm", scale=1)
                            components['qwen_delete_emotion_btn'] = gr.Button("Delete", size="sm", scale=1)
                        components['qwen_emotion_save_name'] = gr.Textbox(visible=False, value="")

                        with gr.Row():
                            components['qwen_do_sample'] = gr.Checkbox(
                                label="Enable Sampling",
                                value=True,
                                info="Qwen3 recommends sampling enabled (default: True)"
                            )
                            components['qwen_temperature'] = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.9,
                                step=0.05,
                                label="Temperature",
                                info="Sampling temperature"
                            )

                        with gr.Row():
                            components['qwen_top_k'] = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=1,
                                label="Top-K",
                                info="Keep only top K tokens"
                            )
                            components['qwen_top_p'] = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=1.0,
                                step=0.05,
                                label="Top-P (Nucleus)",
                                info="Cumulative probability threshold"
                            )

                        with gr.Row():
                            components['qwen_repetition_penalty'] = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                value=1.05,
                                step=0.05,
                                label="Repetition Penalty",
                                info="Penalize repeated tokens"
                            )
                            components['qwen_max_new_tokens'] = gr.Slider(
                                minimum=512,
                                maximum=4096,
                                value=2048,
                                step=256,
                                label="Max New Tokens",
                                info="Maximum codec tokens to generate"
                            )

                    # VibeVoice Advanced Parameters
                    components['vv_params_accordion'] = gr.Accordion("VibeVoice Advanced Parameters", open=False, visible=not is_qwen_initial)
                    with components['vv_params_accordion']:

                        with gr.Row():
                            components['vv_cfg_scale'] = gr.Slider(
                                minimum=1.0,
                                maximum=5.0,
                                value=3.0,
                                step=0.1,
                                label="CFG Scale",
                                info="Controls audio adherence to voice prompt"
                            )
                            components['vv_num_steps'] = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=20,
                                step=1,
                                label="Inference Steps",
                                info="Number of diffusion steps"
                            )

                        gr.Markdown("Stochastic Sampling Parameters")
                        with gr.Row():
                            components['vv_do_sample'] = gr.Checkbox(
                                label="Enable Sampling",
                                value=False,
                                info="Enable stochastic sampling (default: False)"
                            )
                        with gr.Row():
                            components['vv_repetition_penalty'] = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                label="Repetition Penalty",
                                info="Penalize repeated tokens"
                            )

                            components['vv_temperature'] = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                label="Temperature",
                                info="Sampling temperature"
                            )

                        with gr.Row():
                            components['vv_top_k'] = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=1,
                                label="Top-K",
                                info="Keep only top K tokens"
                            )
                            components['vv_top_p'] = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=1.0,
                                step=0.05,
                                label="Top-P (Nucleus)",
                                info="Cumulative probability threshold"
                            )

                    components['generate_btn'] = gr.Button("Generate Audio", variant="primary", size="lg")

                    components['output_audio'] = gr.Audio(
                        label="Generated Audio",
                        type="filepath"
                    )

                    components['clone_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)

        return components
    
    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Clone tab events."""
        
        # Get helper functions
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        get_prompt_cache_path = shared_state['get_prompt_cache_path']
        apply_emotion_preset = shared_state['apply_emotion_preset']
        generate_audio = shared_state['generate_audio']
        refresh_samples = shared_state['refresh_samples']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        
        import soundfile as sf
        
        def load_selected_sample(sample_name):
            """Load audio, text, and info for the selected sample."""
            if not sample_name:
                return None, "", ""
            samples = get_available_samples()
            for s in samples:
                if s["name"] == sample_name:
                    # Check cache status for both model sizes
                    cache_small = get_prompt_cache_path(sample_name, "0.6B").exists()
                    cache_large = get_prompt_cache_path(sample_name, "1.7B").exists()

                    if cache_small and cache_large:
                        cache_status = "Qwen Cache: ⚡ Small, Large"
                    elif cache_small:
                        cache_status = "Qwen Cache: ⚡ Small"
                    elif cache_large:
                        cache_status = "Qwen Cache: ⚡ Large"
                    else:
                        cache_status = "Qwen Cache: 📦 Not cached"

                    try:
                        audio_data, sr = sf.read(s["wav_path"])
                        duration = len(audio_data) / sr
                        info = f"**Info**\n\nDuration: {duration:.2f}s | {cache_status}"
                    except:
                        info = f"**Info**\n\n{cache_status}"

                    # Add design instructions if this was a Voice Design sample
                    meta = s.get("meta", {})
                    if meta.get("Type") == "Voice Design" and meta.get("Instruct"):
                        info += f"\n\n**Voice Design:**\n{meta['Instruct']}"

                    return s["wav_path"], s["ref_text"], info
            return None, "", ""

        # Connect event handlers for Voice Clone tab
        components['sample_dropdown'].change(
            load_selected_sample,
            inputs=[components['sample_dropdown']],
            outputs=[components['sample_audio'], components['sample_text'], components['sample_info']]
        )

        components['load_sample_btn'].click(
            load_selected_sample,
            inputs=[components['sample_dropdown']],
            outputs=[components['sample_audio'], components['sample_text'], components['sample_info']]
        )

        components['refresh_samples_btn'].click(
            refresh_samples,
            outputs=[components['sample_dropdown']]
        )

        components['generate_btn'].click(
            generate_audio,
            inputs=[components['sample_dropdown'], components['text_input'], components['language_dropdown'], components['seed_input'], components['clone_model_dropdown'],
                    components['qwen_do_sample'], components['qwen_temperature'], components['qwen_top_k'], components['qwen_top_p'], components['qwen_repetition_penalty'],
                    components['qwen_max_new_tokens'],
                    components['vv_do_sample'], components['vv_temperature'], components['vv_top_k'], components['vv_top_p'], components['vv_repetition_penalty'],
                    components['vv_cfg_scale'], components['vv_num_steps']],
            outputs=[components['output_audio'], components['clone_status']]
        )

        # Toggle language visibility based on model selection
        def toggle_language_visibility(model_selection):
            is_qwen = "Qwen" in model_selection
            return gr.update(visible=is_qwen)

        components['clone_model_dropdown'].change(
            toggle_language_visibility,
            inputs=[components['clone_model_dropdown']],
            outputs=[components['language_row']]
        )

        # Toggle accordion visibility based on engine
        def toggle_engine_params(model_selection):
            is_qwen = "Qwen" in model_selection
            return gr.update(visible=is_qwen), gr.update(visible=not is_qwen)

        components['clone_model_dropdown'].change(
            toggle_engine_params,
            inputs=[components['clone_model_dropdown']],
            outputs=[components['qwen_params_accordion'], components['vv_params_accordion']]
        )

        # Apply emotion preset to Qwen parameters
        # Update when emotion changes
        components['qwen_emotion_preset'].change(
            apply_emotion_preset,
            inputs=[components['qwen_emotion_preset'], components['qwen_emotion_intensity']],
            outputs=[components['qwen_temperature'], components['qwen_top_p'], components['qwen_repetition_penalty'], components['qwen_emotion_intensity']]
        )

        # Update when intensity changes
        components['qwen_emotion_intensity'].change(
            apply_emotion_preset,
            inputs=[components['qwen_emotion_preset'], components['qwen_emotion_intensity']],
            outputs=[components['qwen_temperature'], components['qwen_top_p'], components['qwen_repetition_penalty'], components['qwen_emotion_intensity']]
        )

        # Emotion management buttons
        components['qwen_save_emotion_btn'].click(
            fn=None,
            inputs=[components['qwen_emotion_preset']],
            outputs=None,
            js=show_input_modal_js(
                title="Save Emotion Preset",
                message="Enter a name for this emotion preset:",
                placeholder="e.g., Happy, Sad, Excited",
                context="qwen_emotion_"
            )
        )

        # Handler for when user submits from input modal
        def handle_qwen_emotion_input(input_value, intensity, temp, rep_pen, top_p):
            """Process input modal submission for Voice Clone emotion save."""
            # Context filtering: only process if this is our context
            if not input_value or not input_value.startswith("qwen_emotion_"):
                return gr.update(), gr.update()

            # Extract emotion name from context prefix
            # Remove context prefix and timestamp
            parts = input_value.split("_")
            if len(parts) >= 3:
                # Format: qwen_emotion_<name>_<timestamp> or qwen_emotion_cancel_<timestamp>
                if parts[2] == "cancel":
                    return gr.update(), ""
                # Everything between qwen_emotion_ and final timestamp
                emotion_name = "_".join(parts[2:-1])
                return save_emotion_handler(emotion_name, intensity, temp, rep_pen, top_p)

            return gr.update(), gr.update()

        components['qwen_delete_emotion_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Emotion Preset?",
                message="This will permanently delete this emotion preset from your configuration.",
                confirm_button_text="Delete",
                context="qwen_emotion_"
            )
        )

        components['clone_model_dropdown'].change(
            lambda x: save_preference("voice_clone_model", x),
            inputs=[components['clone_model_dropdown']],
            outputs=[]
        )

        # Emotion delete confirmation handler for Voice Clone tab
        def delete_qwen_emotion_wrapper(confirm_value, emotion_name):
            """Only process if context matches qwen_emotion_."""
            if not confirm_value or not confirm_value.startswith("qwen_emotion_"):
                return gr.update(), gr.update()
            # Call the delete handler with both parameters
            dropdown_update, status_msg, clear_trigger = delete_emotion_handler(confirm_value, emotion_name)
            return dropdown_update, status_msg

        confirm_trigger.change(
            delete_qwen_emotion_wrapper,
            inputs=[confirm_trigger, components['qwen_emotion_preset']],
            outputs=[components['qwen_emotion_preset'], components['clone_status']]
        )

        input_trigger.change(
            handle_qwen_emotion_input,
            inputs=[input_trigger, components['qwen_emotion_intensity'], components['qwen_temperature'], components['qwen_repetition_penalty'], components['qwen_top_p']],
            outputs=[components['qwen_emotion_preset'], components['clone_status']]
        )

        # Refresh emotion dropdowns when tab is selected
        components['voice_clone_tab'].select(
            lambda: gr.update(choices=shared_state['get_emotion_choices'](shared_state['_active_emotions'])),
            outputs=[components['qwen_emotion_preset']]
        )


# Export for tab registry
get_tab_class = lambda: VoiceCloneTab
