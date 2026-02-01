"""
Prep Samples Tab

Prepare audio samples for voice cloning.
"""

import gradio as gr
from modules.core_components.tools.base import Tab, TabConfig


class PrepSamplesTab(Tab):
    """Prep Samples tab implementation."""
    
    config = TabConfig(
        name="Prep Samples",
        module_name="tab_prep_samples",
        description="Prepare and manage voice samples",
        enabled=True,
        category="preparation"
    )
    
    @classmethod
    def create_tab(cls, shared_state):
        """Create Prep Samples tab UI."""
        components = {}
        
        # Get helper functions and config
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        get_audio_duration = shared_state['get_audio_duration']
        format_time = shared_state['format_time']
        load_existing_sample = shared_state['load_existing_sample']
        refresh_samples = shared_state['refresh_samples']
        delete_sample = shared_state['delete_sample']
        clear_sample_cache = shared_state['clear_sample_cache']
        on_prep_audio_load = shared_state['on_prep_audio_load']
        normalize_audio = shared_state['normalize_audio']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']
        transcribe_audio = shared_state['transcribe_audio']
        save_as_sample = shared_state['save_as_sample']
        _user_config = shared_state['_user_config']
        LANGUAGES = shared_state['LANGUAGES']
        WHISPER_AVAILABLE = shared_state['WHISPER_AVAILABLE']
        DEEPFILTER_AVAILABLE = shared_state['DEEPFILTER_AVAILABLE']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        
        with gr.TabItem("Prep Samples"):
            gr.Markdown("Prepare audio samples for voice cloning")
            with gr.Row():
                # Left column - Existing samples browser
                with gr.Column(scale=1):
                    gr.Markdown("### Existing Samples")

                    existing_sample_choices = get_sample_choices()
                    components['existing_sample_dropdown'] = gr.Dropdown(
                        choices=existing_sample_choices,
                        value=existing_sample_choices[0] if existing_sample_choices else None,
                        label="Browse Samples",
                        info="Select a sample to preview or edit"
                    )

                    with gr.Row():
                        components['preview_sample_btn'] = gr.Button("Preview Sample", size="sm")
                        components['refresh_preview_btn'] = gr.Button("Refresh Preview", size="sm")
                        components['load_sample_btn'] = gr.Button("Load to Editor", size="sm")
                        components['clear_cache_btn'] = gr.Button("Clear Cache", size="sm")
                        components['delete_sample_btn'] = gr.Button("Delete", size="sm")

                    components['existing_sample_audio'] = gr.Audio(
                        label="Sample Preview",
                        type="filepath",
                        interactive=False
                    )

                    components['existing_sample_text'] = gr.Textbox(
                        label="Sample Text",
                        max_lines=10,
                        interactive=False
                    )

                    components['existing_sample_info'] = gr.Textbox(
                        label="Info",
                        interactive=False
                    )

                    with gr.Row():
                        components['whisper_language'] = gr.Dropdown(
                            choices=["Auto-detect"] + LANGUAGES[1:],
                            value=_user_config.get("whisper_language", "Auto-detect"),
                            label="Language",
                        )

                        available_models = ['VibeVoice ASR']
                        if WHISPER_AVAILABLE:
                            available_models.insert(0, 'Whisper')

                        default_model = _user_config.get("transcribe_model", "Whisper")
                        if default_model not in available_models:
                            default_model = available_models[0]

                        components['transcribe_model'] = gr.Dropdown(
                            choices=available_models,
                            value=default_model,
                            label="Model",
                        )

                # Right column - Audio/Video editing
                with gr.Column(scale=2):
                    gr.Markdown("### Edit Audio/Video")

                    components['prep_file_input'] = gr.File(
                        label="Audio or Video File",
                        type="filepath",
                        file_types=["audio", "video"],
                        interactive=True
                    )

                    components['prep_audio_editor'] = gr.Audio(
                        label="Audio Editor (Use Trim icon ✂️ to edit)",
                        type="filepath",
                        interactive=True,
                        visible=False
                    )

                    with gr.Row():
                        components['clear_btn'] = gr.Button("Clear", scale=1, size="sm")
                        components['clean_btn'] = gr.Button("AI Denoise", scale=2, size="sm", variant="secondary", visible=DEEPFILTER_AVAILABLE)
                        components['normalize_btn'] = gr.Button("Normalize Volume", scale=2, size="sm")
                        components['mono_btn'] = gr.Button("Convert to Mono", scale=2, size="sm")

                    components['prep_audio_info'] = gr.Textbox(
                        label="Audio Info",
                        interactive=False
                    )
                    with gr.Column(scale=2):
                        gr.Markdown("### Transcription / Reference Text")
                        components['transcription_output'] = gr.Textbox(
                            label="Text",
                            lines=4,
                            max_lines=10,
                            interactive=True,
                            placeholder="Transcription will appear here, or enter/edit text manually..."
                        )

                        with gr.Row():
                            components['transcribe_btn'] = gr.Button("Transcribe Audio", variant="primary")
                            components['save_sample_btn'] = gr.Button("Save Sample", variant="primary")

                        components['save_status'] = gr.Textbox(label="Status", interactive=False, scale=1)

        return components
    
    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Prep Samples tab events."""
        
        # Get helper functions
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        get_audio_duration = shared_state['get_audio_duration']
        format_time = shared_state['format_time']
        load_existing_sample = shared_state['load_existing_sample']
        refresh_samples = shared_state['refresh_samples']
        delete_sample = shared_state['delete_sample']
        clear_sample_cache = shared_state['clear_sample_cache']
        on_prep_audio_load = shared_state['on_prep_audio_load']
        normalize_audio = shared_state['normalize_audio']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']
        transcribe_audio = shared_state['transcribe_audio']
        save_as_sample = shared_state['save_as_sample']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        
        def load_sample_to_editor(sample_name):
            """Load sample into the working audio editor."""
            if not sample_name:
                return None, None, "", "No sample selected", gr.update(visible=False)
            samples = get_available_samples()
            for s in samples:
                if s["name"] == sample_name:
                    duration = get_audio_duration(s["wav_path"])
                    info = f"Duration: {format_time(duration)} ({duration:.2f}s)"
                    return s["wav_path"], s["wav_path"], s["ref_text"], info, gr.update(visible=True)
            return None, None, "", "Sample not found", gr.update(visible=False)

        components['load_sample_btn'].click(
            load_sample_to_editor,
            inputs=[components['existing_sample_dropdown']],
            outputs=[components['prep_file_input'], components['prep_audio_editor'], components['transcription_output'], components['prep_audio_info'], components['prep_audio_editor']]
        )

        # Preview on dropdown change
        components['existing_sample_dropdown'].change(
            load_existing_sample,
            inputs=[components['existing_sample_dropdown']],
            outputs=[components['existing_sample_audio'], components['existing_sample_text'], components['existing_sample_info']]
        )

        # Preview button
        components['preview_sample_btn'].click(
            load_existing_sample,
            inputs=[components['existing_sample_dropdown']],
            outputs=[components['existing_sample_audio'], components['existing_sample_text'], components['existing_sample_info']]
        )

        # Refresh preview button
        components['refresh_preview_btn'].click(
            refresh_samples,
            outputs=[components['existing_sample_dropdown']]
        )

        # Delete sample
        components['delete_sample_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Sample?",
                message="This will permanently delete the sample audio, text, and cached files. This action cannot be undone.",
                confirm_button_text="Delete",
                context="sample_"
            )
        )

        # Process confirmation
        confirm_trigger.change(
            delete_sample,
            inputs=[confirm_trigger, components['existing_sample_dropdown']],
            outputs=[components['save_status'], components['existing_sample_dropdown'], shared_state['sample_dropdown']]
        )

        # Clear cache
        components['clear_cache_btn'].click(
            clear_sample_cache,
            inputs=[components['existing_sample_dropdown']],
            outputs=[components['save_status'], components['existing_sample_info']]
        )

        # When file is loaded/changed
        components['prep_file_input'].change(
            on_prep_audio_load,
            inputs=[components['prep_file_input']],
            outputs=[components['prep_audio_editor'], components['prep_audio_info']]
        ).then(
            lambda audio: (
                gr.update(visible=audio is not None),
                gr.update(visible=audio is None)
            ),
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor'], components['prep_file_input']]
        )

        # Clear file input and reset
        components['clear_btn'].click(
            lambda: (None, None, ""),
            outputs=[components['prep_file_input'], components['prep_audio_editor'], components['prep_audio_info']]
        )

        # Normalize
        components['normalize_btn'].click(
            normalize_audio,
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor']]
        )

        # Convert to mono
        components['mono_btn'].click(
            convert_to_mono,
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor']]
        )

        # Clean audio
        components['clean_btn'].click(
            clean_audio,
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor']]
        )

        # Transcribe
        components['transcribe_btn'].click(
            transcribe_audio,
            inputs=[components['prep_audio_editor'], components['whisper_language'], components['transcribe_model']],
            outputs=[components['transcription_output']]
        )

        # Save sample - show modal
        components['save_sample_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_input_modal_js(
                title="Save Voice Sample",
                message="Enter a name for this voice sample:",
                placeholder="e.g., MyVoice, Female-Accent, John-Doe",
                context="save_sample_"
            )
        )

        # Handler for save sample input modal
        def handle_save_sample_input(input_value, audio, transcription):
            """Process input modal submission for saving sample."""
            if not input_value or not input_value.startswith("save_sample_"):
                return gr.update(), gr.update(), gr.update()

            parts = input_value.split("_")
            if len(parts) >= 3:
                if parts[2] == "cancel":
                    return gr.update(), gr.update(), gr.update()
                sample_name = "_".join(parts[2:-1])
                status, dropdown1_update, dropdown2_update, _ = save_as_sample(audio, transcription, sample_name)
                return status, dropdown1_update, dropdown2_update

            return gr.update(), gr.update(), gr.update()

        input_trigger.change(
            handle_save_sample_input,
            inputs=[input_trigger, components['prep_audio_editor'], components['transcription_output']],
            outputs=[components['save_status'], components['existing_sample_dropdown'], shared_state['sample_dropdown']]
        )

        # Save preferences
        components['transcribe_model'].change(
            lambda x: save_preference("transcribe_model", x),
            inputs=[components['transcribe_model']],
            outputs=[]
        )

        components['whisper_language'].change(
            lambda x: save_preference("whisper_language", x),
            inputs=[components['whisper_language']],
            outputs=[]
        )


# Export for tab registry
get_tab_class = lambda: PrepSamplesTab
