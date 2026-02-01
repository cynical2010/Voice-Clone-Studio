"""
Voice Design Tab

Create new voices from natural language descriptions using Qwen3-TTS VoiceDesign.
"""

import gradio as gr
from modules.core_components.tools.base import Tab, TabConfig
from modules.core_components.ui_components import create_qwen_advanced_params


class VoiceDesignTab(Tab):
    """Voice Design tab implementation."""
    
    config = TabConfig(
        name="Voice Design",
        module_name="tab_voice_design",
        description="Create voices from natural language descriptions",
        enabled=True,
        category="generation"
    )
    
    @classmethod
    def create_tab(cls, shared_state):
        """Create Voice Design tab UI."""
        components = {}
        
        # Get shared utilities
        LANGUAGES = shared_state.get('LANGUAGES', ['Auto'])
        user_config = shared_state.get('user_config', {})
        
        with gr.TabItem("Voice Design"):
            gr.Markdown("Create new voices from natural language descriptions")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Create Design")

                    components['design_text_input'] = gr.Textbox(
                        label="Reference Text",
                        placeholder="Enter the text for the voice design (this will be spoken in the designed voice)...",
                        lines=3,
                        value="Thank you for listening to this voice design sample. This sentence is intentionally a bit long so you can hear the full range and quality of the generated voice."
                    )

                    components['design_instruct_input'] = gr.Textbox(
                        label="Voice Design Instructions",
                        placeholder="Describe the voice: e.g., 'Young female voice, bright and cheerful, slightly breathy' or 'Deep male voice with a warm, comforting tone, speak slowly'",
                        lines=3
                    )

                    with gr.Row():
                        components['design_language'] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=user_config.get("language", "Auto"),
                            label="Language",
                            scale=2
                        )
                        components['design_seed'] = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    components['save_to_output_checkbox'] = gr.Checkbox(
                        label="Save to Output folder instead of Temp",
                        value=False
                    )

                    # Qwen Advanced Parameters
                    design_params = create_qwen_advanced_params(
                        include_emotion=False,
                        visible=True
                    )
                    components.update(design_params)

                    components['design_generate_btn'] = gr.Button("Generate Voice", variant="primary", size="lg")
                    components['design_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=3)

                with gr.Column(scale=1):
                    gr.Markdown("### Preview & Save")
                    components['design_output_audio'] = gr.Audio(
                        label="Generated Audio",
                        type="filepath"
                    )

                    components['design_save_btn'] = gr.Button("Save Sample", variant="primary")
        
        return components
    
    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Design events."""
        
        # Get handler functions
        generate_voice_design = shared_state.get('generate_voice_design')
        show_input_modal_js = shared_state.get('show_input_modal_js')
        save_preference = shared_state.get('save_preference')
        
        if not generate_voice_design:
            return
        
        def generate_voice_design_with_checkbox(text, language, instruct, seed, save_to_output,
                                               do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens,
                                               progress=gr.Progress()):
            return generate_voice_design(text, language, instruct, seed,
                                        do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens,
                                        progress=progress, save_to_output=save_to_output)

        components['design_generate_btn'].click(
            generate_voice_design_with_checkbox,
            inputs=[components['design_text_input'], components['design_language'], components['design_instruct_input'],
                   components['design_seed'], components['save_to_output_checkbox'],
                   components['do_sample'], components['temperature'], components['top_k'],
                   components['top_p'], components['repetition_penalty'], components['max_new_tokens']],
            outputs=[components['design_output_audio'], components['design_status']]
        )

        # Save designed voice - show modal
        components['design_save_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_input_modal_js(
                title="Save Designed Voice",
                message="Enter a name for this voice design:",
                placeholder="e.g., Bright-Female, Deep-Male, Cheerful-Voice",
                context="save_design_"
            )
        )
        
        # Save language preference
        if save_preference:
            components['design_language'].change(
                lambda x: save_preference("language", x),
                inputs=[components['design_language']],
                outputs=[]
            )


# Export for registry
get_tab_class = lambda: VoiceDesignTab
