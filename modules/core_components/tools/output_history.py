"""
Output History Tab

Browse and manage previously generated audio files.
"""

import gradio as gr
from pathlib import Path
from modules.core_components.tools.base import Tab, TabConfig


class OutputHistoryTab(Tab):
    """Output History tab implementation."""
    
    config = TabConfig(
        name="Output History",
        module_name="tab_output_history",
        description="Browse and manage generated audio files",
        enabled=True,
        category="utility"
    )
    
    @classmethod
    def create_tab(cls, shared_state):
        """Create Output History tab UI."""
        components = {}
        
        # Get helper functions from shared state
        get_output_files = shared_state.get('get_output_files', lambda: [])
        
        with gr.TabItem("Output History"):
            gr.Markdown("Browse and manage previously generated audio files")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Column(scale=1, elem_id="output-files-container"):
                        components['output_dropdown'] = gr.Radio(
                            choices=get_output_files(),
                            show_label=False,
                            interactive=True,
                            elem_id="output-files-group"
                        )
                    components['refresh_outputs_btn'] = gr.Button("Refresh", size="sm")

                with gr.Column(scale=1):
                    components['history_audio'] = gr.Audio(
                        label="Playback",
                        type="filepath"
                    )

                    components['history_metadata'] = gr.Textbox(
                        label="Generation Info",
                        interactive=False,
                        max_lines=10
                    )
                    components['delete_output_btn'] = gr.Button("Delete", size="sm")
        
        return components
    
    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Output History events."""
        
        # Get helper functions
        delete_output_file_handler = shared_state.get('delete_output_file_handler')
        refresh_outputs = shared_state.get('refresh_outputs')
        load_output_audio = shared_state.get('load_output_audio')
        get_output_files = shared_state.get('get_output_files', lambda: [])
        show_confirmation_modal_js = shared_state.get('show_confirmation_modal_js')
        confirm_trigger = shared_state.get('confirm_trigger')
        OUTPUT_DIR = shared_state.get('OUTPUT_DIR')
        
        if not all([delete_output_file_handler, refresh_outputs, load_output_audio, 
                   show_confirmation_modal_js, confirm_trigger, OUTPUT_DIR]):
            # Required handlers not available
            return
        
        # Show modal on delete button click
        components['delete_output_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Output File?",
                message="This will permanently delete the generated audio and its metadata. This action cannot be undone.",
                confirm_button_text="Delete",
                context="output_"
            )
        )

        # Process confirmation
        confirm_trigger.change(
            delete_output_file_handler,
            inputs=[confirm_trigger, components['output_dropdown']],
            outputs=[components['output_dropdown'], components['history_audio'], components['history_metadata']]
        )

        components['refresh_outputs_btn'].click(
            refresh_outputs,
            outputs=[components['output_dropdown']]
        )

        # Load on dropdown change
        components['output_dropdown'].change(
            load_output_audio,
            inputs=[components['output_dropdown']],
            outputs=[components['history_audio'], components['history_metadata']]
        )


# Export for registry
get_tab_class = lambda: OutputHistoryTab
