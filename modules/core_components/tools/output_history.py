"""
Output History Tab

Browse and manage previously generated audio files.

Standalone testing:
    python -m modules.core_components.tools.output_history
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path as PathLib
    project_root = PathLib(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
from pathlib import Path
from modules.core_components.tool_base import Tab, TabConfig


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

        # Get OUTPUT_DIR from shared_state
        OUTPUT_DIR = shared_state.get('OUTPUT_DIR')

        # Helper function to get output files
        def get_output_files():
            """Get list of generated output files with date/time.
            
            Returns:
                List of [filename, date, time] for dataframe
            """
            from datetime import datetime
            if not OUTPUT_DIR or not OUTPUT_DIR.exists():
                return []
            files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Format as [filename, date, time]
            result = []
            for f in files:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                result.append([f.name, mtime.strftime('%Y-%m-%d'), mtime.strftime('%H:%M:%S')])
            return result

        with gr.TabItem("Output History"):
            gr.Markdown("Browse and manage previously generated audio files")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Column(scale=1, elem_id="output-files-container"):
                        components['output_dropdown'] = gr.Dataframe(
                            value=get_output_files(),
                            headers=["Filename", "Date", "Time"],
                            interactive=False,
                            elem_id="output-files-group",
                            row_count=(15, "fixed"),
                            col_count=(3, "fixed"),
                            column_widths=["auto", 150, 150],
                            wrap=True
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
                        max_lines=15
                    )
                    # Hidden textbox to store selected filename for delete
                    components['selected_file'] = gr.Textbox(visible=False)
                    components['delete_output_btn'] = gr.Button("Delete", size="sm")

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Output History events."""

        # Get required items from shared_state
        OUTPUT_DIR = shared_state.get('OUTPUT_DIR')
        show_confirmation_modal_js = shared_state.get('show_confirmation_modal_js')
        confirm_trigger = shared_state.get('confirm_trigger')

        # Local helper functions
        def get_output_files():
            """Get list of generated output files with date/time.
            
            Returns:
                List of [filename, date, time] for dataframe
            """
            from datetime import datetime
            if not OUTPUT_DIR or not OUTPUT_DIR.exists():
                return []
            files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Format as [filename, date, time]
            result = []
            for f in files:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                result.append([f.name, mtime.strftime('%Y-%m-%d'), mtime.strftime('%H:%M:%S')])
            return result

        def refresh_outputs():
            """Refresh the output file list."""
            return gr.update(value=get_output_files())

        def sort_by_name(sort_state):
            """Toggle sort by name: normal -> reversed -> back to date."""
            clicks = sort_state['clicks']['name']

            if clicks == 0:
                # First click: sort by name ascending
                sort_state['by'] = 'name'
                sort_state['reverse'] = False
                sort_state['clicks']['name'] = 1
                sort_state['clicks']['date'] = 0
            elif clicks == 1:
                # Second click: sort by name descending
                sort_state['by'] = 'name'
                sort_state['reverse'] = True
                sort_state['clicks']['name'] = 2
            else:
                # Third click: revert to default (date descending)
                sort_state['by'] = 'date'
                sort_state['reverse'] = True
                sort_state['clicks']['name'] = 0
                sort_state['clicks']['date'] = 0

            return sort_state, gr.update(value=get_output_files(sort_state['by'], sort_state['reverse']))

        def sort_by_date(sort_state):
            """Toggle sort by date: normal -> reversed -> back to normal."""
            clicks = sort_state['clicks']['date']

            if clicks == 0:
                # First click: sort by date ascending (oldest first)
                sort_state['by'] = 'date'
                sort_state['reverse'] = False
                sort_state['clicks']['date'] = 1
                sort_state['clicks']['name'] = 0
            elif clicks == 1:
                # Second click: sort by date descending (newest first)
                sort_state['by'] = 'date'
                sort_state['reverse'] = True
                sort_state['clicks']['date'] = 2
            else:
                # Third click: revert to default (date descending)
                sort_state['by'] = 'date'
                sort_state['reverse'] = True
                sort_state['clicks']['date'] = 0
                sort_state['clicks']['name'] = 0

            return sort_state, gr.update(value=get_output_files(sort_state['by'], sort_state['reverse']))

        def load_output_audio(evt: gr.SelectData):
            """Load a selected output file for playback and show metadata."""
            if not evt or not evt.value:
                return None, ""
            
            # evt.value contains the cell value, evt.index contains [row, col]
            # We need the filename from column 0 of the selected row
            file_path = evt.value if evt.index[1] == 0 else None
            if not file_path:
                return None, ""
            
            file_path = OUTPUT_DIR / file_path

            if file_path.exists():
                metadata_file = file_path.with_suffix(".txt")
                if metadata_file.exists():
                    try:
                        metadata = metadata_file.read_text(encoding="utf-8")
                        return str(file_path), metadata, file_path.name
                    except:
                        pass
                return str(file_path), "No metadata available", file_path.name
            return None, "", ""

        def delete_output_file(action, selected_file):
            """Delete output file and metadata."""
            if not action or not action.strip() or not action.startswith("output_"):
                return gr.update(), gr.update(), gr.update()

            if "cancel" in action:
                return gr.update(), gr.update(), gr.update()

            if "confirm" not in action:
                return gr.update(), gr.update(), gr.update()

            try:
                if not selected_file:
                    return gr.update(), None, "[ERROR] No file selected"
                
                audio_path = OUTPUT_DIR / selected_file

                txt_path = audio_path.with_suffix(".txt")
                deleted = []
                if audio_path.exists():
                    audio_path.unlink()
                    deleted.append("audio")
                if txt_path.exists():
                    txt_path.unlink()
                    deleted.append("text")

                updated_list = get_output_files()
                msg = f"Deleted: {audio_path.name} ({', '.join(deleted)})" if deleted else "[ERROR] Files not found"
                return gr.update(value=updated_list), None, msg
            except Exception as e:
                return gr.update(), None, f"[ERROR] Error: {str(e)}"

        # Show modal on delete button click (only if modal available)
        if show_confirmation_modal_js and confirm_trigger:
            components['delete_output_btn'].click(
                fn=None,
                js=show_confirmation_modal_js(
                    title="Delete Output File?",
                    message="This will permanently delete the generated audio and its metadata. This action cannot be undone.",
                    confirm_button_text="Delete",
                    context="output_"
                )
            )

            # Process confirmation
            confirm_trigger.change(
                delete_output_file,
                inputs=[confirm_trigger, components['selected_file']],
                outputs=[components['output_dropdown'], components['history_audio'], components['history_metadata']]
            )

        # Refresh button
        components['refresh_outputs_btn'].click(
            refresh_outputs,
            outputs=[components['output_dropdown']]
        )

        # Load on dataframe selection
        components['output_dropdown'].select(
            load_output_audio,
            inputs=None,
            outputs=[components['history_audio'], components['history_metadata'], components['selected_file']]
        )


# Export for registry
get_tab_class = lambda: OutputHistoryTab


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(OutputHistoryTab, port=7868, title="Output History - Standalone")
