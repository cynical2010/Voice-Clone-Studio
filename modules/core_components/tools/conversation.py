"""
Conversation Tab

Create conversations using VibeVoice, Qwen Base, or Qwen CustomVoice.
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
import numpy as np
import random
import re
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from modules.core_components.tools.base import Tab, TabConfig
from modules.core_components.tool_utils import format_help_html
from modules.core_components.ai_models.tts_manager import get_tts_manager

# TODO: Add these helper methods to this class before setup_events():
# - generate_conversation_handler() - Qwen CustomVoice with preset speakers
# - generate_conversation_base_handler() - Qwen Base with custom voice samples  
# - generate_vibevoice_longform_handler() - VibeVoice long-form generation
# Each handler should call get_tts_manager() internally and be fully self-contained
# See voice_clone_studio.py lines 1813-2400 for implementation reference


class ConversationTab(Tab):
    """Conversation tab implementation."""
    
    config = TabConfig(
        name="Conversation",
        module_name="tab_conversation",
        description="Create multi-speaker conversations",
        enabled=True,
        category="generation"
    )
    
    @classmethod
    def create_tab(cls, shared_state):
        """Create Conversation tab UI."""
        components = {}
        
        # Get helper functions and config
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        create_vibevoice_advanced_params = shared_state['create_vibevoice_advanced_params']
        create_qwen_advanced_params = shared_state['create_qwen_advanced_params']
        create_emotion_intensity_slider = shared_state['create_emotion_intensity_slider']
        _user_config = shared_state['_user_config']
        LANGUAGES = shared_state['LANGUAGES']
        MODEL_SIZES_CUSTOM = shared_state['MODEL_SIZES_CUSTOM']
        MODEL_SIZES_BASE = shared_state['MODEL_SIZES_BASE']
        MODEL_SIZES_VIBEVOICE = shared_state['MODEL_SIZES_VIBEVOICE']
        generate_conversation = shared_state['generate_conversation']
        generate_conversation_base = shared_state['generate_conversation_base']
        generate_vibevoice_longform = shared_state['generate_vibevoice_longform']
        
        # Model selector at top
        initial_conv_model = _user_config.get("conv_model_type", "VibeVoice")
        is_vibevoice = initial_conv_model == "VibeVoice"
        is_qwen_base = initial_conv_model == "Qwen Base"
        is_qwen_custom = initial_conv_model == "Qwen CustomVoice"

        components['conv_model_type'] = gr.Radio(
            choices=["VibeVoice", "Qwen Base", "Qwen CustomVoice"],
            value=initial_conv_model,
            show_label=False,
            container=False
        )

        # Get sample choices once for all dropdowns
        conversation_available_samples = get_sample_choices()
        conversation_first_sample = conversation_available_samples[0] if conversation_available_samples else None

        with gr.Row():
            # Left - Script input and model-specific controls
            with gr.Column(scale=2):
                gr.Markdown("### Conversation Script")

                components['conversation_script'] = gr.Textbox(
                    label="Script:",
                    placeholder=dedent("""\
                        Use [N]: format for speaker labels (1-4 for VibeVoice, 1-8 for Base, 1-9 for CustomVoice).
                        Qwen also supports (style) for emotions:

                        [1]: (cheerful) Hey, how's it going?
                        [2]: (excited) I'm doing great, thanks for asking!
                        [1]: That's wonderful to hear.
                        [3]: (curious) Mind if I join this conversation?

                        VibeVoice: Natural long-form generation.
                        Base: Your custom voice clips with advanced pause control, with hacked Style control.
                        CustomVoice: Qwen Preset speakers with style control and Pause Controls"""),
                    lines=18
                )

                # Qwen speaker mapping
                speaker_guide = dedent("""\
                    **Qwen Speaker Numbers → Preset Voices:**

                    | # | Speaker | Voice | Language |   | # | Speaker | Voice | Language |
                    |---|---------|-------|----------|---|---|---------|-------|----------|
                    | 1 | Vivian | Bright young female | 🇨🇳 Chinese |   | 6 | Ryan | Dynamic male | 🇺🇸 English |
                    | 2 | Serena | Warm gentle female | 🇨🇳 Chinese |   | 7 | Aiden | Sunny American male | 🇺🇸 English |
                    | 3 | Uncle_Fu | Seasoned mellow male | 🇨🇳 Chinese |   | 8 | Ono_Anna | Playful female | 🇯🇵 Japanese |
                    | 4 | Dylan | Youthful Beijing male | 🇨🇳 Chinese |   | 9 | Sohee | Warm female | 🇰🇷 Korean |
                    | 5 | Eric | Lively Chengdu male | 🇨🇳 Chinese |  |  |  |  |  |

                    *Each speaker works best in their native language.*
                    """)

                components['qwen_speaker_table'] = gr.HTML(
                    value=format_help_html(speaker_guide),
                    container=True,
                    padding=True,
                    visible=is_qwen_custom
                )

                # Qwen Base voice sample selectors
                components['qwen_base_voices_section'] = gr.Column(visible=is_qwen_base)
                with components['qwen_base_voices_section']:
                    gr.Markdown("### Voice Samples (Up to 8 Speakers)")

                    with gr.Row():
                        with gr.Column():
                            components['qwen_voice_sample_1'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[1] Voice Sample",
                                info="Select from your prepared samples"
                            )
                        with gr.Column():
                            components['qwen_voice_sample_2'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[2] Voice Sample",
                                info="Select from your prepared samples"
                            )

                    with gr.Row():
                        with gr.Column():
                            components['qwen_voice_sample_3'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[3] Voice Sample",
                                info="Select from your prepared samples"
                            )
                        with gr.Column():
                            components['qwen_voice_sample_4'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[4] Voice Sample",
                                info="Select from your prepared samples"
                            )

                    with gr.Row():
                        with gr.Column():
                            components['qwen_voice_sample_5'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[5] Voice Sample",
                                info="Select from your prepared samples"
                            )
                        with gr.Column():
                            components['qwen_voice_sample_6'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[6] Voice Sample",
                                info="Select from your prepared samples"
                            )

                    with gr.Row():
                        with gr.Column():
                            components['qwen_voice_sample_7'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[7] Voice Sample",
                                info="Select from your prepared samples"
                            )
                        with gr.Column():
                            components['qwen_voice_sample_8'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[8] Voice Sample",
                                info="Select from your prepared samples"
                            )

                    components['refresh_qwen_samples_btn'] = gr.Button("Refresh Voice Samples", size="md")

                # VibeVoice voice sample selectors
                components['vibevoice_voices_section'] = gr.Column(visible=is_vibevoice)
                with components['vibevoice_voices_section']:
                    gr.Markdown("### Voice Samples (Up to 4 Speakers)")

                    with gr.Row():
                        with gr.Column():
                            components['voice_sample_1'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[1] Voice Sample",
                                info="Select from your prepared samples"
                            )
                        with gr.Column():
                            components['voice_sample_2'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[2] Voice Sample",
                                info="Select from your prepared samples"
                            )

                    with gr.Row():
                        with gr.Column():
                            components['voice_sample_3'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[3] Voice Sample",
                                info="Select from your prepared samples"
                            )
                        with gr.Column():
                            components['voice_sample_4'] = gr.Dropdown(
                                choices=conversation_available_samples,
                                value=conversation_first_sample,
                                label="[4] Voice Sample",
                                info="Select from your prepared samples"
                            )

                    components['refresh_conv_samples_btn'] = gr.Button("Refresh Voice Samples", size="md")

            # Right - Settings and output
            with gr.Column(scale=1):
                gr.Markdown("### Settings")

                # Qwen CustomVoice settings
                components['qwen_custom_settings'] = gr.Column(visible=is_qwen_custom)
                with components['qwen_custom_settings']:
                    components['conv_model_size'] = gr.Dropdown(
                        choices=MODEL_SIZES_CUSTOM,
                        value=_user_config.get("conv_model_size", "Large"),
                        label="Model Size",
                        info="Small = Faster, Large = Better Quality"
                    )

                # Qwen Base settings
                components['qwen_base_settings'] = gr.Column(visible=is_qwen_base)
                with components['qwen_base_settings']:
                    components['conv_base_model_size'] = gr.Dropdown(
                        choices=MODEL_SIZES_BASE,
                        value=_user_config.get("conv_base_model_size", "Small"),
                        label="Model Size",
                        info="Small = Faster, Large = Better Quality"
                    )

                # Shared Language and Seed
                components['qwen_language_seed'] = gr.Column(visible=(is_qwen_custom or is_qwen_base))
                with components['qwen_language_seed']:
                    with gr.Row():
                        components['conv_language'] = gr.Dropdown(
                            scale=5,
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                            info="Language for all lines (Auto recommended)"
                        )
                        components['conv_seed'] = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                            info="(-1 for random)"
                        )

                # Shared Pause Controls
                components['qwen_pause_controls'] = gr.Accordion("Pause Controls", open=False, visible=(is_qwen_custom or is_qwen_base))
                with components['qwen_pause_controls']:
                    with gr.Column():
                        components['conv_pause_linebreak'] = gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            value=_user_config.get("conv_pause_linebreak", 0.5),
                            step=0.1,
                            label="Pause Between Lines",
                            info="Silence between each speaker turn"
                        )

                        with gr.Row():
                            components['conv_pause_period'] = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=_user_config.get("conv_pause_period", 0.4),
                                step=0.1,
                                label="After Period (.)",
                                info="Pause after periods"
                            )
                            components['conv_pause_comma'] = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=_user_config.get("conv_pause_comma", 0.2),
                                step=0.1,
                                label="After Comma (,)",
                                info="Pause after commas"
                            )

                        with gr.Row():
                            components['conv_pause_question'] = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=_user_config.get("conv_pause_question", 0.6),
                                step=0.1,
                                label="After Question (?)",
                                info="Pause after questions"
                            )
                            components['conv_pause_hyphen'] = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=_user_config.get("conv_pause_hyphen", 0.3),
                                step=0.1,
                                label="After Hyphen (-)",
                                info="Pause after hyphens"
                            )

                # VibeVoice-specific settings
                components['vibevoice_settings'] = gr.Column(visible=is_vibevoice)
                with components['vibevoice_settings']:
                    components['longform_model_size'] = gr.Dropdown(
                        choices=MODEL_SIZES_VIBEVOICE,
                        value=_user_config.get("vibevoice_model_size", "Large"),
                        label="Model Size",
                        info="Small = Faster, Large = Better Quality"
                    )

                    # VibeVoice Advanced Parameters
                    vv_conv_params = create_vibevoice_advanced_params(
                        initial_num_steps=20,
                        initial_cfg_scale=3.0,
                        visible=is_vibevoice
                    )
                    components['vv_conv_num_steps'] = vv_conv_params['num_steps']
                    components['longform_cfg_scale'] = vv_conv_params['cfg_scale']
                    components['vv_conv_do_sample'] = vv_conv_params['do_sample']
                    components['vv_conv_repetition_penalty'] = vv_conv_params['repetition_penalty']
                    components['vv_conv_temperature'] = vv_conv_params['temperature']
                    components['vv_conv_top_k'] = vv_conv_params['top_k']
                    components['vv_conv_top_p'] = vv_conv_params['top_p']

                # Qwen Advanced Parameters
                components['qwen_conv_advanced'] = gr.Column(visible=(is_qwen_custom or is_qwen_base))
                with components['qwen_conv_advanced']:
                    # Emotion intensity slider
                    components['conv_emotion_intensity_row'] = gr.Row(visible=is_qwen_base)
                    with components['conv_emotion_intensity_row']:
                        components['conv_emotion_intensity'] = create_emotion_intensity_slider(
                            initial_intensity=1.0,
                            label="Emotion Intensity",
                            visible=is_qwen_base
                        )
                    
                    # Qwen advanced parameters
                    qwen_conv_params = create_qwen_advanced_params(
                        include_emotion=False,
                        visible=(is_qwen_custom or is_qwen_base)
                    )
                    components['qwen_conv_do_sample'] = qwen_conv_params['do_sample']
                    components['qwen_conv_temperature'] = qwen_conv_params['temperature']
                    components['qwen_conv_top_k'] = qwen_conv_params['top_k']
                    components['qwen_conv_top_p'] = qwen_conv_params['top_p']
                    components['qwen_conv_repetition_penalty'] = qwen_conv_params['repetition_penalty']
                    components['qwen_conv_max_new_tokens'] = qwen_conv_params['max_new_tokens']

                # Shared settings
                components['conv_generate_btn'] = gr.Button("Generate Conversation", variant="primary", size="lg")

                components['conv_output_audio'] = gr.Audio(
                    label="Generated Conversation",
                    type="filepath"
                )
                components['conv_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)

                # Model-specific tips
                qwen_custom_tips_text = dedent("""\
                **Qwen CustomVoice Tips:**
                - Fast generation with preset voices
                - Up to 9 different speakers
                - Tip: Use `[break=1.5]` inline for custom pauses
                - Each voice optimized for their native language
                - Style instructions: (cheerful), (sad), (excited), etc.
                """)

                qwen_base_tips_text = dedent("""\
                **Qwen Base Tips:**
                - Use your own custom voice samples
                - Up to 8 different speakers
                - Tip: Use `[break=1.5]` inline for custom pauses
                - Advanced pause control (periods, commas, questions, hyphens)
                - Prepare 3-10 second voice samples in samples/ folder
                """)

                vibevoice_tips_text = dedent("""\
                **VibeVoice Tips:**
                - Up to 90 minutes continuous generation
                - Up to 4 speakers with custom voices
                - May spontaneously add background music/sounds
                - Longer scripts work best with Large model
                - Natural conversation flow (no manual pause control)
                """)

                components['qwen_custom_tips'] = gr.HTML(
                    value=format_help_html(qwen_custom_tips_text),
                    container=True,
                    padding=True,
                    visible=is_qwen_custom
                )

                components['qwen_base_tips'] = gr.HTML(
                    value=format_help_html(qwen_base_tips_text),
                    container=True,
                    padding=True,
                    visible=is_qwen_base
                )

                components['vibevoice_tips'] = gr.HTML(
                    value=format_help_html(vibevoice_tips_text),
                    container=True,
                    padding=True,
                    visible=is_vibevoice
                )

        return components
    
    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Conversation tab events."""
        
        # Get helper functions
        get_available_samples = shared_state['get_available_samples']
        get_sample_choices = shared_state['get_sample_choices']
        generate_conversation = shared_state['generate_conversation']
        generate_conversation_base = shared_state['generate_conversation_base']
        generate_vibevoice_longform = shared_state['generate_vibevoice_longform']
        save_preference = shared_state['save_preference']
        
        def prepare_voice_samples_dict(v1, v2=None, v3=None, v4=None, v5=None, v6=None, v7=None, v8=None):
            """Prepare voice samples dictionary for generation."""
            samples = {}
            available_samples = get_available_samples()

            voice_inputs = [("Speaker1", v1), ("Speaker2", v2), ("Speaker3", v3), ("Speaker4", v4),
                            ("Speaker5", v5), ("Speaker6", v6), ("Speaker7", v7), ("Speaker8", v8)]

            for speaker_num, sample_name in voice_inputs:
                if sample_name:
                    for s in available_samples:
                        if s["name"] == sample_name:
                            samples[speaker_num] = {
                                "wav_path": s["wav_path"],
                                "ref_text": s["ref_text"]
                            }
                            break
            return samples

        def unified_conversation_generate(
            model_type, script,
            # Qwen CustomVoice params
            qwen_custom_pause_linebreak, qwen_custom_pause_period, qwen_custom_pause_comma,
            qwen_custom_pause_question, qwen_custom_pause_hyphen, qwen_custom_model_size,
            # Qwen Base params
            qwen_base_v1, qwen_base_v2, qwen_base_v3, qwen_base_v4, qwen_base_v5, qwen_base_v6, qwen_base_v7, qwen_base_v8,
            qwen_base_pause_linebreak, qwen_base_pause_period, qwen_base_pause_comma, qwen_base_pause_question,
            qwen_base_pause_hyphen, qwen_base_model_size,
            # Shared Qwen params
            qwen_lang, qwen_seed, emotion_intensity,
            # Qwen advanced params
            qwen_do_sample, qwen_temperature, qwen_top_k, qwen_top_p, qwen_repetition_penalty, qwen_max_new_tokens,
            # VibeVoice params
            vv_v1, vv_v2, vv_v3, vv_v4, vv_model_size, vv_cfg,
            # VibeVoice advanced params
            vv_num_steps, vv_do_sample, vv_temperature, vv_top_k, vv_top_p, vv_repetition_penalty,
            # Shared
            seed, progress=gr.Progress()
        ):
            """Route to appropriate generation function based on model type."""
            if model_type == "Qwen CustomVoice":
                qwen_size = "1.7B" if qwen_custom_model_size == "Large" else "0.6B"
                return generate_conversation(script, qwen_custom_pause_linebreak, qwen_custom_pause_period,
                                             qwen_custom_pause_comma, qwen_custom_pause_question,
                                             qwen_custom_pause_hyphen, qwen_lang, qwen_seed, qwen_size,
                                             qwen_do_sample, qwen_temperature, qwen_top_k, qwen_top_p,
                                             qwen_repetition_penalty, qwen_max_new_tokens)
            elif model_type == "Qwen Base":
                qwen_size = "1.7B" if qwen_base_model_size == "Large" else "0.6B"
                voice_samples = prepare_voice_samples_dict(
                    qwen_base_v1, qwen_base_v2, qwen_base_v3, qwen_base_v4,
                    qwen_base_v5, qwen_base_v6, qwen_base_v7, qwen_base_v8
                )
                return generate_conversation_base(script, voice_samples, qwen_base_pause_linebreak,
                                                  qwen_base_pause_period, qwen_base_pause_comma,
                                                  qwen_base_pause_question, qwen_base_pause_hyphen,
                                                  qwen_lang, qwen_seed, qwen_size,
                                                  qwen_do_sample, qwen_temperature, qwen_top_k, qwen_top_p,
                                                  qwen_repetition_penalty, qwen_max_new_tokens,
                                                  emotion_intensity, progress)
            else:  # VibeVoice
                if vv_model_size == "Small":
                    vv_size = "1.5B"
                elif vv_model_size == "Large (4-bit)":
                    vv_size = "Large (4-bit)"
                else:
                    vv_size = "Large"
                voice_samples = prepare_voice_samples_dict(vv_v1, vv_v2, vv_v3, vv_v4)
                return generate_vibevoice_longform(script, voice_samples, vv_size, vv_cfg, seed,
                                                   vv_num_steps, vv_do_sample, vv_temperature, vv_top_k,
                                                   vv_top_p, vv_repetition_penalty, progress)

        # Event handlers
        components['conv_generate_btn'].click(
            unified_conversation_generate,
            inputs=[
                components['conv_model_type'], components['conversation_script'],
                # Qwen CustomVoice
                components['conv_pause_linebreak'], components['conv_pause_period'], components['conv_pause_comma'],
                components['conv_pause_question'], components['conv_pause_hyphen'], components['conv_model_size'],
                # Qwen Base
                components['qwen_voice_sample_1'], components['qwen_voice_sample_2'], components['qwen_voice_sample_3'], components['qwen_voice_sample_4'],
                components['qwen_voice_sample_5'], components['qwen_voice_sample_6'], components['qwen_voice_sample_7'], components['qwen_voice_sample_8'],
                components['conv_pause_linebreak'], components['conv_pause_period'], components['conv_pause_comma'],
                components['conv_pause_question'], components['conv_pause_hyphen'], components['conv_base_model_size'],
                # Shared Qwen
                components['conv_language'], components['conv_seed'], components['conv_emotion_intensity'],
                # Qwen advanced params
                components['qwen_conv_do_sample'], components['qwen_conv_temperature'], components['qwen_conv_top_k'], components['qwen_conv_top_p'],
                components['qwen_conv_repetition_penalty'], components['qwen_conv_max_new_tokens'],
                # VibeVoice
                components['voice_sample_1'], components['voice_sample_2'], components['voice_sample_3'], components['voice_sample_4'],
                components['longform_model_size'], components['longform_cfg_scale'],
                # VibeVoice advanced params
                components['vv_conv_num_steps'], components['vv_conv_do_sample'], components['vv_conv_temperature'], components['vv_conv_top_k'],
                components['vv_conv_top_p'], components['vv_conv_repetition_penalty'],
                # Shared
                components['conv_seed']
            ],
            outputs=[components['conv_output_audio'], components['conv_status']]
        )

        # Toggle UI based on model selection
        def toggle_conv_ui(model_type):
            is_qwen_custom = model_type == "Qwen CustomVoice"
            is_qwen_base = model_type == "Qwen Base"
            is_vibevoice = model_type == "VibeVoice"
            is_qwen = is_qwen_custom or is_qwen_base
            return {
                components['qwen_speaker_table']: gr.update(visible=is_qwen_custom),
                components['qwen_base_voices_section']: gr.update(visible=is_qwen_base),
                components['vibevoice_voices_section']: gr.update(visible=is_vibevoice),
                components['qwen_custom_settings']: gr.update(visible=is_qwen_custom),
                components['qwen_base_settings']: gr.update(visible=is_qwen_base),
                components['qwen_language_seed']: gr.update(visible=is_qwen),
                components['qwen_pause_controls']: gr.update(visible=is_qwen),
                components['conv_emotion_intensity_row']: gr.update(visible=is_qwen_base),
                components['qwen_conv_advanced']: gr.update(visible=is_qwen),
                components['vibevoice_settings']: gr.update(visible=is_vibevoice),
                components['qwen_custom_tips']: gr.update(visible=is_qwen_custom),
                components['qwen_base_tips']: gr.update(visible=is_qwen_base),
                components['vibevoice_tips']: gr.update(visible=is_vibevoice)
            }

        components['conv_model_type'].change(
            toggle_conv_ui,
            inputs=[components['conv_model_type']],
            outputs=[components['qwen_speaker_table'], components['qwen_base_voices_section'], components['vibevoice_voices_section'],
                     components['qwen_custom_settings'], components['qwen_base_settings'], components['qwen_language_seed'], components['qwen_pause_controls'],
                     components['conv_emotion_intensity_row'], components['qwen_conv_advanced'], components['vibevoice_settings'],
                     components['qwen_custom_tips'], components['qwen_base_tips'], components['vibevoice_tips']]
        )

        # Refresh voice samples handler
        def refresh_voice_samples():
            """Refresh all voice sample dropdowns."""
            updated_samples = get_sample_choices()
            return [gr.update(choices=updated_samples)] * 4

        def refresh_qwen_voice_samples():
            """Refresh Qwen Base voice sample dropdowns."""
            updated_samples = get_sample_choices()
            return [gr.update(choices=updated_samples)] * 8

        components['refresh_conv_samples_btn'].click(
            refresh_voice_samples,
            inputs=[],
            outputs=[components['voice_sample_1'], components['voice_sample_2'], components['voice_sample_3'], components['voice_sample_4']]
        )

        components['refresh_qwen_samples_btn'].click(
            refresh_qwen_voice_samples,
            inputs=[],
            outputs=[components['qwen_voice_sample_1'], components['qwen_voice_sample_2'], components['qwen_voice_sample_3'], components['qwen_voice_sample_4'],
                     components['qwen_voice_sample_5'], components['qwen_voice_sample_6'], components['qwen_voice_sample_7'], components['qwen_voice_sample_8']]
        )

        # Save preferences
        components['conv_model_type'].change(
            lambda x: save_preference("conv_model_type", x),
            inputs=[components['conv_model_type']],
            outputs=[]
        )

        components['conv_model_size'].change(
            lambda x: save_preference("conv_model_size", x),
            inputs=[components['conv_model_size']],
            outputs=[]
        )

        components['conv_base_model_size'].change(
            lambda x: save_preference("conv_base_model_size", x),
            inputs=[components['conv_base_model_size']],
            outputs=[]
        )

        components['longform_model_size'].change(
            lambda x: save_preference("vibevoice_model_size", x),
            inputs=[components['longform_model_size']],
            outputs=[]
        )

        components['conv_language'].change(
            lambda x: save_preference("language", x),
            inputs=[components['conv_language']],
            outputs=[]
        )

        # Save conversation pause preferences
        components['conv_pause_linebreak'].change(
            lambda x: save_preference("conv_pause_linebreak", x),
            inputs=[components['conv_pause_linebreak']],
            outputs=[]
        )

        components['conv_pause_period'].change(
            lambda x: save_preference("conv_pause_period", x),
            inputs=[components['conv_pause_period']],
            outputs=[]
        )

        components['conv_pause_comma'].change(
            lambda x: save_preference("conv_pause_comma", x),
            inputs=[components['conv_pause_comma']],
            outputs=[]
        )

        components['conv_pause_question'].change(
            lambda x: save_preference("conv_pause_question", x),
            inputs=[components['conv_pause_question']],
            outputs=[]
        )

        components['conv_pause_hyphen'].change(
            lambda x: save_preference("conv_pause_hyphen", x),
            inputs=[components['conv_pause_hyphen']],
            outputs=[]
        )


# Export for tab registry
get_tab_class = lambda: ConversationTab


if __name__ == "__main__":
    """Standalone testing of Conversation tool."""
    print("[*] Starting Conversation Tool - Standalone Mode")
    print("[!] Note: Conversation handlers not yet fully refactored")
    
    from pathlib import Path
    import sys
    import json
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from modules.core_components.ui_components import (
        create_qwen_advanced_params,
        create_vibevoice_advanced_params,
        create_emotion_intensity_slider
    )
    from modules.core_components.constants import (
        LANGUAGES,
        CUSTOM_VOICE_SPEAKERS,
        MODEL_SIZES_CUSTOM,
        MODEL_SIZES_BASE,
        MODEL_SIZES_VIBEVOICE,
        QWEN_GENERATION_DEFAULTS,
        VIBEVOICE_GENERATION_DEFAULTS
    )
    from modules.core_components.tool_utils import load_config, save_preference as save_pref_to_file
    
    # Load config
    user_config = load_config()
    
    SAMPLES_DIR = project_root / "samples"
    OUTPUT_DIR = project_root / "output"
    SAMPLES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    def get_sample_choices():
        samples = []
        for json_file in SAMPLES_DIR.glob("*.json"):
            samples.append(json_file.stem)
        return samples if samples else ["(No samples found)"]
    
    def get_available_samples():
        samples = []
        for json_file in SAMPLES_DIR.glob("*.json"):
            wav_file = json_file.with_suffix(".wav")
            if wav_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        meta = json.load(f)
                    samples.append({
                        "name": meta.get("name", json_file.stem),
                        "wav_path": str(wav_file),
                        "ref_text": meta.get("text", ""),
                        "meta": meta
                    })
                except:
                    pass
        return samples
    
    shared_state = {
        'get_sample_choices': get_sample_choices,
        'get_available_samples': get_available_samples,
        'LANGUAGES': LANGUAGES,
        'CUSTOM_VOICE_SPEAKERS': CUSTOM_VOICE_SPEAKERS,
        'MODEL_SIZES_CUSTOM': MODEL_SIZES_CUSTOM,
        'MODEL_SIZES_BASE': MODEL_SIZES_BASE,
        'MODEL_SIZES_VIBEVOICE': MODEL_SIZES_VIBEVOICE,
        'create_vibevoice_advanced_params': create_vibevoice_advanced_params,
        'create_qwen_advanced_params': create_qwen_advanced_params,
        'create_emotion_intensity_slider': create_emotion_intensity_slider,
        '_user_config': user_config,
        'OUTPUT_DIR': OUTPUT_DIR,
        'SAMPLES_DIR': SAMPLES_DIR,
        'generate_conversation': lambda *args: (None, "TODO: Not yet implemented"),
        'generate_conversation_base': lambda *args: (None, "TODO: Not yet implemented"),
        'generate_vibevoice_longform': lambda *args: (None, "TODO: Not yet implemented"),
        'save_preference': lambda k, v: save_pref_to_file(user_config, k, v),
        'play_completion_beep': lambda: print("[Beep] Complete!"),
    }
    
    print(f"[*] Samples: {SAMPLES_DIR} ({len(get_available_samples())} found)")
    print(f"[*] Output: {OUTPUT_DIR}")
    
    # Load custom theme
    theme = gr.themes.Base.load('modules/core_components/theme.json')
    
    with gr.Blocks(title="Conversation - Standalone") as app:
        gr.Markdown("# 💬 Conversation Tool (Standalone Testing)")
        gr.Markdown("*Standalone mode with persistent settings*")
        gr.Markdown("*⚠️ Generation handlers not yet refactored - will show TODO message*")
        
        components = ConversationTab.create_tab(shared_state)
        ConversationTab.setup_events(components, shared_state)
    
    print("[*] Launching on http://127.0.0.1:7864")
    app.launch(theme=theme, server_port=7864, server_name="127.0.0.1", share=False, inbrowser=True)
