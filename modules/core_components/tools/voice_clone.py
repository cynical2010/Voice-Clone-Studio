"""
Voice Clone Tab

Clone voices from samples using Qwen3-TTS or VibeVoice.
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
from pathlib import Path
from textwrap import dedent

from modules.core_components.tools.base import Tab, TabConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager


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
        
        # Get helper functions and directories
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        get_prompt_cache_path = shared_state['get_prompt_cache_path']
        get_or_create_voice_prompt = shared_state['get_or_create_voice_prompt']
        apply_emotion_preset = shared_state['apply_emotion_preset']
        refresh_samples = shared_state['refresh_samples']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        OUTPUT_DIR = shared_state['OUTPUT_DIR']
        play_completion_beep = shared_state.get('play_completion_beep')
        
        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()
        
        def generate_audio_handler(sample_name, text_to_generate, language, seed, model_selection="Qwen3 - Small",
                                   qwen_do_sample=True, qwen_temperature=0.9, qwen_top_k=50, qwen_top_p=1.0, qwen_repetition_penalty=1.05,
                                   qwen_max_new_tokens=2048,
                                   vv_do_sample=False, vv_temperature=1.0, vv_top_k=50, vv_top_p=1.0, vv_repetition_penalty=1.0,
                                   vv_cfg_scale=3.0, vv_num_steps=20, progress=gr.Progress()):
            """Generate audio using voice cloning - supports both Qwen and VibeVoice engines."""
            if not sample_name:
                return None, "❌ Please select a voice sample first."

            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate."

            # Parse model selection to determine engine and size
            if "VibeVoice" in model_selection:
                engine = "vibevoice"
                if "Small" in model_selection:
                    model_size = "1.5B"
                elif "4-bit" in model_selection:
                    model_size = "Large (4-bit)"
                else:  # Large
                    model_size = "Large"
            else:  # Qwen3
                engine = "qwen"
                if "Small" in model_selection:
                    model_size = "0.6B"
                else:  # Large
                    model_size = "1.7B"

            # Find the selected sample
            samples = get_available_samples()
            sample = None
            for s in samples:
                if s["name"] == sample_name:
                    sample = s
                    break

            if not sample:
                return None, f"❌ Sample '{sample_name}' not found."

            try:
                # Set the seed for reproducibility
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                seed_msg = f"Seed: {seed}"

                if engine == "qwen":
                    # Qwen engine - uses cached prompts
                    progress(0.1, desc=f"Loading Qwen3 model ({model_size})...")

                    # Get or create the voice prompt (with caching)
                    model = tts_manager.get_qwen3_custom_voice(model_size)
                    prompt_items, was_cached = get_or_create_voice_prompt(
                        model=model,
                        sample_name=sample_name,
                        wav_path=sample["wav_path"],
                        ref_text=sample["ref_text"],
                        model_size=model_size,
                        progress_callback=progress
                    )

                    cache_status = "cached" if was_cached else "newly processed"
                    progress(0.6, desc=f"Generating audio ({cache_status} prompt)...")

                    # Generate using manager method
                    audio_data, sr = tts_manager.generate_voice_clone_qwen(
                        text=text_to_generate,
                        language=language,
                        prompt_items=prompt_items,
                        seed=seed,
                        do_sample=qwen_do_sample,
                        temperature=qwen_temperature,
                        top_k=qwen_top_k,
                        top_p=qwen_top_p,
                        repetition_penalty=qwen_repetition_penalty,
                        max_new_tokens=qwen_max_new_tokens,
                        model_size=model_size
                    )
                    wavs = [audio_data]

                    engine_display = f"Qwen3-{model_size}"

                else:  # vibevoice engine
                    progress(0.1, desc=f"Loading VibeVoice model ({model_size})...")

                    # Generate using manager method
                    audio_data, sr = tts_manager.generate_voice_clone_vibevoice(
                        text=text_to_generate,
                        voice_sample_path=sample["wav_path"],
                        seed=seed,
                        do_sample=vv_do_sample,
                        temperature=vv_temperature,
                        top_k=vv_top_k,
                        top_p=vv_top_p,
                        repetition_penalty=vv_repetition_penalty,
                        cfg_scale=vv_cfg_scale,
                        num_steps=vv_num_steps,
                        model_size=model_size,
                        user_config=shared_state.get('_user_config', {})
                    )
                    wavs = [audio_data]

                    engine_display = f"VibeVoice-{model_size}"
                    cache_status = "no caching (VibeVoice)"

                progress(0.8, desc="Saving audio...")
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
                output_file = OUTPUT_DIR / f"{safe_name}_{timestamp}.wav"

                sf.write(str(output_file), wavs[0], sr)

                # Save metadata file
                metadata_file = output_file.with_suffix(".txt")
                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Sample: {sample_name}
                    Engine: {engine_display}
                    Language: {language}
                    Seed: {seed}
                    Text: {text_to_generate.strip()}
                    """)
                metadata_file.write_text(metadata, encoding="utf-8")

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                return str(output_file), f"Generated using {engine_display}. {cache_status}\n{seed_msg}"

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error generating audio: {str(e)}"
        
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
            generate_audio_handler,
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


if __name__ == "__main__":
    """Standalone testing of Voice Clone tool."""
    print("[*] Starting Voice Clone Tool - Standalone Mode")
    
    from pathlib import Path
    import sys
    import json
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Import constants and modals
    from modules.core_components.constants import (
        LANGUAGES,
        VOICE_CLONE_OPTIONS,
        DEFAULT_VOICE_CLONE_MODEL,
        QWEN_GENERATION_DEFAULTS,
        VIBEVOICE_GENERATION_DEFAULTS
    )
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
    from modules.core_components.tool_utils import (
        load_config,
        save_preference as save_pref_to_file,
        TRIGGER_HIDE_CSS
    )
    
    # Load config
    user_config = load_config()
    active_emotions = user_config.get('emotions', CORE_EMOTIONS)
    
    SAMPLES_DIR = project_root / "samples"
    OUTPUT_DIR = project_root / "output"
    SAMPLES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Helper: Get sample choices
    def get_sample_choices():
        samples = []
        for json_file in SAMPLES_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    samples.append(meta.get("name", json_file.stem))
            except:
                samples.append(json_file.stem)
        return samples if samples else ["(No samples found)"]
    
    def get_available_samples():
        samples = []
        for json_file in SAMPLES_DIR.glob("*.json"):
            wav_file = json_file.with_suffix(".wav")
            if wav_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
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
    
    def get_prompt_cache_path(sample_name, model_size):
        return project_root / "temp" / f"{sample_name}_{model_size}_prompt.pt"
    
    def get_or_create_voice_prompt(model, sample_name, wav_path, ref_text, model_size, progress_callback=None):
        # Mock - just return that it's not cached
        return None, False
    
    # Shared state with real modal support
    shared_state = {
        'LANGUAGES': LANGUAGES,
        'VOICE_CLONE_OPTIONS': VOICE_CLONE_OPTIONS,
        'DEFAULT_VOICE_CLONE_MODEL': DEFAULT_VOICE_CLONE_MODEL,
        '_user_config': user_config,
        '_active_emotions': active_emotions,
        'OUTPUT_DIR': OUTPUT_DIR,
        'SAMPLES_DIR': SAMPLES_DIR,
        'get_sample_choices': get_sample_choices,
        'get_available_samples': get_available_samples,
        'get_prompt_cache_path': get_prompt_cache_path,
        'get_or_create_voice_prompt': get_or_create_voice_prompt,
        'apply_emotion_preset': lambda e, i: (0.9, 1.0, 1.05, i),
        'refresh_samples': lambda: gr.update(choices=get_sample_choices()),
        'show_input_modal_js': show_input_modal_js,
        'show_confirmation_modal_js': show_confirmation_modal_js,
        'save_emotion_handler': lambda name, intensity, temp, rep_pen, top_p: handle_save_emotion(name, intensity, temp, rep_pen, top_p, user_config, active_emotions),
        'delete_emotion_handler': lambda confirm_val, emotion_name: handle_delete_emotion(confirm_val, emotion_name, user_config, active_emotions),
        'save_preference': lambda k, v: save_pref_to_file(user_config, k, v),
        'play_completion_beep': lambda: print("[Beep] Complete!"),
        'confirm_trigger': None,
        'input_trigger': None,
    }
    
    from modules.core_components.ui_components import create_qwen_advanced_params, create_vibevoice_advanced_params
    shared_state['create_qwen_advanced_params'] = create_qwen_advanced_params
    shared_state['create_vibevoice_advanced_params'] = create_vibevoice_advanced_params
    
    print(f"[*] Samples: {SAMPLES_DIR}")
    print(f"[*] Output: {OUTPUT_DIR}")
    print(f"[*] Found {len(get_available_samples())} samples")
    
    # Load custom theme
    theme = gr.themes.Base.load('modules/core_components/theme.json')
    
    with gr.Blocks(title="Voice Clone - Standalone", head=CONFIRMATION_MODAL_CSS + INPUT_MODAL_CSS, css=TRIGGER_HIDE_CSS) as app:
        # Add modal HTML
        gr.HTML(CONFIRMATION_MODAL_HTML)
        gr.HTML(INPUT_MODAL_HTML)
        
        gr.Markdown("# 🎤 Voice Clone Tool (Standalone Testing)")
        gr.Markdown("*Standalone mode with full modal support*")
        
        # Hidden trigger widgets - visible but hidden via CSS
        with gr.Row():
            confirm_trigger = gr.Textbox(label="Confirm Trigger", value="", elem_id="confirm-trigger")
            input_trigger = gr.Textbox(label="Input Trigger", value="", elem_id="input-trigger")
        shared_state['confirm_trigger'] = confirm_trigger
        shared_state['input_trigger'] = input_trigger
        
        components = VoiceCloneTab.create_tab(shared_state)
        VoiceCloneTab.setup_events(components, shared_state)
    
    print("[*] Launching on http://127.0.0.1:7862")
    app.launch(theme=theme, server_port=7862, server_name="127.0.0.1", share=False, inbrowser=True)
