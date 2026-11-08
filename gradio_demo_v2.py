#!/usr/bin/env python3
"""
OpenVoice V2 Gradio Demo
A simple web interface for OpenVoice V2 voice cloning
"""

import os
import torch
import gradio as gr
import tempfile
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
ckpt_converter = 'checkpoints_v2/checkpoints_v2/converter'
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)

print(f"Loading OpenVoice V2 on device: {device}")

# Load ToneColorConverter
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

print("✅ ToneColorConverter loaded successfully!")

def clone_voice(reference_audio, text_input, language="EN_NEWEST", speed=1.0):
    """
    Clone voice using OpenVoice V2
    """
    try:
        # Import MeloTTS here to avoid initialization issues
        from melo.api import TTS

        if reference_audio is None:
            return "❌ Please upload a reference audio file", None

        if not text_input.strip():
            return "❌ Please enter some text to synthesize", None

        # Extract target speaker embedding from reference audio
        print(f"Extracting speaker embedding from: {reference_audio}")
        target_se, audio_name = se_extractor.get_se(reference_audio, tone_color_converter, vad=True)

        # Initialize MeloTTS model for the specified language
        print(f"Loading MeloTTS model for language: {language}")
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id

        # Use the first available speaker
        speaker_key = list(speaker_ids.keys())[0]
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')

        # Load source speaker embedding
        source_se_path = f'checkpoints_v2/checkpoints_v2/base_speakers/ses/{speaker_key}.pth'
        if not os.path.exists(source_se_path):
            return f"❌ Source speaker embedding not found: {source_se_path}", None

        source_se = torch.load(source_se_path, map_location=device)

        # Generate speech with MeloTTS
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            src_path = tmp_file.name

        print(f"Generating speech with MeloTTS...")
        model.tts_to_file(text_input, speaker_id, src_path, speed=speed)

        # Convert tone color
        output_path = os.path.join(output_dir, f'cloned_voice_{audio_name}.wav')
        print(f"Converting tone color...")

        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            message="@OpenVoice"
        )

        # Clean up temporary file
        os.unlink(src_path)

        success_msg = f"✅ Voice cloning successful!\nOutput saved to: {output_path}"
        return success_msg, output_path

    except Exception as e:
        error_msg = f"❌ Error during voice cloning: {str(e)}"
        print(error_msg)
        return error_msg, None

# Create Gradio interface
def create_demo():
    with gr.Blocks(title="OpenVoice V2 Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎤 OpenVoice V2 Voice Cloning Demo

        Upload a reference audio file and enter text to clone the voice!

        **Supported Languages:** English, Spanish, French, Chinese, Korean

        **Note:** Japanese support may be limited due to installation constraints.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 Input")

                reference_audio = gr.Audio(
                    label="Reference Audio (Upload your voice sample)",
                    type="filepath",
                    format="wav"
                )

                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter the text you want to speak in the cloned voice...",
                    lines=3,
                    max_lines=10
                )

                language = gr.Dropdown(
                    choices=["EN_NEWEST", "EN", "ES", "FR", "ZH", "KR"],
                    value="EN_NEWEST",
                    label="Language",
                    info="Select the language for text-to-speech"
                )

                speed = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Speech Speed",
                    info="Adjust the speed of generated speech"
                )

                clone_btn = gr.Button("🎯 Clone Voice", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("### 📥 Output")

                status_output = gr.Textbox(
                    label="Status",
                    lines=3,
                    max_lines=10,
                    interactive=False
                )

                audio_output = gr.Audio(
                    label="Cloned Voice Output",
                    type="filepath"
                )

        # Example section
        gr.Markdown("""
        ### 📋 Instructions:
        1. **Upload Reference Audio**: Choose a clear audio file (WAV/MP3) of the voice you want to clone
        2. **Enter Text**: Type the text you want the cloned voice to speak
        3. **Select Language**: Choose the appropriate language for your text
        4. **Adjust Speed**: Fine-tune the speech speed if needed
        5. **Click Clone Voice**: Generate the cloned voice audio

        ### 🎯 Tips for Best Results:
        - Use clear, high-quality reference audio (at least 3-10 seconds)
        - Avoid background noise in reference audio
        - Keep text length reasonable (under 200 words for best performance)
        - Try different languages and speeds to find what works best
        """)

        # Connect the button to the function
        clone_btn.click(
            fn=clone_voice,
            inputs=[reference_audio, text_input, language, speed],
            outputs=[status_output, audio_output]
        )

    return demo

if __name__ == "__main__":
    print("🚀 Starting OpenVoice V2 Gradio Demo...")
    demo = create_demo()

    # Launch the demo
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=True,  # Creates a public link
        show_error=True,
        quiet=False
    )
