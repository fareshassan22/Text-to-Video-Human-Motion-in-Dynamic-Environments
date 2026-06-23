
# Bonus Text-to-Audio Module

The optional bonus component adds synchronized narration to the final generated video. It uses the neural SpeechT5 text-to-speech model rather than a basic TTS library.

## Implemented advanced features

1. Neural TTS model: `microsoft/speecht5_tts` with `microsoft/speecht5_hifigan`.
2. Multi-voice support: speaker embeddings from `Matthijs/cmu-arctic-xvectors`.
3. Emotion/style control: neutral, happy, and calm presets modify speed, pitch, and pause length.
4. Context-aware synthesis: the narration text is split using punctuation and natural pauses are inserted between chunks.
5. Video integration: ffmpeg mixes the generated waveform with the final enhanced video.

## Selected configuration

- Voice preset: `voice_1`
- Style preset: `neutral`
- Audio duration: `6.09` seconds
- Output audio: `/kaggle/working/phase2_results/bonus_text_to_audio/phase2_neural_narration.wav`
- Final video with audio: `/kaggle/working/phase2_results/bonus_text_to_audio/final_video_with_neural_tts.mp4`

This module improves storytelling quality by adding a concise narration that explains the generated scene and the temporal stabilization enhancement.
