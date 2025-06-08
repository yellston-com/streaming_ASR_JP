

# Streaming ASR with Nvidia Nemo QuartzNet

A minimal real-time Japanese automatic speech recognition (ASR) application that captures microphone audio and outputs transcriptions to the console.

## Repository Structure

- `buffered_jp_quartznet.py`: Main script for streaming ASR. Loads a pretrained QuartzNet model, configures audio preprocessing, and uses PyAudio for live capture.

## Model

- **Pretrained Model**: `japanese-quartznet-large.nemo` (QuartzNet15x5 architecture)
- **Restore Path**: `./quartznet_jp/japanese-quartznet-large.nemo` cloned from https://huggingface.co/kinouchi/japanese-quartznet-large

## Requirements

- **Python**: 3.10 or above
- **PyTorch**: 2.7.0
- **Nvidia Nemo Toolkit**:  2.3.1 or compatible
- **PyAudio**: For audio capture
- **NumPy**, **OmegaConf**

Install dependencies via pip:
```bash
pip install numpy pyaudio torch nemo_toolkit[all] omegaconf
```

## Usage

1. Ensure the `.nemo` model file is available at the restore path.
2. Run the script:
   ```bash
   python buffered_jp_quartznet.py
   ```
3. Select the desired microphone device when prompted.
4. Speak into the microphone — transcriptions will appear in real time in the console.

## Notes

- Tested on Ubuntu 20.04 with Python 3.10.
- Adjust `SAMPLE_RATE` or `CHUNK_SIZE` in the script if needed for different audio hardware.