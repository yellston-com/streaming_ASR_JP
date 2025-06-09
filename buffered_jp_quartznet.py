import numpy as np
import pyaudio as pa
import os, time
import argparse
import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import copy
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader

"""
Module for streaming Japanese ASR using Nvidia Nemo QuartzNet.
Encapsulated as ASRService with initialization and inference methods.
"""

class ASRService:
    """ASR service encapsulating model initialization and streaming inference."""
    def __init__(self, device=None, frame_len=2.0, frame_overlap=2.5, offset=4,
                 model_path='./quartznet_jp/japanese-quartznet-large.nemo'):
        # Hidden initialization parameters
        self.device = device
        self.frame_len = frame_len
        self.frame_overlap = frame_overlap
        self.offset = offset
        self.sample_rate = 16000
        self.model_path = model_path

        # Normalization constants
        self.normalization = {
            'fixed_mean': [
                -14.95827016, -12.71798736, -11.76067913, -10.83311182,
                -10.6746914,  -10.15163465, -10.05378331, -9.53918999,
                -9.41858904,  -9.23382904,  -9.46470918,  -9.56037,
                -9.57434245,  -9.47498732,  -9.7635205,   -10.08113074,
                -10.05454561, -9.81112681,  -9.68673603,  -9.83652977,
                -9.90046248,  -9.85404766,  -9.92560366,  -9.95440354,
                -10.17162966, -9.90102482,  -9.47471025,  -9.54416855,
                -10.07109475, -9.98249912,  -9.74359465,  -9.55632283,
                -9.23399915,  -9.36487649,  -9.81791084,  -9.56799225,
                -9.70630899,  -9.85148006,  -9.8594418,   -10.01378735,
                -9.98505315,  -9.62016094,  -10.342285,   -10.41070709,
                -10.10687659, -10.14536695, -10.30828702, -10.23542833,
                -10.88546868, -11.31723646, -11.46087382, -11.54877829,
                -11.62400934, -11.92190509, -12.14063815, -11.65130117,
                -11.58308531, -12.22214663, -12.42927197, -12.58039805,
                -13.10098969, -13.14345864, -13.31835645, -14.47345634
            ],
            'fixed_std': [
                3.81402054, 4.12647781, 4.05007065, 3.87790987,
                3.74721178, 3.68377423, 3.69344,    3.54001005,
                3.59530412, 3.63752368, 3.62826417, 3.56488469,
                3.53740577, 3.68313898, 3.67138151, 3.55707266,
                3.54919572, 3.55721289, 3.56723346, 3.46029304,
                3.44119672, 3.49030548, 3.39328435, 3.28244406,
                3.28001423, 3.26744937, 3.46692348, 3.35378948,
                2.96330901, 2.97663111, 3.04575148, 2.89717604,
                2.95659301, 2.90181116, 2.7111687,  2.93041291,
                2.86647897, 2.73473181, 2.71495654, 2.75543763,
                2.79174615, 2.96076456, 2.57376336, 2.68789782,
                2.90930817, 2.90412004, 2.76187531, 2.89905006,
                2.65896173, 2.81032176, 2.87769857, 2.84665271,
                2.80863137, 2.80707634, 2.83752184, 3.01914511,
                2.92046439, 2.78461139, 2.90034605, 2.94599508,
                2.99099718, 3.0167554,  3.04649716, 2.94116777
            ]
        }

        # Model initialization
        self.model = nemo_asr.models.EncDecCTCModel.restore_from(
            restore_path=self.model_path)
        cfg = copy.deepcopy(self.model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.normalize = self.normalization
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        OmegaConf.set_struct(cfg.preprocessor, True)
        self.model.preprocessor = self.model.from_config_dict(cfg.preprocessor)
        self.model.eval()
        self.model.to(self.model.device)

        # Streaming decoder
        self.asr = FrameASR(
            model=self.model,
            model_definition={
                'sample_rate': self.sample_rate,
                'AudioToMelSpectrogramPreprocessor': cfg.preprocessor,
                'JasperEncoder': cfg.encoder,
                'labels': cfg.decoder.vocabulary
            },
            frame_len=self.frame_len,
            frame_overlap=self.frame_overlap,
            offset=self.offset
        )
        self.asr.reset()

    def start_stream(self):
        """Start PyAudio streaming and perform ASR."""
        p = pa.PyAudio()
        # List devices
        devices = [i for i in range(p.get_device_count())
                   if p.get_device_info_by_index(i).get('maxInputChannels') > 0]
        for i in devices:
            print(i, p.get_device_info_by_index(i).get('name'))
        dev_idx = self.device if self.device in devices else None
        empty_counter = 0

        def callback(in_data, frame_count, time_info, status):
            nonlocal empty_counter
            frame = np.frombuffer(in_data, dtype=np.int16)
            text = self.asr.transcribe(frame)
            if text:
                print(text, end='', flush=True)
                empty_counter = self.offset
            elif empty_counter > 0:
                empty_counter -= 1
                if empty_counter == 0:
                    print(' ', end='', flush=True)
            return in_data, pa.paContinue

        stream = p.open(format=pa.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=dev_idx,
                        frames_per_buffer=int(self.frame_len * self.sample_rate),
                        stream_callback=callback)
        print("Listening...")
        stream.start_stream()
        try:
            while stream.is_active():
                time.sleep(0.02)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("\nStream stopped")

# Data layer: wraps raw audio into a single-instance IterableDataset for inference
# simple data layer to pass audio signal
class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)/32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1

    def collate_fn(self, batch):
        return batch[0]

data_layer = AudioDataLayer(sample_rate=16000)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

# inference method for audio signal (single instance)
def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    # Ensure batch dimension for model input
    if audio_signal.ndim == 1:
        audio_signal = audio_signal.unsqueeze(0)
    if audio_signal_len.ndim == 0:
        audio_signal_len = audio_signal_len.unsqueeze(0)
    audio_signal, audio_signal_len = audio_signal.to(model.device), audio_signal_len.to(model.device)
    log_probs, encoded_len, predictions = model.forward(
        input_signal=audio_signal, input_signal_length=audio_signal_len
    )
    return log_probs

# Streaming ASR class: processes audio frames and performs incremental decoding
# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
class FrameASR:
    
    def __init__(self, model, model_definition,
                 frame_len=2, frame_overlap=2.5, 
                 offset=10):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        # Store reference to the NeMo model
        self.model = model
        
        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')
        
        self.sr = model_definition['sample_rate']
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMelSpectrogramPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()
        
    def _decode(self, frame, offset=0):
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = infer_signal(self.model, self.buffer).cpu().numpy()[0]
        # print(logits.shape)
        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap:-self.n_timesteps_overlap], 
            self.vocab
        )
        return decoded[:len(decoded)-offset]
    
    @torch.no_grad()
    def transcribe(self, frame=None, merge=True):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        if not merge:
            return unmerged
        return self.greedy_merge(unmerged)
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s

    def greedy_merge(self, s):
        s_merged = ''
        
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
        return s_merged
    
def main():
    service = ASRService()
    service.start_stream()

if __name__ == "__main__":
    main()