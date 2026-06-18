import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import math
from torch.nn.utils import weight_norm
from .utils.build_vocab import Vocab
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .utils import Utility as utility
from .wav2vec import Wav2Vec2Model, Wav2Vec2ForSpeechClassification
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from .vqvae_modules import VectorQuantizerEMA, ConvNormRelu, Res_CNR_Stack
import time
cur_time = time.time()
os.environ['CUDA_LAUNCH_BLOCKING']='1'

class WavEncoder2(nn.Module):

    def __init__(self, num_hiddens=512, num_residual_layers=1):
        super().__init__()

        self.device = 'cuda:0'
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        content_model = os.environ.get(
            "PFMG_WAV2VEC2_MODEL",
            "jonatasgrosman/wav2vec2-large-xlsr-53-english",
        )
        emotion_model = os.environ.get(
            "PFMG_WAV2VEC2_EMOTION_MODEL",
            "r-f/wav2vec-english-speech-emotion-recognition",
        )
        self.audio_encoder_cont = Wav2Vec2Model.from_pretrained(content_model)
        self.processor = Wav2Vec2Processor.from_pretrained(content_model)
        self.audio_encoder_cont.feature_extractor._freeze_parameters()
        self.audio_encoder_emo = Wav2Vec2ForSpeechClassification.from_pretrained(emotion_model)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(emotion_model)
        self.audio_encoder_emo.wav2vec2.feature_extractor._freeze_parameters()
        self.audio_feature_map_cont = nn.Linear(1024, 512)
        self.audio_feature_map_emo = nn.Linear(1024, 832)
        self.audio_feature_map_emo2 = nn.Linear(832, 256)
        self.relu = nn.ReLU()

    def forward(self, wav_data, time_steps=34):

        time_steps = math.ceil(wav_data.shape[1] / 16000 * 15)
        inputs12 = self.processor(wav_data, sampling_rate=16000, return_tensors="pt",
                                  padding="longest").input_values.to(self.device)
        hidden_states_cont1 = self.audio_encoder_cont(inputs12, frame_num=time_steps).last_hidden_state
        inputs12 = self.feature_extractor(wav_data, sampling_rate=16000, padding=True,
                                          return_tensors="pt").input_values.to(self.device)
        output_emo1 = self.audio_encoder_emo(inputs12, frame_num=time_steps)
        hidden_states_emo1 = output_emo1.hidden_states

        hidden_states_cont1 = self.audio_feature_map_cont(hidden_states_cont1)
        hidden_states_emo11_832 = self.audio_feature_map_emo(hidden_states_emo1)
        hidden_states_emo11_256 = self.relu(
            self.audio_feature_map_emo2(hidden_states_emo11_832))

        return hidden_states_cont1, hidden_states_emo11_256
    
class audio2face(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pre_length = args.pre_frames 
        self.gen_length = args.pose_length - args.pre_frames
        self.pose_dims = args.pose_dims
        self.facial_f = args.facial_f
        self.speaker_f = args.speaker_f
        self.audio_f = args.audio_f
        # self.word_f = args.word_f
        self.emotion_f = args.emotion_f
        self.facial_dims = args.facial_dims
        self.speaker_dims = args.speaker_dims
        self.emotion_dims = args.emotion_dims
        self.in_size = self.audio_f
        self.audio_encoder = WavEncoder2(self.audio_f)
        self.hidden_size = args.hidden_size
        self.n_layer = args.n_layer

        num_hiddens = 256
        num_residual_layers = 1
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers

        self._enc_1 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)
        self._down_1 = ConvNormRelu(self._num_hiddens, self._num_hiddens, leaky=True, residual=True, sample='none')
        self._enc_2 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)
        self._down_2 = ConvNormRelu(self._num_hiddens, self._num_hiddens, leaky=True, residual=True, sample='none')
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)

        self.speaker_embedding = None
        if self.speaker_f != 0:
            self.in_size += self.speaker_f
            self.speaker_embedding = nn.Sequential(
                nn.Embedding(self.speaker_dims, self.speaker_f),
                nn.Linear(self.speaker_f, self.speaker_f), 
                nn.LeakyReLU(True)
            )

        self.emotion_embedding = None
        if self.emotion_f != 0:
            self.in_size += self.emotion_f
            
            self.emotion_embedding = nn.Sequential(
                nn.Embedding(self.emotion_dims, self.emotion_f),
                nn.Linear(self.emotion_f, self.emotion_f) 
            )

            self.emotion_embedding_tail = nn.Sequential( 
                nn.Conv1d(self.emotion_f, 8, 9, 1, 4),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(8, 16, 9, 1, 4),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(16, 16, 9, 1, 4),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(16, self.emotion_f, 9, 1, 4),
                nn.BatchNorm1d(self.emotion_f),
                nn.LeakyReLU(0.3, inplace=True),
            )
        
        self.LSTM = nn.LSTM(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layer, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)

        extral_size = 8
        self.lstm_fc_eye = nn.Sequential(
            nn.Linear(self.hidden_size + extral_size,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 14))

        self.lstm_fc_exp = nn.Sequential(
            nn.Linear(self.hidden_size + extral_size,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 14))

        self.lstm_fc_mouth = nn.Sequential(
            nn.Linear(self.hidden_size + extral_size,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 23))

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True
            
    def forward(self, in_audio=None, in_text=None, in_id=None, in_emo=None, is_test=False):

        if self.do_flatten_parameters:
            self.LSTM.flatten_parameters()

        text_feat_seq = audio_feat_seq = None
        if in_audio is not None:
            hidden_states_cont1, hidden_states_emo11_256 = self.audio_encoder(in_audio)

        speaker_feat_seq = None
        if self.speaker_embedding: 
            speaker_feat_seq = self.speaker_embedding(in_id)

        emo_feat_seq = None
        if self.emotion_embedding:
            emo_feat_seq = self.emotion_embedding(in_emo)
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
            emo_feat_seq = self.emotion_embedding_tail(emo_feat_seq) 
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
        
        in_data = torch.cat(
            [hidden_states_cont1, hidden_states_emo11_256], dim=2)

        if emo_feat_seq is not None:
            in_data = torch.cat((in_data, emo_feat_seq), dim=2)
        
        if speaker_feat_seq is not None:
            repeated_s = speaker_feat_seq
            if len(repeated_s.shape) == 2:
                repeated_s = repeated_s.reshape(1, repeated_s.shape[1], repeated_s.shape[0])
            repeated_s = repeated_s.repeat(1, in_data.shape[1], 1)
            in_data = torch.cat((in_data, repeated_s), dim=2)

        output, _ = self.LSTM(in_data)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        output = output.transpose(1, 2)
        h = self._enc_1(output)
        h = self._down_1(h)
        h = self._enc_2(h)
        h = self._down_2(h)
        output = self._enc_3(h)
        output = output.transpose(1, 2)
        # output, _ = self.LSTM(in_data)
        # output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        # face_feature = output 
        # output = self.out(output.reshape(-1, output.shape[2]))
        # output = torch.cat((output, text_feat_seq), dim=2)
        output = torch.cat((output, emo_feat_seq), dim=2)

        fc_out  = []
        for step_t in range(in_data.size(1)):

            fc_in = output[:,step_t,:]
            # print('111', fc_in.shape)
            eye = self.lstm_fc_eye(fc_in)
            exp = self.lstm_fc_exp(fc_in)
            mouth = self.lstm_fc_mouth(fc_in)
            aa = torch.cat([exp[:, :8], eye, exp[:, 8:-2], mouth, exp[:, -2:]],dim=1)
            fc_out.append(aa)
            
        output = torch.stack(fc_out, dim = 1)
        decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)
        face_feature = decoder_outputs
        # print(output.shape, decoder_outputs.shape)
        return decoder_outputs, hidden_states_cont1, hidden_states_emo11_256
    
class ConvDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.facial_dims

        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(self.input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            nn.Conv1d(8, 8, 3),
        )

        self.LSTM = nn.LSTM(8, hidden_size=self.hidden_size, num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(34-6, 1)
       
        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True
            
        decoder_hidden = None
        
    def forward(self, faces):
        if self.do_flatten_parameters:
            self.LSTM.flatten_parameters()
        faces = faces.transpose(1, 2)
        feat = self.pre_conv(faces)
        feat = feat.transpose(1, 2)
        output, _ = self.LSTM(feat)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  
        batch_size = faces.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)
        return output