import torch
import torch.nn as nn
import os
import pickle
import numpy as np
from torch.nn.utils import weight_norm
from .utils.build_vocab import Vocab
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .utils import Utility as utility
from .audio2face import audio2face
from .motion_vqvae import VQVAE
from collections import OrderedDict
from loguru import logger
from .pfmg_vqvae import PfMG_VQ


os.environ['CUDA_LAUNCH_BLOCKING']='1'

def load_checkpoints(model, save_path, load_name='model'):
    states = torch.load(save_path)
    new_weights = OrderedDict()

    flag=False
    for k, v in states['model_state'].items():
        if "module" not in k:
            break
        else:
            new_weights[k[7:]]=v
            flag=True
    if flag: 
        model.load_state_dict(new_weights, strict=True)
    else:
        model.load_state_dict(states['model_state'])
    logger.info(f"load self-pretrained checkpoints for {load_name}")

class Model(torch.nn.Module):
    def __init__(self, gating_indices, gating_input, gating_hidden, gating_output, main_indices, main_input, main_hidden, main_output, dropout, input_norm=None, output_norm=None):
        super(Model, self).__init__()

        # if len(gating_indices) + len(main_indices) != len(input_norm[0]):
        #     print("Warning: Number of gating features (" + str(len(gating_indices)) + ") and main features (" + str(len(main_indices)) + ") are not the same as input features (" + str(len(input_norm[0])) + ").")

        self.gating_indices = gating_indices
        self.main_indices = main_indices

        self.G1 = nn.Linear(gating_input, gating_hidden)
        self.G2 = nn.Linear(gating_hidden, gating_hidden)
        self.G3 = nn.Linear(gating_hidden, gating_output)

        self.E1 = ExpertLinear(gating_output, main_input, main_hidden)
        self.E2 = ExpertLinear(gating_output, main_hidden, main_hidden)
        self.E3 = ExpertLinear(gating_output, main_hidden, main_output)

        self.dropout = dropout
        # self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)
        # self.Ynorm = Parameter(torch.from_numpy(output_norm), requires_grad=False)

    def forward(self, x):
        # x = utility.Normalize(x, self.Xnorm)

        #Gating
        # print(x.shape, self.gating_indices[-1])
        g = x[:, :, self.gating_indices]
        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G1(g)
        g = F.elu(g)

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G2(g)
        g = F.elu(g)

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G3(g)

        w = F.softmax(g, dim=1)
        #Main
        m = x[:, :, self.main_indices]

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E1(m, w)
        m = F.elu(m)

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E2(m , w)
        m = F.elu(m)

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E3(m, w)

        # return utility.Renormalize(m, self.Ynorm), w
        return m, w
    
class ExpertLinear(torch.nn.Module):
    def __init__(self, experts, input_dim, output_dim):
        super(ExpertLinear, self).__init__()

        self.experts = experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = self.weights([experts, input_dim, output_dim])
        self.b = self.bias([experts, 1, output_dim])

    def forward(self, x, weights):
        y = torch.zeros((x.shape[0], x.shape[1], self.output_dim), device=x.device, requires_grad=True)
        for i in range(self.experts):
            y = y + weights[:, :, i].unsqueeze(2) * (x.matmul(self.W[i,:,:]) + self.b[i,:,:])
        return y

    def weights(self, shape):
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(np.random.uniform(low=-alpha_bound, high=alpha_bound, size=shape), dtype=np.float32)
        return Parameter(torch.from_numpy(alpha), requires_grad=True)

    def bias(self, shape):
        return Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            #print(pre_trained_embedding.shape)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layer
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], args.word_f)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0


class BasicBlock(nn.Module):
    """ based on timm: https://github.com/rwightman/pytorch-image-models """
    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU,   norm_layer=nn.BatchNorm1d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=ker_size, stride=stride, padding=first_dilation,
            dilation=dilation, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size//2, dilation=dilation, bias=True)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,  stride=stride, kernel_size=ker_size, padding=first_dilation, dilation=dilation, bias=True),
                norm_layer(planes), 
            )
        else: self.downsample=None
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x


class WavEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(1, 32, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(32, 32, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(32, 32, 15, 1, first_dilation=7, ),
                BasicBlock(32, 64, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(64, 64, 15, 1, first_dilation=7),
                BasicBlock(64, 128, 15, 6,  first_dilation=0,downsample=True),     
            )
        
    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1) 
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2) 


class PoseGenerator(nn.Module):
    """
    End2End model
    audio, text and speaker ID encoder are customized based on Yoon et al. SIGGRAPH ASIA 2020
    """
    def __init__(self, args):
        super().__init__()
        self.pre_length = args.pre_frames 
        self.gen_length = args.pose_length - args.pre_frames
        self.pose_dims = args.pose_dims
        self.facial_f = args.facial_f
        self.speaker_f = args.speaker_f
        self.audio_f = args.audio_f
        self.word_f = args.word_f
        self.emotion_f = args.emotion_f
        # self.mfcc_f = args.mfcc_f
        self.facial_dims = args.facial_dims
        self.speaker_dims = args.speaker_dims
        self.emotion_dims = args.emotion_dims
        self.in_size = self.audio_f + self.pose_dims + self.facial_f + self.word_f + 1

        self.vqvae_pose = PfMG_VQ(args)
        model_path = args.root_path + '/data/beat_cache/beat_4english_15_141/weights/vqvae.bin'
        load_checkpoints(self.vqvae_pose, model_path, 'vqvae')
        for param in self.vqvae_pose.parameters():
            param.requires_grad=False
        
        self.hidden_size = 256
        self.n_layer = args.n_layer
        
        gating_indices = torch.tensor([(617 + i) for i in range(10)])
        main_indices = torch.tensor([(0 + i) for i in range(617)])
        dropout = 0.3
        gating_hidden = 64
        main_hidden = 1024
        experts = 5
        output_dim = 27
        
        self.body_GNN = utility.ToDevice(Model(
                gating_indices=gating_indices, 
                gating_input=len(gating_indices), 
                gating_hidden=gating_hidden, 
                gating_output=experts, 
                main_indices=main_indices, 
                main_input=len(main_indices), 
                main_hidden=main_hidden, 
                main_output=output_dim,
                dropout=dropout
            ))
        
        self.batch=256
        self.width=64

        self.LSTM = nn.LSTM(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layer, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)

        gating_indices = torch.tensor([(617 + i) for i in range(10)])
        main_indices = torch.tensor([(0 + i) for i in range(617)])
        dropout = 0.3
        gating_hidden = 64
        main_hidden = 1024
        experts = 5
        output_dim = 114
        
        self.hand_GNN = utility.ToDevice(Model(
                gating_indices=gating_indices, 
                gating_input=len(gating_indices), 
                gating_hidden=gating_hidden, 
                gating_output=experts, 
                main_indices=main_indices, 
                main_input=len(main_indices), 
                main_hidden=main_hidden, 
                main_output=output_dim,
                dropout=dropout
            ))

        self.LSTM_hands = nn.LSTM(self.in_size+27, hidden_size=self.hidden_size, num_layers=args.n_layer, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True
            

    def forward(self, pre_seq, in_audio=None, in_facial=None, in_text=None, in_id=None, in_emo=None, is_test=False):
        if self.do_flatten_parameters:
            self.LSTM.flatten_parameters()

        text_feat_seq = audio_feat_seq = None
        if in_audio is not None:
            audio_feat_seq = self.audio_encoder(in_audio) 
        if in_text is not None:
            text_feat_seq, _ = self.text_encoder(in_text)
            assert(audio_feat_seq.shape[1] == text_feat_seq.shape[1])
        
        if self.facial_f != 0:
            face_feat_seq = self.facial_encoder(in_facial.permute([0, 2, 1]))
            face_feat_seq = face_feat_seq.permute([0, 2, 1])

        speaker_feat_seq = None
        if self.speaker_embedding: 
            speaker_feat_seq = self.speaker_embedding(in_id)

        emo_feat_seq = None
        if self.emotion_embedding:
            emo_feat_seq = self.emotion_embedding(in_emo)
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
            emo_feat_seq = self.emotion_embedding_tail(emo_feat_seq) 
            emo_feat_seq = emo_feat_seq.permute([0,2,1])

        if  audio_feat_seq.shape[1] != pre_seq.shape[1]:
            diff_length = pre_seq.shape[1] - audio_feat_seq.shape[1]
            audio_feat_seq = torch.cat((audio_feat_seq, audio_feat_seq[:,-diff_length:, :].reshape(1,diff_length,-1)),1)
       
        if self.audio_f != 0 and self.facial_f == 0:
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        elif self.audio_f != 0 and self.facial_f != 0:
            in_data = torch.cat((pre_seq, audio_feat_seq, face_feat_seq), dim=2)
        else: pass
        
        if text_feat_seq is not None:
            in_data = torch.cat((in_data, text_feat_seq), dim=2)
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
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)
        return decoder_outputs
    

class PfMG_VQ_PAE(PoseGenerator):
    def __init__(self, args):
        super().__init__(args)
        self.args = args 
        self.audio_fusion_dim = self.audio_f+self.speaker_f+self.emotion_f+self.word_f
        self.facial_fusion_dim = self.audio_fusion_dim + self.facial_f
        self.audio_fusion = nn.Sequential(
            nn.Linear(self.audio_fusion_dim, self.hidden_size//2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size//2, self.audio_f),
            nn.LeakyReLU(True),
        )
        
        self.facial_fusion = nn.Sequential(
            nn.Linear(self.facial_fusion_dim, self.hidden_size//2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size//2, self.facial_f),
            nn.LeakyReLU(True),
        )
        
    def forward(self, pre_seq, in_audio=None, in_facial=None, in_pae=None, in_text=None, in_id=None, in_emo=None, in_pose=None):
        if self.do_flatten_parameters:
            self.LSTM.flatten_parameters()

        self.vqvae_pose.eval()
        decoder_outputs_iperiod, _, _, _, _, in_data= self.vqvae_pose(pre_seq, in_audio, in_facial, in_pae, in_text, in_id, in_emo, in_pose)

        hidden_data = torch.cat((decoder_outputs_iperiod, in_data), dim=2)

        decoder_outputs_period, _ = self.body_GNN(hidden_data)
        decoder_outputs_iperiod[:, :, 0:18] = decoder_outputs_iperiod[:, :, 0:18] + decoder_outputs_period[:, :, 0:18]
        decoder_outputs_iperiod[:, :, 75:84] = decoder_outputs_iperiod[:, :, 75:84] + decoder_outputs_period[:, :, 18:27]

        hidden_data = torch.cat((decoder_outputs_iperiod, in_data), dim=2)
        decoder_outputs_period, _ = self.hand_GNN(hidden_data)
        decoder_outputs_iperiod[:, :, 18:75] = decoder_outputs_iperiod[:, :, 18:75] + decoder_outputs_period[:, :, 0:57]
        decoder_outputs_iperiod[:, :, 84:141] = decoder_outputs_iperiod[:, :, 84:141] + decoder_outputs_period[:, :, 57:114]

        decoder_outputs_final = decoder_outputs_iperiod

        return decoder_outputs_final

    
class ConvDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.pose_dims

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

    def forward(self, poses):
        if self.do_flatten_parameters:
            self.LSTM.flatten_parameters()
        poses = poses.transpose(1, 2)
        feat = self.pre_conv(poses)
        feat = feat.transpose(1, 2)
        output, _ = self.LSTM(feat)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)
        return output