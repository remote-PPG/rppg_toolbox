import torch
from torch import nn




class LSTCrPPG(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_block = EncoderBlock()
        self.decoder_block = DecoderBlock()

    def forward(self, x):
        [B,C,T,W,H] = x.shape
        e = self.encoder_block.forward(x)
        out = self.decoder_block.forward(e)
        return out.view(B,T)
class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        #, in_channel, out_channel, kernel_size, stride, padding
        self.encoder_block1 = nn.Sequential(
            ConvBlock3D(3, 16, [3,3,3], [1,1,1], [1,1,1]),
            ConvBlock3D(16, 16, [3,3,3], [1,1,1], [1,1,1]),
            nn.BatchNorm3d(16)
        )
        self.encoder_block2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            nn.BatchNorm3d(16)
        )
        self.encoder_block3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            nn.BatchNorm3d(32)
        )
        self.encoder_block4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            nn.BatchNorm3d(32)
        )
        self.encoder_block5 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            nn.BatchNorm3d(64)
        )
        self.encoder_block6 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            nn.BatchNorm3d(64)
        )
        self.encoder_block7 = nn.Sequential(
            ConvBlock3D(64, 64, [5, 3, 3], [1, 1, 1], [0,1,1]),
            nn.BatchNorm3d(64)
        )

    def forward(self, x):
        e1 = self.encoder_block1(x)
        e2 = self.encoder_block2(e1)
        e3 = self.encoder_block3(e2)
        e4 = self.encoder_block4(e3)
        e5 = self.encoder_block5(e4)
        e6 = self.encoder_block6(e5)
        e7 = self.encoder_block7(e6)
        return [e7,e6,e5,e4,e3,e2,e1]

class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.decoder_block6_transpose = nn.ConvTranspose3d(64,64,[5,1,1],[1,1,1])
        self.decoder_block6 = nn.Sequential(
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1,1,1]),
            nn.BatchNorm3d(64)
        )
        self.decoder_block5_transpose =nn.ConvTranspose3d(64, 64, [4, 1, 1],[2,1,1])
        self.decoder_block5 = nn.Sequential(
            ConvBlock3D(64, 32, [3, 3, 3], [1, 1, 1],[1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1],[0,1,1]),
            nn.BatchNorm3d(32)
        )
        self.decoder_block4_transpose = nn.ConvTranspose3d(32, 32, [4, 1, 1],[2,1,1])
        self.decoder_block4 = nn.Sequential(
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(32, 32, [3, 3, 3], [1, 1, 1], [0,1,1]),
            nn.BatchNorm3d(32)
        )
        self.decoder_block3_transpose = nn.ConvTranspose3d(32, 32, [4, 1, 1],[2,1,1])
        self.decoder_block3 = nn.Sequential(
            ConvBlock3D(32, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [0,1,1]),
            nn.BatchNorm3d(16)
        )
        self.decoder_block2_transpose = nn.ConvTranspose3d(16, 16, [4, 1, 1],[2,1,1])
        self.decoder_block2 = nn.Sequential(
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(16, 16, [3, 3, 3], [1, 1, 1], [0,1,1]),
            nn.BatchNorm3d(16)
        )
        self.decoder_block1_transpose = nn.ConvTranspose3d(16, 16, [4, 1, 1],[2,1,1])
        self.decoder_block1 = nn.Sequential(
            ConvBlock3D(16, 3, [3, 3, 3], [1, 1, 1], [1,1,1]),
            ConvBlock3D(3, 3, [3, 3, 3], [1, 1, 1], [0,1,1]),
            nn.BatchNorm3d(3)
        )
        self.predictor = nn.Conv3d(3, 1 ,[1,4,4])



    def forward(self, encoded_features):
        encoded_feature_0,encoded_feature_1,encoded_feature_2,encoded_feature_3,\
            encoded_feature_4,encoded_feature_5,encoded_feature_6 = encoded_features


        d = self.decoder_block6_transpose(encoded_feature_0)
        d = self.TARM(encoded_feature_1, d)
        d6 = self.decoder_block6(d)
        d5 = self.decoder_block5(self.TARM(encoded_feature_2,self.decoder_block5_transpose(d6)))
        d4 = self.decoder_block4(self.TARM(encoded_feature_3,self.decoder_block4_transpose(d5)))
        d3 = self.decoder_block3(self.TARM(encoded_feature_4,self.decoder_block3_transpose(d4)))
        d2 = self.decoder_block2(self.TARM(encoded_feature_5,self.decoder_block2_transpose(d3)))
        d1 = self.decoder_block1(self.TARM(encoded_feature_6,self.decoder_block1_transpose(d2)))

        # d6 = self.decoder_block6(self.decoder_block6_transpose(encoded_feature_0))
        # d5 = self.decoder_block5(self.decoder_block5_transpose(d6))
        # d4 = self.decoder_block4(self.decoder_block4_transpose(d5))
        # d3 = self.decoder_block3(self.decoder_block3_transpose(d4))
        # d2 = self.decoder_block2(self.decoder_block2_transpose(d3))
        # d1 = self.decoder_block1(self.decoder_block1_transpose(d2))


        predictor = self.predictor(d1)
        # return torch.sigmoid(predictor)
        return predictor

    def TARM(self, e,d):
        target = d
        shape = d.shape #d: B C T W H
        e = nn.functional.adaptive_avg_pool3d(e,d.shape[2:])
        # e: B C T W H
        e = e.view(e.shape[0],e.shape[1], shape[2],-1)
        # e: B C T W*H
        d = d.view(d.shape[0], shape[1], shape[2], -1)
        # d: B C T W*H
        temporal_attention_map = e @ torch.transpose(d,3,2)
        temporal_attention_map = nn.functional.softmax(temporal_attention_map,dim=-1)
        refined_map = temporal_attention_map@e
        out = torch.reshape(refined_map,shape)
        return out

class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.ELU()
        )

    def forward(self, x):
        return self.conv_block_3d(x)