import torch
import torch.nn as nn

class VGG19_10(nn.Module):
    def __init__(self, inp=3):
        super(VGG19_10, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(inp,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),

            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),

            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
        )
        # self._initialize_pretrained()
    def forward(self, x):
        x = self.model(x)
        return x
    def _initialize_pretrained(self):
        vgg_state_dict = torch.load('vgg19-dcbb9e9d.pth')
        vgg_keys = vgg_state_dict.keys()
        weights_load = {}
        for i in range(20):
            weights_load[list(self.model.state_dict().keys())[i]] = vgg_state_dict[list(vgg_keys)[i]]
        state = self.model.state_dict()
        state.update(weights_load)
        self.model.load_state_dict(state)

class Generator_SL_0(nn.Module):
    def __init__(self, inp=128, keypoints=18, limbs=36):
        super(Generator_SL_0, self).__init__()
        self.model_s = nn.Sequential(
            nn.Conv2d(inp,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,512,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,keypoints,kernel_size=1,stride=1,padding=0),
        )
        self.model_l = nn.Sequential(
            nn.Conv2d(inp,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,512,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,limbs,kernel_size=1,stride=1,padding=0),
        )
    def forward(self, f):
        s = self.model_s(f)
        l = self.model_l(f)
        return s,l

class Generator_SL_Refine(nn.Module):
    def __init__(self, inp=185, keypoints=18, limbs=36):
        super(Generator_SL_Refine, self).__init__()
        self.model_s = nn.Sequential(
            nn.Conv2d(inp,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,keypoints,kernel_size=1,stride=1,padding=0),
        )
        self.model_l = nn.Sequential(
            nn.Conv2d(inp,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,limbs,kernel_size=1,stride=1,padding=0),
        )
    def forward(self, f_new,s_new,l_new,s,l):
        x = torch.cat([f_new,s_new,l_new,s,l],dim=1)
        s_new = self.model_s(x)
        l_new = self.model_l(x)
        return s_new, l_new


class Generator_SL_Refine_Single(nn.Module):
    def __init__(self, inp=185, keypoints=18, limbs=36):
        super(Generator_SL_Refine_Single, self).__init__()
        self.model_s = nn.Sequential(
            nn.Conv2d(inp,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,keypoints,kernel_size=1,stride=1,padding=0),
        )
        self.model_l = nn.Sequential(
            nn.Conv2d(inp,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=7,stride=1,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,limbs,kernel_size=1,stride=1,padding=0),
        )
    def forward(self, f, s, l):
        x = torch.cat([f,s,l],dim=1)
        s = self.model_s(x)
        l = self.model_l(x)
        return s,l

class PoseModel(nn.Module):
    def __init__(self, inp=3, keypoints=18, limbs=36):
        super(PoseModel, self).__init__()
        self.model1 = VGG19_10(inp=inp)
        self.model_SL_0 = Generator_SL_0(inp=128, keypoints=keypoints, limbs=limbs)
        self.model_SL_R = Generator_SL_Refine(inp=128+keypoints*2+limbs*2, keypoints=keypoints, limbs=limbs)

    def forward(self, x_new, s, l, single_mode = False):
        f_new = self.model1(x_new)
        s_new, l_new = self.model_SL_0(f_new)
        if single_mode:
            s_new_improved, l_new_improved = self.model_SL_R(f_new,s_new,l_new,s_new,l_new)
        else:
            s_new_improved, l_new_improved = self.model_SL_R(f_new,s_new,l_new,s,l)
        # if s == None and l == None:
        #     s_new_improved, l_new_improved = self.model_SL(f_new,s_new,l_new,s_new,l_new)
        # else: 
            # s_new_improved, l_new_improved = self.model_SL(f_new,s_new,l_new,s,l)
        return s_new, l_new, s_new_improved, l_new_improved

class PoseModelDeep(nn.Module):
    def __init__(self, inp=3, keypoints=18, limbs=36):
        super(PoseModelDeep, self).__init__()
        self.model1 = VGG19_10(inp=inp)


        self.model_SL_0 = Generator_SL_0(inp=128, keypoints=keypoints, limbs=limbs)

        self.model_SL_S1 = Generator_SL_Refine_Single(inp=128+keypoints+limbs, keypoints=keypoints, limbs=limbs)
        self.model_SL_S2 = Generator_SL_Refine_Single(inp=128+keypoints+limbs, keypoints=keypoints, limbs=limbs)
        self.model_SL_S3 = Generator_SL_Refine_Single(inp=128+keypoints+limbs, keypoints=keypoints, limbs=limbs)

        self.model_SL_R1 = Generator_SL_Refine(inp=128+keypoints*2+limbs*2, keypoints=keypoints, limbs=limbs)
        self.model_SL_R2 = Generator_SL_Refine(inp=128+keypoints*2+limbs*2, keypoints=keypoints, limbs=limbs)


    def forward(self, x_new, s, l, single_mode = False):
        f_new = self.model1(x_new)

        output_s = []
        output_l = []
        s_new, l_new = self.model_SL_0(f_new)
        output_s.append(s_new)
        output_l.append(l_new)
        s_new, l_new = self.model_SL_S1(f_new, s_new, l_new)
        output_s.append(s_new)
        output_l.append(l_new)
        s_new, l_new = self.model_SL_S2(f_new, s_new, l_new)
        output_s.append(s_new)
        output_l.append(l_new)
        s_new, l_new = self.model_SL_S3(f_new, s_new, l_new)
        output_s.append(s_new)
        output_l.append(l_new)

        if single_mode:
            s_new, l_new = self.model_SL_R1(f_new, s_new, l_new, s_new, l_new)
            output_s.append(s_new)
            output_l.append(l_new)
            s_new, l_new = self.model_SL_R1(f_new, s_new, l_new, s_new, l_new)
            output_s.append(s_new)
            output_l.append(l_new)
            return output_s, output_l
        else:
            s_new, l_new = self.model_SL_R1(f_new, s_new, l_new, s, l)
            output_s.append(s_new)
            output_l.append(l_new)
            s_new, l_new = self.model_SL_R1(f_new, s_new, l_new, s, l)
            output_s.append(s_new)
            output_l.append(l_new)

            return output_s, output_l