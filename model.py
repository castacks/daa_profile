import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

class HRNetw32Segmentation(nn.Module):
    def __init__(self, input_frames=2):
        super().__init__()
        base_model_name: str = 'hrnet_w32'
        input_depth: int = input_frames
        feature_location: str = ''
        self.combine_outputs_dim: int = 512
        self.upscale_mode: str = 'nearest'
        self.pred_scale: int = 8

        self.base_model = timm.create_model(base_model_name,
                                            features_only=True,
                                            feature_location='',
                                            out_indices=(1, 2, 3, 4),
                                            in_chans=input_depth,
                                            pretrained=False)

        self.combine_outputs_kernel = 1
        self.fc_comb = nn.Conv2d(480, self.combine_outputs_dim, kernel_size=self.combine_outputs_kernel)
        hrnet_outputs: int = self.combine_outputs_dim

        self.fc_cls = nn.Conv2d(hrnet_outputs, 7, kernel_size=1)
        self.fc_size = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_offset = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_distance = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_tracking = nn.Conv2d(hrnet_outputs, 2, kernel_size=1)
        self.fc_mask = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)
        self.fc_horizon = nn.Conv2d(hrnet_outputs, 1, kernel_size=1)

    def forward(self, inputs):
        stages_output = self.base_model(inputs)

        x = [
            F.avg_pool2d(stages_output[0], 2),
            stages_output[1],
            F.interpolate(stages_output[2], scale_factor=2.0, mode="nearest"),
            F.interpolate(stages_output[3], scale_factor=4.0, mode="nearest"),
        ]

        x = torch.cat(x, dim=1)
        x = F.relu(self.fc_comb(x))

        size = self.fc_size(x)
        offset = self.fc_offset(x)
        distance = self.fc_distance(x)
        tracking = self.fc_tracking(x)
        mask = self.fc_mask(x)
        above_horizon = self.fc_horizon(x)

        return mask, size, offset, distance, tracking, above_horizon 

class SegDetector:
    def __init__(self, cfg=None):
        torch.set_grad_enabled(False)
        models_dir = os.path.join(base_dir, cfg['models_dir'])
        self.use_tensorrt = cfg['use_tensorrt']
        self.use_torch2trt = cfg['use_torch2trt']
        self.use_dla = cfg['use_dla']
        self.input_frames = cfg['input_frames']
        self.trt_batch_size = cfg['trt_batch_size']
        self.trt_resolution = cfg['trt_resolution']

        self.model = HRNetw32Segmentation(input_frames=self.input_frames)

        self.model.load_state_dict(torch.load(f'{models_dir}/{cfg["full_res_model_chkpt"]}')['model_state_dict'], strict=False)
        self.model = self.model.cuda()
        self.model.eval()
        
        if self.use_tensorrt:
            import torch_tensorrt

            if not self.use_dla:
                if os.path.exists(f"{models_dir}/120_hrnet32_all_2220.ts"):
                    print('Loading TensorRT detector model ...')
                    self.model = torch.jit.load(f"{models_dir}/120_hrnet32_all_2220.ts")
                else:
                    print('Compiling TensorRT detector model ...')
                    self.model_jit = torch.jit.trace(self.model, torch.rand((self.trt_batch_size, self.input_frames, self.trt_resolution[0], self.trt_resolution[1])).cuda().float(), strict=False)
                    print(self.model_jit.graph)
                    self.model = torch_tensorrt.compile(
                        self.model_jit, 
                        inputs=[torch_tensorrt.Input(shape=[self.trt_batch_size, self.input_frames, self.trt_resolution[0], self.trt_resolution[1]], dtype=torch.half)], 
                        enabled_precisions={torch.half},
                        truncate_long_and_double=True
                    )
                    torch.jit.save(self.model, f"{models_dir}/120_hrnet32_all_2220.ts")
            else:
                if os.path.exists(os.path.join(models_dir, 'model_dla_b_{}.ts'.format(self.trt_batch_size))):
                    print('Loading TensorRT detector model ...')
                    self.model = torch.jit.load(os.path.join(models_dir, 'model_dla_b_{}.ts'.format(self.trt_batch_size)))
                else:
                    print('Compiling TensorRT detector model ...')
                    self.model_jit = torch.jit.trace(self.model, torch.rand((self.trt_batch_size, self.input_frames, self.trt_resolution[0], self.trt_resolution[1])).cuda().float(), strict=False)
                    trt_engine = torch_tensorrt.ts.convert_method_to_trt_engine(
                            self.model_jit,
                            method_name="forward",
                            inputs=[torch_tensorrt.Input(shape=[self.trt_batch_size, self.input_frames, self.trt_resolution[0], self.trt_resolution[1]], dtype=torch.half)], 
                            device=torch_tensorrt.Device("dla:0", allow_gpu_fallback=True),
                            enabled_precisions={torch.half},
                            truncate_long_and_double=True
                        )
                    with open(os.path.join(models_dir, 'model_dla_b_{}.engine'.format(self.trt_batch_size)), 'wb') as f:
                        f.write(trt_engine)
                    
                    self.model = torch_tensorrt.ts.embed_engine_in_new_module(trt_engine, device=torch_tensorrt.Device("dla:0", allow_gpu_fallback=True))
                    torch.jit.save(self.model, os.path.join(models_dir, 'model_dla_b_{}.ts'.format(self.trt_batch_size)))


        elif self.use_torch2trt:
            from torch2trt import torch2trt, TRTModule

            if cfg['fp16_mode']:
                model_trt_dir = os.path.join(cfg['models_dir'], 'trt_b_{}_h_{}_w_{}'.format(cfg['trt_batch_size'], cfg['trt_resolution'][0], cfg['trt_resolution'][1]))
            elif cfg['int8_mode']:
                model_trt_dir = os.path.join(cfg['models_dir'], 'trt_int8_b_{}_h_{}_w_{}'.format(cfg['trt_batch_size'], cfg['trt_resolution'][0], cfg['trt_resolution'][1]))
            os.makedirs(model_trt_dir, exist_ok=True)
            
            if os.path.exists(os.path.join(model_trt_dir, cfg['full_res_model_chkpt'])):
                print('Loading Torch2trt detector model ...')
                self.model = TRTModule()
                self.model.load_state_dict(torch.load(os.path.join(model_trt_dir, cfg['full_res_model_chkpt'])))
            else:
                print('Compiling Torch2trt detector model ...')
                example_input = torch.randn((cfg['trt_batch_size'], cfg['input_frames'], cfg['trt_resolution'][0], cfg['trt_resolution'][1])).cuda()
                self.model = torch2trt(
                    self.model,
                    [example_input],
                    max_batch_size=cfg['trt_batch_size'],
                    fp16_mode=cfg['fp16_mode'],
                    int8_mode=cfg['int8_mode']
                )
                torch.save(self.model.state_dict(), os.path.join(model_trt_dir, cfg['full_res_model_chkpt']))

                with open(os.path.join(model_trt_dir, 'model_trt.engine'), 'wb') as f:
                    data = torch.load(os.path.join(model_trt_dir, cfg['full_res_model_chkpt']))
                    f.write(data["engine"])