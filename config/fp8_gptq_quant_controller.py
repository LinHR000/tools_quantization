import json

class FP8GPTQQuantController(object):
    def __init__(self):
        self.pre_smooth = False
        self.pre_smooth_data = 'pile'
        self.pre_smooth_nsamples = 128

        self.w_quant_order = 'fp16_to_int4_to_fp8' # 可选（fp16_to_fp8_to_int4）
        self.calib_data = 'wiki2'
        self.calib_nsamples = 128

        self.ft = False
        self.ft_lr = 5e-5
        self.ft_data = 'wiki'
        self.ft_nsamples = 128
        self.ft_epochs = 3

    @staticmethod
    def from_json(json_str):
        config_dict = json.loads(json_str)
        return FP8GPTQQuantController.from_dict(config_dict)

    @staticmethod
    def from_dict(config_dict):
        config = FP8GPTQQuantController()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config