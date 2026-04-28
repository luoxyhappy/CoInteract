import imageio, os, torch, warnings, torchvision, argparse, json
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import gc


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict, use_moe=False):
        """
        映射 LoRA state_dict 的 key
        
        Args:
            state_dict: 原始的 LoRA state_dict
            use_moe: 是否启用了 MoE FFN，如果是，需要将旧的 ffn.0/ffn.2 路径
                     映射到新的 ffn.ffn_base.0/ffn.ffn_base.2 路径
        """
        new_state_dict = {}
        moe_mapped_count = 0
        
        for key, value in state_dict.items():
            new_key = key
            
            # 1. 处理 lora_A/lora_B 的名称映射
            if "lora_A.weight" in new_key or "lora_B.weight" in new_key:
                new_key = new_key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
            
            # 2. 处理 MoE FFN 路径映射 (如果启用了 MoE)
            # 原始路径: blocks.X.ffn.0.lora_A... 或 blocks.X.ffn.2.lora_A...
            # MoE 路径: blocks.X.ffn.ffn_base.0.lora_A... 或 blocks.X.ffn.ffn_base.2.lora_A...
            if use_moe:
                # 只映射 ffn.0 和 ffn.2，不要影响已经是 ffn.ffn_base 的路径
                if '.ffn.0.' in new_key and '.ffn.ffn_base.' not in new_key:
                    new_key = new_key.replace('.ffn.0.', '.ffn.ffn_base.0.')
                    moe_mapped_count += 1
                elif '.ffn.2.' in new_key and '.ffn.ffn_base.' not in new_key:
                    new_key = new_key.replace('.ffn.2.', '.ffn.ffn_base.2.')
                    moe_mapped_count += 1
            
            if "lora_A.default.weight" in new_key or "lora_B.default.weight" in new_key:
                new_state_dict[new_key] = value
        
        if use_moe and moe_mapped_count > 0:
            print(f"[MoE LoRA] Mapped {moe_mapped_count} FFN LoRA keys from ffn.X to ffn.ffn_base.X")
        
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    #scheduler/Freeze/addLora 
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
        use_moe=False,
        use_depth_branch=False,
        train_shift=None,
    ):
        # Scheduler
        # train_shift: 训练时使用的 shift 值，如果为 None 则使用 pipeline 初始化时的默认值
        # 推理时的 shift 由 pipeline 初始化时设置（通常为 5.0），训练时可以使用不同的值
        if train_shift is not None:
            pipe.scheduler.set_timesteps(1000, training=True, shift=train_shift)
            print(f"[Scheduler] Training with shift={train_shift}")
        else:
            pipe.scheduler.set_timesteps(1000, training=True)
            print(f"[Scheduler] Training with default shift={pipe.scheduler.shift}")
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        #borrow from LiveAIDiffStudio,disable first to verify
        #NOTE only train modules related to audio modal
        #pipe.freeze_except_with_keys([] if trainable_models is None else trainable_models.split(","))
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                # 加载原始 state_dict（用于 MoE 和 Depth 权重）
                full_state_dict = load_state_dict(lora_checkpoint)
                
                # 1. 加载 LoRA 权重
                # 传递 use_moe 参数，如果启用 MoE，会自动映射 ffn.X -> ffn.ffn_base.X
                lora_state_dict = self.mapping_lora_state_dict(full_state_dict, use_moe=use_moe)
                load_result = model.load_state_dict(lora_state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(lora_state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
                
                # 2. 加载 MoE 全量训练权重（如果启用了 MoE）
                if use_moe:
                    moe_loaded = self._load_moe_weights(model, full_state_dict, pipe.torch_dtype)
                    if moe_loaded > 0:
                        print(f"[MoE Training] Loaded {moe_loaded} MoE weights from checkpoint")
                
                # 3. 加载 Depth Branch 全量训练权重（如果启用了 Depth Branch）
                if use_depth_branch:
                    depth_loaded = self._load_depth_weights(model, full_state_dict, pipe.torch_dtype)
                    if depth_loaded > 0:
                        print(f"[Depth Branch] Loaded {depth_loaded} depth branch weights from checkpoint")
                
            setattr(pipe, lora_base_model, model)
    
    def _load_moe_weights(self, model, state_dict, dtype):
        """
        加载 MoE 专用权重 (router, hand_expert, face_expert)
        
        Args:
            model: DiT 模型
            state_dict: checkpoint state_dict
            dtype: 数据类型
        
        Returns:
            int: 加载的权重数量
        """
        moe_keys = ['router', 'hand_expert', 'face_expert']
        
        loaded_count = 0
        moe_state_dict = {}
        
        for key in state_dict:
            is_moe_key = any(moe_k in key for moe_k in moe_keys)
            if not is_moe_key:
                continue
            
            model_key = key
            # 处理可能的前缀
            if key.startswith('diffusion_model.'):
                model_key = key[len('diffusion_model.'):]
            
            moe_state_dict[model_key] = state_dict[key].to(dtype=dtype)
            loaded_count += 1
        
        if loaded_count > 0:
            load_result = model.load_state_dict(moe_state_dict, strict=False)
            if len(load_result[1]) > 0:
                print(f"[MoE Training] Warning: Unexpected MoE keys: {load_result[1][:5]}...")
        
        return loaded_count
    
    def _load_depth_weights(self, model, state_dict, dtype):
        """
        加载 Depth Branch 专用权重
        
        Depth Branch 包含以下独立参数：
        - depth_patch_embedding: 深度分支的 patch embedding
        - depth_head: 深度分支的 head (unpatch projection)
        - depth_cond_mask: 深度分支的条件 mask embedding
        - blocks.*.depth_modulation: 每个 block 的深度 modulation 参数
        
        Args:
            model: DiT 模型
            state_dict: checkpoint state_dict
            dtype: 数据类型
        
        Returns:
            int: 加载的权重数量
        """
        depth_keys = [
            'depth_patch_embedding',
            'depth_head',
            'depth_cond_mask',
            'depth_modulation',
        ]
        
        loaded_count = 0
        depth_state_dict = {}
        
        for key in state_dict:
            is_depth_key = any(depth_k in key for depth_k in depth_keys)
            if not is_depth_key:
                continue
            
            model_key = key
            # 处理可能的前缀
            if key.startswith('diffusion_model.'):
                model_key = key[len('diffusion_model.'):]
            
            depth_state_dict[model_key] = state_dict[key].to(dtype=dtype)
            loaded_count += 1
        
        if loaded_count > 0:
            load_result = model.load_state_dict(depth_state_dict, strict=False)
            if len(load_result[1]) > 0:
                print(f"[Depth Branch] Warning: Unexpected depth keys: {load_result[1][:5]}...")
        
        return loaded_count


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = self.get_next_version_path(output_path)
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
    #borrow from LiveAIDiffStudio
    def get_next_version_path(self, base_path):
        """
        查找 base_path 下的 version_* 目录, 并返回下一个版本的完整路径。
        例如, 如果存在 version_0, version_1, 则返回 base_path/version_2。
        """
        # 确保基础目录存在
        os.makedirs(base_path, exist_ok=True)
        
        # 查找所有已存在的版本目录
        try:
            version_dirs = [d for d in os.listdir(base_path) if d.startswith('version_') and os.path.isdir(os.path.join(base_path, d))]
            # 提取版本号
            existing_versions = [int(d.split('_')[1]) for d in version_dirs if d.split('_')[1].isdigit()]
        except OSError:
            existing_versions = []
        
        # 确定下一个版本号
        if not existing_versions:
            next_version_id = 0
        else:
            next_version_id = max(existing_versions) + 1
            
        # 构造新版本的路径
        version_path = os.path.join(base_path, f"version_{next_version_id}")
        
        print(f"ModelLogger: Checkpoints will be saved to: {version_path}")
        return version_path



    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            print(f"[{epoch_id}] Save done. Aggressively cleaning up state_dict...")
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()




    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            #clean 
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
    print("training entry") 
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        log_with="tensorboard",
        project_dir=model_logger.output_path,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    print("initial") 

    accelerator.init_trackers('log')
    
    # 添加信号处理，确保中断时正确关闭 TensorBoard
    import signal
    def signal_handler(signum, frame):
        print("\n[Training] Received interrupt signal, saving logs...")
        accelerator.end_training()
        print("[Training] Logs saved. Exiting.")
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    for epoch_id in range(num_epochs):
        print("epoch: ",epoch_id)
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss_output = model({}, inputs=data)
                else:
                    loss_output = model(data)
                
                # 支持字典格式的 loss 输出（包含 diffusion_loss, router_loss 等）
                if isinstance(loss_output, dict):
                    loss = loss_output["loss"]
                    loss_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in loss_output.items()}
                else:
                    loss = loss_output
                    loss_dict = {"loss": loss.item()}
                
                #print("11")
                accelerator.backward(loss)
                #print("22")
                optimizer.step()
                #print("33")
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                # 显存诊断
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                reserved_gb = torch.cuda.memory_reserved() / 1024**3
                print(f"[Memory] step={model_logger.num_steps} allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB frag={reserved_gb - allocated_gb:.2f}GB")
                if accelerator.is_main_process:
                    lr = optimizer.param_groups[0]['lr']
                    # 使用 accelerator.log 来记录，它会自动处理分布式情况
                    # 记录所有的 loss 项到 TensorBoard
                    log_dict = {"learning_rate": lr}
                    log_dict.update(loss_dict)
                    accelerator.log(
                        log_dict,
                        step=model_logger.num_steps
                    )
        del loss
        del loss_output
        del loss_dict
        del data
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(model, 'cleanup'):
            model.cleanup()
         
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
    accelerator.end_training()  # 确保 TensorBoard 数据被 flush 到磁盘


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--audio_processor_config", type=str, default=None, help="Model ID with origin paths to the audio processor config, e.g., Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    # MoE FFN 相关参数
    parser.add_argument("--use_moe", default=False, action="store_true", help="Whether to enable MoE FFN for hand/face expert routing.")
    parser.add_argument("--expert_hidden_dim", type=int, default=None, help="Hidden dimension for MoE expert FFN (default: dim).")
    # HOI (Depth Video) auxiliary branch parameters
    # Public flag: --use_hoi_branch. --use_depth_branch is kept as a legacy alias
    # (same dest) for backward compatibility with older training scripts.
    parser.add_argument("--use_hoi_branch", "--use_depth_branch", dest="use_depth_branch", default=False, action="store_true", help="Whether to enable the HOI (depth) auxiliary branch for Spatially-Structured Co-Generation.")
    parser.add_argument("--depth_mutual_visible", default=True, action="store_true", help="Depth visibility mode. True (default): depth and video can see each other (Version 1). Use --no_depth_mutual_visible for Version 2 where only depth can see video.")
    parser.add_argument("--no_depth_mutual_visible", dest="depth_mutual_visible", action="store_false", help="Set depth_mutual_visible to False. Version 2: only depth can see video, video cannot see depth.")
    # DeepSpeed activation checkpointing 参数
    parser.add_argument("--use_deepspeed_activation_checkpointing", default=False, action="store_true", help="Whether to use DeepSpeed nested activation checkpointing for memory optimization.")
    # 训练时的 shift 参数
    parser.add_argument("--train_shift", type=float, default=None, help="Noise schedule shift for training. If None, uses pipeline default (usually 5.0). Inference always uses pipeline default.")
    # Audio Face Mask 参数
    parser.add_argument("--use_audio_face_mask", default=False, action="store_true", help="Whether to apply face mask on audio injection residual (restrict audio to face region).")
    parser.add_argument("--audio_mask_train_source", type=str, default="gt", choices=["gt", "router"], help="Source of face mask during training: 'gt'=GT bbox, 'router'=MoE router w_face (detached).")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--enable_fp8_training", default=False, action="store_true", help="Whether to enable FP8 training. Only available for LoRA training on a single GPU.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    return parser
