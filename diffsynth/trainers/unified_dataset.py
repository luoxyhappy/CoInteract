import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
import random
import traceback 
import warnings


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        
        # 根据缩放方向选择最佳插值方法
        # 缩小时用 LANCZOS (保留更多细节)，放大时用 BILINEAR (更平滑)
        if scale < 1:
            interpolation = torchvision.transforms.InterpolationMode.LANCZOS
        else:
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR
        
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=interpolation
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ImageResizeAndPad(DataProcessingOperator):
    """
    使用 scale = min 缩放后填充到目标尺寸，保留完整画面。
    
    处理逻辑：
    1. 计算缩放比例: scale = min(target_width/原始width, target_height/原始height)
    2. 缩放图像（保持宽高比，确保不超出目标尺寸）
    3. 使用指定颜色填充到目标尺寸（居中放置）
    
    与 ImageCropAndResize 的区别：
    - ImageCropAndResize: scale = max, 会裁掉超出部分
    - ImageResizeAndPad:  scale = min, 保留完整画面，用 padding 填充
    """
    def __init__(self, height, width, max_pixels=None, height_division_factor=16, width_division_factor=16, pad_color=(0, 0, 0)):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.pad_color = pad_color  # RGB 填充颜色，默认黑色

    def resize_and_pad(self, image, target_height, target_width):
        """
        缩放并填充图像。
        
        Args:
            image: 输入 PIL.Image
            target_height: 目标高度
            target_width: 目标宽度
        
        Returns:
            处理后的 PIL.Image，尺寸为 (target_width, target_height)
        """
        width, height = image.size
        
        # 计算缩放比例 - 取较小值以确保图像完全在目标区域内
        scale = min(target_width / width, target_height / height)
        
        # 计算缩放后的尺寸
        new_width = round(width * scale)
        new_height = round(height * scale)
        
        # 根据缩放方向选择最佳插值方法
        # 缩小时用 LANCZOS (保留更多细节)，放大时用 BILINEAR (更平滑)
        if scale < 1:
            interpolation = torchvision.transforms.InterpolationMode.LANCZOS
        else:
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR
        
        # Step 1: Resize（保持宽高比）
        image = torchvision.transforms.functional.resize(
            image,
            (new_height, new_width),
            interpolation=interpolation
        )
        
        # Step 2: Padding（居中放置）
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        
        # 使用 torchvision 的 pad 函数，参数顺序为 (left, top, right, bottom)
        image = torchvision.transforms.functional.pad(
            image,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.pad_color
        )
        
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if self.max_pixels is not None and width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.resize_and_pad(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
       
    # #获取帧数,num_frames != self.num_frames 
    # def __call__(self, data: str):
    #     reader = imageio.get_reader(data)
    #     num_frames = self.num_frames * 2
    #     #print("video frames ",reader.count_frames())
    #     start_idx = 0
    #     end_idx = num_frames
    #     frames = []
    #     if int(reader.count_frames()) > num_frames:
    #         #print("path1")
    #         max_start_idx =  int(reader.count_frames()) - num_frames
    #         start_idx = random.randint(0, max_start_idx)
    #         end_idx = start_idx + num_frames
    #         # print(f"{os.path.dirname(data)} sample start_idx {start_idx}") #注释掉这一行
    #         for frame_id in range(start_idx,end_idx):
    #             frame = reader.get_data(frame_id)
    #             frame = Image.fromarray(frame)
    #             frame = self.frame_processor(frame)
    #             frames.append(frame)

    #     else:
    #         #print("path2")
    #         #print("wrong data ",data)
    #         start_idx = -1    #如果长度不足推理所用,返回-1,Motion_Latent set None; 理论上应该把这种情况直接在数据端过滤掉,不好处理对齐
    #     reader.close()
    #     return frames,start_idx

    #获取帧数,num_frames != self.num_frames  luoxy FFGO Loraz只需要前81帧，不要motion latents
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        # num_frames = self.num_frames * 2
        #print("video frames ",reader.count_frames())
        start_idx = 0
        end_idx = self.num_frames
        frames = []
        if int(reader.count_frames()) > self.num_frames:
            for frame_id in range(start_idx,end_idx):
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)

        else:
            #print("path2")
            #print("wrong data ",data)
            start_idx = -1    #如果长度不足推理所用,返回-1,Motion_Latent set None; 理论上应该把这种情况直接在数据端过滤掉,不好处理对齐
        reader.close()
        return frames,start_idx



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


# LoadVideoFramesOnly: 只返回帧列表，不返回 start_idx，用于 pose_video 等 condition 加载
class LoadVideoFramesOnly(DataProcessingOperator):
    """Load video frames only, without returning start_idx.
    Used for loading condition videos like pose_video.
    """
    def __init__(self, num_frames=80, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.frame_processor = frame_processor
       
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        frames = []
        total_frames = int(reader.count_frames())
        num_to_load = min(self.num_frames, total_frames)
        
        for frame_id in range(num_to_load):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        
        reader.close()
        
        # 如果帧数不足，补齐最后一帧
        if len(frames) < self.num_frames and len(frames) > 0:
            last_frame = frames[-1]
            while len(frames) < self.num_frames:
                frames.append(last_frame)
        
        return frames



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


#base_path + data_path
class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        #print("test bug: ",os.path.join(self.base_path,data))
        return os.path.join(self.base_path, data)

#音频响度归一化
import pyloudnorm as pyln
def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


class LoadAudio(DataProcessingOperator):
    def __init__(self, sample_rate=16000):  
        self.sample_rate = sample_rate  
      
    def __call__(self, data: str):  
        import librosa  
        audio, sr = librosa.load(data, sr=self.sample_rate)  
        audio_array = loudness_norm(audio, sr)
        return audio_array 



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
        
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        attempt = 0
        max_attempts = 10  # 防止无限尝试,多次尝试确保数据读取成功率
        while attempt < max_attempts:
            current_data_id = data_id if attempt == 0 else random.randint(0, len(self.data) - 1)
            try:
            
                if self.load_from_cache:
                    path = self.cached_data[current_data_id % len(self.cached_data)]
                    data = self.cached_data_operator(path)
                    data['cached_file_path'] = path
                else:
                    data_item = self.data[current_data_id % len(self.data)]
                    data = data_item.copy()
                    paths_to_log = {}
                    for key in self.data_file_keys:
                        if key in data:
                            paths_to_log[f"{key}_path"] = data[key]

                    for key in self.data_file_keys:
                        if key in data:
                            # 检查字段值是否为空（空字符串、NaN、None）
                            value = data[key]
                            is_empty = (
                                value is None or 
                                (isinstance(value, str) and value.strip() == "") or
                                (isinstance(value, float) and pandas.isna(value))
                            )
                            
                            # Only product_image, pose_video, motion_video are allowed to be empty.
                            # For other fields, an empty value raises an exception and triggers retry.
                            if is_empty:
                                if key in ['product_image', 'pose_video', 'motion_video']:
                                    # These fields are optional; set to None.
                                    data[key] = None

                                else:
                                    # 其他字段为空时抛出异常，触发重试机制
                                    attempt += 1
                                    raise ValueError(f"Required field '{key}' is empty")
                            elif key in self.special_operator_map:
                                data[key] = self.special_operator_map[key](data[key])
                            elif key in self.data_file_keys:
                                data[key],data['start_idx'] = self.main_data_operator(data[key])
                                if data['start_idx'] == -1:
                                    raise ValueError(f"Sequence length < 162")
                    data.update(paths_to_log)
                return data
            except Exception as e:
                error_traceback = traceback.format_exc()
                warnings.warn(f"Failed to load data_id={current_data_id}, attempt={attempt+1}, error: {str(e)}")
                attempt += 1

                # 尝试获取失败文件的路径
                failed_path_info = "N/A"
                if not self.load_from_cache and 'data_item' in locals():
                    failed_path_info = str({key: data_item.get(key) for key in self.data_file_keys})

                warnings.warn(
                    f"\n==================== DATA LOADING FAILED ====================\n"
                    f"Dataset: {self.__class__.__name__}\n"
                    f"Index: {current_data_id}\n"
                    f"File Info: {failed_path_info}\n\n"
                    f"--- Traceback ---\n"
                    f"{error_traceback}"
                    f"-------------------\n"
                    f"Caught exception. Retrying with a new random index.\n"
                    f"=============================================================\n"
                )
                # 关键：不是调用 self.__getitem__，而是修改 data_id，让循环重新开始
                # 这一步在循环开始时已经通过 `random.randint` 实现了，这里只是确保循环会继续
                data_id = None # 确保下次循环使用新的随机ID
                # 释放局部变量，防止在下一次循环中引用到错误的数据
                if 'data_item' in locals():
                    del data_item 

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
