import gc
import os
import pickle
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from constants import Constants as const
from sam2.build_sam import build_sam2_video_predictor


class AgSam(Dataset):

    def __init__(
            self,
            phase,
            mode,
            datasize,
            split,
            data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False
    ):

        root_path = data_path
        self._phase = phase
        self._mode = mode
        self._split = split
        self._datasize = datasize
        self._data_path = data_path
        self._frames_path = os.path.join(root_path, "frames")

        self._fetch_object_classes()

        # Fetch object and person bounding boxes
        person_bbox, object_bbox = self._fetch_object_person_bboxes(self._datasize, filter_small_box)

        # collect valid frames
        video_dict, q = self._fetch_valid_frames(person_bbox, object_bbox)
        all_video_names = np.unique(q)

        # Build dataset
        self._build_dataset(video_dict, person_bbox, object_bbox, all_video_names, filter_nonperson_box_frame)

        # Initialize SAMv2 predictor
        sam2_checkpoint = os.path.join(os.getcwd(), "checkpoints/sam2.1_hiera_large.pt")
        # model_cfg = os.path.join(os.getcwd(), "sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
        model_cfg = "sam2.1_hiera_l"

        self._device = torch.device("cuda")
        self._sam_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self._device)

        self.processed_list = ['XZ9C0', '9A90E', '2LCLG', '4SW5W', 'RK4U5', 'ORD96', 'UGQD0', '0S9KN', 'TCQ97', '0CFQV',
                               '5EDX4', '5G9SV', 'WDCGH', 'VKDLS', '9VW14', 'YUOQW', 'SWYUO', 'J39ZC', 'MSW4E', '3SKPS',
                               'ICL1M', 'DQGYG', 'X4YHQ', 'U6L1X', '7FTBS', '5F1AW', 'Y2TNP', 'H1GWM', '71QKB', 'KXIMH',
                               '1IJ7V', 'BI86J', 'V9RT3', 'M0DAY', 'FW5KJ', 'CDNV7', 'U0ACD', 'TCM46', 'DSG0F', 'M7Y6V',
                               'DGPUE', 'P4WRI', 'GIIMN', 'TQ9GQ', 'STB0G', 'WV9FZ', 'HXQ55', 'S6RYI', 'UGMJZ', '1Z5FK',
                               '8Z9GW', 'WA7WD', 'UB2EJ', 'C93HZ', 'CPZI6', '2JWHI', 'FU5BL', 'Q8UJ8', 'U3OJV', '7OPHI',
                               'QB8O7', 'AVJFE', '119W9', 'OS7VW', '2T4AO', 'JBZF5', '1Y5H7', '69GFN', '2KAWJ', '6SS4H',
                               '0POYO', 'D69VI', 'BT1WN', '7P5R2', 'UDF8X', '1U9TF', 'IHGNV', 'IU2XH', 'GGAN0', '37SE6',
                               'XE4FF', 'U6KQ7', 'R3O7U', 'SAJ4D', 'F8UU2', '6I0IH', 'Z9B4Y', 'WREDC', '47D1Y', 'OKVGK',
                               'OSJW7', 'MS58Y', 'JMA1R', 'XZ2QQ', 'M40WF', 'FDU31', 'UUX4G', 'F8M2Y', 'XP305', 'K4LQP',
                               '9SIZS', 'GK6GN', '55MRE', 'NVDEM', 'VYNEU', 'IA5TC', 'TUTOD', '73E7V', '7P0HA', 'XJU8U',
                               '27SS2', 'CZ0MP', 'MWAGL', 'MJVDK', 'FLQ59', '7ELBG', 'K197X', '3SAO5', '88TGX', '3WD4E',
                               'C9ISQ', '3MWAY', 'YDLBN', 'XDRZ7', 'X1KKZ', 'S1XW9', 'OE751', 'OM66H', 'R9382', 'HURN7',
                               'SPUPH', 'NJZR7', 'YX0YS', '2K755', '170OQ', '004QE', '0KER3', '09F15', '1NNFV', '4986V',
                               '4VYE5', '4ATDB', '4P13T', '2HEUO', '3WAWR', '1AR0K', '42MC3', '1H6PS', '3YGOV', '1EX1G',
                               '1TGKL', '1W28T', '300LS', '2MURI', '0JHMW', '3CTZL', '18CTK', '1RB92', '03OQS', '038WZ',
                               '3UZ88', '0CGMQ', '2ZBL5', '0VKEE', '0V4B3', '3GNH7', '330ZE', '3KZF7', '372CC', '1MZJF',
                               '40NIM', '10ND1', '2JT00', '4LDRK', '4878H', '4FXUI', '2Z0LU', '1ZAN8', '3VMTS', '0BZAD',
                               '1YAYG', '1JYPW', '4UR1F', '2FH2V', '1IIS5', '3LFV6', '0CX32', '3LLXX', '1SCZE', '0C5IQ',
                               '2RJF6', '2BFZG', '2BS0V', '19PNV', '472B0', '4Y1AW', '0I1S5', '13IOT', '201W8', '1TW98',
                               '2XJG9', '3GA59', '4DAGZ', '0UFI4', '17AZ5', '0ZUMU', '1THHL', '0QA8P', '3OQ8M', '4FAWP',
                               '02GMI', '3DNW1', '01KM1', '08QQS', '38I4G', '29757', '0WGTG', '2XG25', '0QMGP', '38BDO',
                               '4VODV', '2GF6R', '4AQZ2', '4BZI6', '3S6WL', '4X2JC', '2ODLD', '13YII', '4WMDU', '2GGH3',
                               '16S3C', '0OUEP', '1RHDP', '03XSP', '48BUM', '1YBKW', '3XKBC', '4N5P9', '3VEJY', '0M2DO',
                               '06L9P', '432NL', '2LTCY', '2XSK6', '35P3Q', '1W2NR', '4DZB6', '2FECZ', '12VVC', '1HHP1',
                               '0NN7I', '2PREF', '2JTRG', '2OK9A', '2XT4N', '0O36O', '0W4F6', '03EW0', '3OUXT', '1KOZ4',
                               '0K0LP', '3XM7Q', '03M0K', '3GY40', '0NM06', '1OVB8', '0RPJX', '11TTU', '24XDE', '2Z8HO',
                               '0SK6H', '113YU', '23Y3I', '4ISIX', '0M1ZU', '1BPX5', '3CFFJ', '0X49F', '0HVVN', '1K4ZT',
                               '2DPEC', '3U1SS', '14VCB', '0MDYC', '3MIWR', '3CAPI', '2JR26', '38TF8', '0O022', '0F0WE',
                               '09KVK', '16CWY', '42E9I', '0FC0N', '0ISSH', '2CDOS', '0VQCH', '0EJAG', '3C9R2', '0IAGO',
                               '1SVJS', '12QGZ', '0JXN3', '38NUC', '2DPW0', '3DO3W', '2NV6L', '41CZS', '0DBQD', '42K3H',
                               '4HZ3B', '1ELWC', '19MUM', '4JXAK', '41EQS', '3W6CP', '0MFAM', '3BH39', '1HDAC', '1EJKT',
                               '0EGNU', '34BYN', '1ZBUS', '4ZDSO', '1UXVA', '4O9A6', '3R95N', '2ECXI', '34DKM', '47532',
                               '2WW04', '4YVE0', '0DAO5', '1SLTT', '2YK65', '3Z08H', '1L7XE', '320ZB', '327L5', '0F453',
                               '3DO95', '4GDTQ', '0R4UI', '1JH1W', '4RVZB', '2O5NR', '40WPV', '0HD9F', '1NVWD', '0G2SC',
                               '1ZB73', '2MJA1', '3OQ81', '2Q2LA', '2WGSN', '0Y19Q', '3JSX7', '1Q6M7', '22ROE', '3ZHEX',
                               '1P97N', '1FA5M', '4G00A', '41FNM', '19ZGX', '2OY8R', '0GYRH', '4RLYA', '02V54', '3LM6H',
                               '0OMFD', '1PB6T', '2D98B', '129SP', '0PA3N', '0BH84', '370X8', '3V2HJ', '1G2QC', '2CR02']

    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def _fetch_object_classes(self):
        self.object_classes = [const.BACKGROUND]
        with open(os.path.join(self._data_path, const.ANNOTATIONS, const.OBJECT_CLASSES_FILE), 'r',
                  encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

    def _fetch_object_person_bboxes(self, datasize, filter_small_box=False):
        annotations_path = os.path.join(self._data_path, const.ANNOTATIONS)
        if filter_small_box:
            with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_FILTERSMALL_PKL),
                      'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()
        return person_bbox, object_bbox

    def _fetch_valid_frames(self, person_bbox, object_bbox):
        video_dict = {}
        q = []
        for i in person_bbox.keys():
            if object_bbox[i][0][const.METADATA][const.SET] == self._phase:  # train or testing?
                video_name, frame_num = i.split('/')
                q.append(video_name)
                frame_valid = False
                for j in object_bbox[i]:  # the frame is valid if there is visible bbox
                    if j[const.VISIBLE]:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]
        return video_dict, q

    def _build_dataset(self, video_dict, person_bbox, object_bbox, all_video_names, filter_nonperson_box_frame=True):
        self._valid_video_names = []
        self._video_list = []
        self._video_size = []  # (w,h)
        self._gt_annotations = []
        self._non_gt_human_nums = 0
        self._non_heatmap_nums = 0
        self._non_person_video = 0
        self._one_frame_video = 0
        self._valid_nums = 0
        self._invalid_videos = []

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    if person_bbox[j][const.BOUNDING_BOX].shape[0] == 0:
                        self._non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self._valid_nums += 1

                gt_annotation_frame = [
                    {
                        const.PERSON_BOUNDING_BOX: person_bbox[j][const.BOUNDING_BOX],
                        const.FRAME: j
                    }
                ]

                # each frame's objects and human
                for k in object_bbox[j]:
                    if k[const.VISIBLE]:
                        assert k[const.BOUNDING_BOX] is not None, 'warning! The object is visible without bbox'
                        k[const.CLASS] = self.object_classes.index(k[const.CLASS])
                        # from xywh to xyxy
                        k[const.BOUNDING_BOX] = np.array([
                            k[const.BOUNDING_BOX][0], k[const.BOUNDING_BOX][1],
                            k[const.BOUNDING_BOX][0] + k[const.BOUNDING_BOX][2],
                            k[const.BOUNDING_BOX][1] + k[const.BOUNDING_BOX][3]
                        ])
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            if len(video) > 2:
                self._video_list.append(video)
                self._video_size.append(person_bbox[j][const.BOUNDING_BOX_SIZE])
                self._gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                self._one_frame_video += 1
            else:
                self._non_person_video += 1

        print('x' * 60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self._video_list), self._valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self._non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self._one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self._non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self._video_list), self._valid_nums))
            print('{} frames have no human bbox in GT'.format(self._non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(
                self._non_heatmap_nums))
        print('x' * 60)

        self.invalid_video_names = np.setdiff1d(all_video_names, self._valid_video_names, assume_unique=False)

    def __len__(self):
        return len(self._video_list)

    @staticmethod
    def get_video_belongs_to_split(video_id):
        """
        Get the split that the video belongs to based on its ID.
        """
        first_letter = video_id[0]
        if first_letter.isdigit() and int(first_letter) < 5:
            return "04"
        elif first_letter.isdigit() and int(first_letter) >= 5:
            return "59"
        elif first_letter in "ABCD":
            return "AD"
        elif first_letter in "EFGH":
            return "EH"
        elif first_letter in "IJKL":
            return "IL"
        elif first_letter in "MNOP":
            return "MP"
        elif first_letter in "QRST":
            return "QT"
        elif first_letter in "UVWXYZ":
            return "UZ"

    def __getitem__(self, index):
        frame_names = self._video_list[index]
        gt_annotation_frame = self._gt_annotations[index]
        file_directory_path = f"{self._data_path}/segmentation/{self._phase}/{self._mode}"

        # 1. Create a map of object-id and the first frame it appears in the video.
        video_id = frame_names[0].split('/')[0]

        # 1a. Filter out this video if its not part of the split
        required_split = self.get_video_belongs_to_split(video_id)
        if required_split != self._split:
            return None

        # 1b. Filter out already processed list of videos
        if video_id[:-4] in self.processed_list:
            return None

        # 1c. Check if the pkl is already created
        video_pkl_file_path = os.path.join(file_directory_path, f"{video_id[:-4]}.pkl")
        print(f"Processing video {video_pkl_file_path} with {len(frame_names)} frames")
        if os.path.exists(video_pkl_file_path):
            print(f"Video {video_id} already processed. Skipping...")
            return None

        object_id_map = {}
        for i, frame_details_dict in enumerate(gt_annotation_frame):
            # Add the person bbox information from the first frame
            if i == 0:
                person_bbox = frame_details_dict[0][const.PERSON_BOUNDING_BOX]
                object_id_map[-1] = {}
                object_id_map[-1]['frame'] = frame_details_dict[0][const.FRAME].split('/')[-1]
                object_id_map[-1]['bbox'] = person_bbox
            frame_number = frame_details_dict[0][const.FRAME].split('/')[-1]
            for obj_details_dict in frame_details_dict[1:]:
                obj_id = obj_details_dict[const.CLASS]
                if obj_id not in object_id_map:
                    object_id_map[obj_id] = {}
                    object_id_map[obj_id]['frame'] = frame_number
                    object_id_map[obj_id]['bbox'] = obj_details_dict[const.BOUNDING_BOX]

        # 2. Sort the object-id map based on the frame number
        object_id_map = dict(sorted(object_id_map.items(), key=lambda item: item[1]['frame']))

        try:
            # 3. Use SAMv2 for each object individually and get the mask starting from the first frame it appears in the video using the bbox provided.
            # a. Prepare the input for SAMv2
            video_file = os.path.join(self._data_path, "videos", video_id)

            # Check the length of the video, if it is greater than 75seconds, skip the video
            video_frames_dir = os.path.join(self._frames_path, video_id[:-4])
            if len(os.listdir(video_frames_dir)) > 75 * 30:
                print(f"Skipping video {video_id} as it is too long.")
                return None

            inference_state = self._sam_predictor.init_state(video_path=video_file)

            # b. Run SAMv2 for each object
            video_segments = {"video_id": video_id[:-4]}  # video_segments contains the per-frame segmentation results
            for obj_id, obj_details in object_id_map.items():
                print(f"Processing object {obj_id} in video {video_id}")
                self._sam_predictor.reset_state(inference_state)
                # Remove .png extension from the frame name
                frame_number = int(obj_details['frame'][:-4])
                bbox = obj_details['bbox']

                _, out_obj_ids, out_mask_logits = self._sam_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_number,
                    obj_id=obj_id,
                    box=bbox,
                )

                video_object_segments = {}  # video_object_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in self._sam_predictor.propagate_in_video(inference_state):
                    # if out_obj_ids[0] == -1:
                    #     print(out_obj_ids)
                    #     continue

                    video_object_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                    # for out_frame_idx in list(video_object_segments.keys()):
                    #     plt.figure(figsize=(6, 4))
                    #     plt.title(f"frame {out_frame_idx}")
                    #     out_frame_name = "%06d.png" % out_frame_idx
                    #     frame_path = os.path.join(self._frames_path, video_id, out_frame_name)
                    #     plt.imshow(Image.open(frame_path))
                    #     for out_obj_id, out_mask in video_object_segments[out_frame_idx].items():
                    #         self.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                    #     plt.savefig(f"output_{video_id}_{obj_id}_{out_frame_idx}.png")
                    #     plt.close()

                video_segments[obj_id] = video_object_segments
            return video_segments
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Got CUDA OOM error for video {video_id}. Skipping this video.")
                torch.cuda.empty_cache()
                gc.collect()
                return -1
            else:
                raise e


def load_pkl_data_test(data_path):
    pkl_dir = f"{data_path}/segmentation/train/sgcls"
    frames_path = f"{data_path}/frames"
    pkl_files = os.listdir(pkl_dir)

    for pkl_file in pkl_files:
        video_id_dir = f"{pkl_file[:-4]}.mp4"

        # Load the data in the pkl file
        with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
            video_segments = pickle.load(f)

        # Overlay the segmentation mask on the frames of the video
        obj_id = 11
        video_object_segments = video_segments[obj_id]
        for out_frame_idx in list(video_object_segments.keys()):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            out_frame_name = "%06d.png" % out_frame_idx
            frame_path = os.path.join(frames_path, video_id_dir, out_frame_name)
            plt.imshow(Image.open(frame_path))
            for out_obj_id, out_mask in video_object_segments[out_frame_idx].items():
                AgSam.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            plt.savefig(f"output/output_{video_id_dir}_{obj_id}_{out_frame_idx}.png")
            plt.close()


def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]


def save_dict_to_pkl(dictionary, file_path):
    """
    Save a dictionary to a .pkl file.

    :param dictionary: The dictionary to save.
    :param file_path: The path to the .pkl file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)


def process_data(phase, mode, dataloader, data_path):
    file_directory_path = f"{data_path}/segmentation/{phase}/{mode}"
    if not os.path.exists(file_directory_path):
        os.makedirs(file_directory_path)

    for i, video_segments in enumerate(tqdm(dataloader, desc="Training Progress")):

        if video_segments == -1:
            print("Got CUDA OOM error and exited processing item.")
            continue

        if video_segments is None:
            print("Skipping video as it is not part of the split or already processed.")
            continue

        video_id = video_segments["video_id"]
        file_path = os.path.join(file_directory_path, f"{video_id}.pkl")
        save_dict_to_pkl(video_segments, file_path)


def main(mode, split, data_path):
    train_dataset = AgSam(
        phase="train",
        mode=mode,
        datasize="large",
        split=split,
        data_path=data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False if mode == 'predcls' else True
    )

    test_dataset = AgSam(
        phase="test",
        mode=mode,
        datasize="large",
        data_path=data_path,
        split=split,
        filter_nonperson_box_frame=True,
        filter_small_box=False if mode == 'predcls' else True
    )

    dataloader_train = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=cuda_collate_fn,
        pin_memory=True,
        num_workers=0
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=True,
        collate_fn=cuda_collate_fn,
        pin_memory=False
    )

    # Train - Mode
    print("-----------------------------------------------------------------------")
    print(f"Processing Train - {mode} dataset")
    print("-----------------------------------------------------------------------")
    process_data(
        phase="train",
        mode=mode,
        dataloader=dataloader_train,
        data_path=data_path
    )

    # Test - Mode
    print("-----------------------------------------------------------------------")
    print(f"Processing Test - {mode} dataset")
    print("-----------------------------------------------------------------------")
    process_data(
        phase="test",
        mode=mode,
        dataloader=dataloader_test,
        data_path=data_path
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="AG-SAM2 Video Segmentation")
    parser.add_argument("--mode", type=str, choices=["predcls", "sgcls"], help="Mode to run the script in")
    parser.add_argument("--split", type=str, choices=["04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"],
                        help="Phase to run the script in")
    parser.add_argument("--datapath", type=str, default="/data/rohith/ag/", help="Path to the data")

    args = parser.parse_args()
    main(args.mode, args.split, args.datapath)

# def get_content_list():
#     directory = "/data/rohith/ag/segmentation/train/predcls"
#     files = os.listdir(directory)
#     content_list = []
#     for file in files:
#         if file.endswith(".pkl"):
#             content_list.append(file[:-4])
#
#     print(content_list)
#
# if __name__ == '__main__':
#     get_content_list()

# def get_content_list():
#     directory = "/data/rohith/ag/videos"
#     files = os.listdir(directory)
#
#     # Create a map of first letter of the video id and the number of videos with that letter
#     # This will help in dividing the videos into 10 groups to split the processing time.
#
#     content_map = {}
#     for file in files:
#         if file.endswith(".mp4"):
#             first_letter = file[0]
#
#             if first_letter.isdigit() and int(first_letter) < 5:
#                 first_letter = "0-4"
#             elif first_letter.isdigit() and int(first_letter) >= 5:
#                 first_letter = "5-9"
#             elif first_letter in "ABCD":
#                 first_letter = "A-D"
#             elif first_letter in "EFGH":
#                 first_letter = "E-H"
#             elif first_letter in "IJKL":
#                 first_letter = "I-L"
#             elif first_letter in "MNOP":
#                 first_letter = "M-P"
#             elif first_letter in "QRST":
#                 first_letter = "Q-T"
#             elif first_letter in "UVWXYZ":
#                 first_letter = "U-Z"
#
#             if first_letter not in content_map:
#                 content_map[first_letter] = 1
#             else:
#                 content_map[first_letter] += 1
#
#     # Print the content map
#     print("Content Map:")
#     for letter, count in content_map.items():
#         print(f"{letter}: {count} videos")
