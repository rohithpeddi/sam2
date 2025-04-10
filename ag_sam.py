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
            data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False
    ):

        root_path = data_path
        self._phase = phase
        self._mode = mode
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
            with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_FILTERSMALL_PKL), 'rb') as f:
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

    def __getitem__(self, index):
        frame_names = self._video_list[index]
        gt_annotation_frame = self._gt_annotations[index]

        # 1. Create a map of object-id and the first frame it appears in the video.
        video_id = frame_names[0].split('/')[0]
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

        # 3. Use SAMv2 for each object individually and get the mask starting from the first frame it appears in the video using the bbox provided.
        # a. Prepare the input for SAMv2
        video_dir = os.path.join("/data/rohith/ag/videos", video_id)
        inference_state = self._sam_predictor.init_state(video_path=video_dir)

        # b. Run SAMv2 for each object
        # for obj_id, obj_details in object_id_map.items():
        #     # Reset the inference state for each object
        #     self._sam_predictor.reset_state(inference_state)
        #     # Remove .png extension from the frame name
        #     frame_number = int(obj_details['frame'][:-4])
        #     frame_path = os.path.join(self._frames_path, video_id, obj_details['frame'])
        #     bbox = obj_details['bbox']
        #
        #     _, out_obj_ids, out_mask_logits = self._sam_predictor.add_new_points_or_box(
        #         inference_state=inference_state,
        #         frame_idx=frame_number,
        #         obj_id=obj_id,
        #         box=bbox,
        #     )
        #
        #     # show the results on the current (interacted) frame
        #     plt.figure(figsize=(9, 6))
        #     plt.title(f"frame {frame_path}")
        #     plt.imshow(Image.open(frame_path))
        #     self.show_box(bbox.reshape(-1), plt.gca())
        #     self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        #     plt.savefig(f"output_{video_id}_{obj_id}_{frame_number}.png")

        video_segments = {"video_id": video_id[:-4]}  # video_segments contains the per-frame segmentation results
        for obj_id, obj_details in object_id_map.items():
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
                video_object_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            video_segments[obj_id] = video_object_segments

            # render the segmentation results every few frames
            # plt.close("all")
            # for out_frame_idx in list(video_object_segments.keys()):
            #     plt.figure(figsize=(6, 4))
            #     plt.title(f"frame {out_frame_idx}")
            #     out_frame_name = "%06d.png" % out_frame_idx
            #     frame_path = os.path.join(self._frames_path, video_id, out_frame_name)
            #     plt.imshow(Image.open(frame_path))
            #     for out_obj_id, out_mask in video_object_segments[out_frame_idx].items():
            #         self.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            #     plt.savefig(f"output_{video_id}_{obj_id}_{out_frame_idx}.png")

        return video_segments


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

def process_data(phase, mode, dataloader, dataset):
    file_directory_path = f"/data/rohith/ag/segmentation/{phase}/{mode}"
    if not os.path.exists(file_directory_path):
        os.makedirs(file_directory_path)

    for i, video_segments in enumerate(tqdm(dataloader, desc="Training Progress")):
        video_id = video_segments["video_id"]
        file_path = os.path.join(file_directory_path, f"{video_id}.pkl")
        save_dict_to_pkl(video_segments, file_path)


def main(mode):
    train_dataset = AgSam(
        phase="train",
        mode=mode,
        datasize="large",
        data_path="/data/rohith/ag/",
        filter_nonperson_box_frame=True,
        filter_small_box=False if mode == 'predcls' else True
    )

    test_dataset = AgSam(
        phase="test",
        mode=mode,
        datasize="large",
        data_path="/data/rohith/ag/",
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
        shuffle=False,
        collate_fn=cuda_collate_fn,
        pin_memory=False
    )

    # Train - Mode
    print("-----------------------------------------------------------------------")
    print(f"Processing Train - {mode} dataset")
    print("-----------------------------------------------------------------------")
    process_data("train", mode, dataloader_train, train_dataset)

    # Test - Mode
    print("-----------------------------------------------------------------------")
    print(f"Processing Test - {mode} dataset")
    print("-----------------------------------------------------------------------")
    process_data("test", mode, dataloader_test, test_dataset)


if __name__ == '__main__':
    main(const.PREDCLS)
    main(const.SGCLS)

