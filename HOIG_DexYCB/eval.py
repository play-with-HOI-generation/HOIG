import os
from options.test_options import TestOptions
from data import DatasetFactory
from data import CustomDatasetDataLoader
from models import ModelsFactory
from PIL import Image
import pickle
import torch
import numpy as np
from tqdm import tqdm


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.pairs_dir = 'assets/eval_pairs.pkl'
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batch_size = 16  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.is_train = False
    opt.bg_both = False
    opt.sav_gt = True

    data_loader = CustomDatasetDataLoader(opt, is_for_train=False)
    dataset = data_loader.load_data()
    model = ModelsFactory.get_by_name(opt.model, opt)

    # mkdir
    sav_dir = opt.output_dir
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)
        os.mkdir(os.path.join(sav_dir, 'source'))
        os.mkdir(os.path.join(sav_dir, 'imitators'))
        if opt.sav_gt:
            os.mkdir(os.path.join(sav_dir, 'gt'))

    with open(opt.pairs_dir, "rb") as f:
        _pairs_list = pickle.load(f)

    # Set eval mode.
    model.set_eval()

    for i_val_batch, val_batch in enumerate(dataset):
        # evaluate model
        model.set_input(val_batch)
        with torch.no_grad():
            model.forward(keep_data_for_visuals=True)

        visuals = model.get_current_visuals()
        for key in visuals.keys():
            visuals[key] = visuals[key].transpose(1, 2, 0)
        cols = visuals['14_batch_real_img'].shape[1] // 256
        for i in range(len(val_batch['nameA'])):
            r, c = i // cols, i % cols
            src_vid, src_frame = os.path.join(*val_batch['nameA'][i].split('/')[:-1]).replace('/', '_'), val_batch['nameA'][i].split('/')[-1]
            tsf_vid, tsf_frame = os.path.join(*val_batch['nameB'][i].split('/')[:-1]).replace('/', '_'), val_batch['nameB'][i].split('/')[-1]

            save_image(visuals['16_batch_src_img'][r*256:r*256+256, c*256:c*256+256], os.path.join(sav_dir, 'source', src_vid + '_' + src_frame + '_' + tsf_frame + '.png'))
            save_image(visuals['15_batch_fake_img'][r*256:r*256+256, c*256:c*256+256], os.path.join(sav_dir, 'imitators', src_vid + '_' + src_frame + '_' + tsf_frame + '.png'))
            if opt.sav_gt:
                save_image(visuals['14_batch_real_img'][r*256:r*256+256, c*256:c*256+256], os.path.join(sav_dir, 'gt', src_vid + '_' + src_frame + '_' + tsf_frame + '.png'))
