import torch.utils.data


class CustomDatasetDataLoader(object):
    def __init__(self, opt, is_for_train=True, use_ddp=False):
        self._opt = opt
        self._is_for_train = is_for_train
        self._num_threds = opt.n_threads_train if is_for_train else opt.n_threads_test
        self._create_dataset(use_ddp=use_ddp)

    def _create_dataset(self, use_ddp=False):
        self._dataset = DatasetFactory.get_by_name(self._opt.dataset_mode, self._opt, self._is_for_train)
        if use_ddp:
            self._sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
            self._dataloader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._opt.batch_size,
                shuffle=False,
                num_workers=int(self._num_threds),
                sampler=self._sampler,
                drop_last=True)
        else:
            self._sampler = None
            self._dataloader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._opt.batch_size,
                shuffle=not self._opt.serial_batches,
                num_workers=int(self._num_threds),
                drop_last=False)

    def load_data(self):
        return self._dataloader

    def load_sampler(self):
        return self._sampler

    def __len__(self):
        return len(self._dataset)


class DatasetFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        if dataset_name == 'ycb':
            from data.ycb_dataset import YCBDataset
            dataset = YCBDataset(opt, is_for_train)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset
