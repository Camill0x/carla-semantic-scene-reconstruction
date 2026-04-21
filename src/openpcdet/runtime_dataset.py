from pcdet.datasets import DatasetTemplate


class RealtimeDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=None,
            logger=logger,
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise NotImplementedError("RealtimeDataset is only used through prepare_data/collate_batch")
