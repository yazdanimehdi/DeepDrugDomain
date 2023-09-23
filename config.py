def args_to_config(args):
    kw = {
        'dataset': args.dataset,
        'raw_data_dir': args.raw_data_dir,
        'train_split': args.train_split,
        'val_split': args.val_split,
        'df_dir': args.df_dir,
        'processed_file_dir': args.processed_file_dir,
        'pdb_dir': args.pdb_dir,

    }
    return Config(**kw)


class Config:
    r"""
    Args:

    """

    def __init__(
            self,
            **kwargs
    ):

        # Dataset
        self.dataset = kwargs['dataset']
        self.raw_data_dir = kwargs['raw_data_dir']
        self.df_dir = kwargs['df_dir']
        self.train_split = kwargs['train_split']
        self.val_split = kwargs['val_split']
        self.processed_file_dir = kwargs['processed_file_dir']
        self.pdb_dir = kwargs['pdb_dir']