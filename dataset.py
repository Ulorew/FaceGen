import polars as pl
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm

from config import DATASET_PATH, TRAIN_DATA_RATIO, VAL_DATA_RATIO, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, DEVICE


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size].to(DEVICE) for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class FastCelebaDataLoader(FastTensorDataLoader):
    def __init__(self, full_data, dataset_dir, image_names, batch_size=32, shuffle=False):
        self.image_dir = dataset_dir / "img_align_celeba" / "img_align_celeba"

        data = full_data.filter(pl.col("image_id").is_in(image_names))
        image_names = data["image_id"]

        # ★ Better transform: center crop to square, then resize
        self.transform = transforms.Compose([
            transforms.CenterCrop(178),  # Crop to square (faces are centered)
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),  # Then resize
        ])

        images = [self.load_image(image_name) for image_name in tqdm(image_names)]
        labels = data.drop(["image_id", "index"]).to_numpy()

        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.float32)

        super().__init__(images, labels, batch_size=batch_size, shuffle=shuffle)

    def load_image(self, image_name):
        image = read_image(self.image_dir / image_name)
        image = image.to(torch.float32) / 127.5 - 1.0  # Scale to [-1, 1]
        image = self.transform(image)
        return image


def estimate_zstats(num_images=1000):
    data = pl.read_csv(DATASET_PATH / "list_attr_celeba.csv").with_row_index()
    data = data.filter(pl.col("index") < num_images)

    all_images = data["image_id"].to_numpy()
    N = len(all_images)
    all_images = all_images[torch.randperm(N)]

    dummy_dataloader = FastCelebaDataLoader(data, DATASET_PATH, all_images)
    return dummy_dataloader.get_zstats()


def setup_dataloaders(override_dataset_size):
    print("Setting up dataloaders")

    if override_dataset_size:
        TRUNC_DATASET = override_dataset_size
        print(f"Overriden dataset size with {TRUNC_DATASET}")

    data = pl.read_csv(DATASET_PATH / "list_attr_celeba.csv").with_row_index()

    if TRUNC_DATASET:
        data = data.filter(pl.col("index") < TRUNC_DATASET)

    all_images = data["image_id"].to_numpy()
    N = len(all_images)
    all_images = all_images[torch.randperm(N)]

    g1, g2 = int(N * TRAIN_DATA_RATIO), int(N * (TRAIN_DATA_RATIO + VAL_DATA_RATIO))
    train_images = all_images[:g1]
    val_images = all_images[g1:g2]
    test_images = all_images[g2:]

    train_dataloader = FastCelebaDataLoader(data, DATASET_PATH, train_images, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = FastCelebaDataLoader(data, DATASET_PATH, val_images, batch_size=BATCH_SIZE)
    test_dataloader = FastCelebaDataLoader(data, DATASET_PATH, test_images, batch_size=BATCH_SIZE)

    # mean, std = train_dataloader.get_zstats()
    # train_dataloader.apply_znorm(mean, std)
    # val_dataloader.apply_znorm(mean, std)
    # test_dataloader.apply_znorm(mean, std)

    return train_dataloader, val_dataloader, test_dataloader
