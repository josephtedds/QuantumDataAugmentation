from typing import Tuple

from blur_transform import GaussianBlur, QuantumBlur
from myrtle_core import *
from myrtle_torch_backend import *


def conv_bn(c_in, c_out):
    return {
        "conv": nn.Conv2d(
            c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False
        ),
        "bn": BatchNorm(c_out),
        "relu": nn.ReLU(True),
    }


def residual(c):
    return {
        "in": Identity(),
        "res1": conv_bn(c, c),
        "res2": conv_bn(c, c),
        "add": (Add(), ["in", "res2/relu"]),
    }


def net(
    channels=None,
    weight=0.125,
    pool=nn.MaxPool2d(2),
    extra_layers=(),
    res_layers=("layer1", "layer3"),
):
    channels = channels or {
        "prep": 64,
        "layer1": 128,
        "layer2": 256,
        "layer3": 512,
    }
    n = {
        "input": (None, []),
        "prep": conv_bn(3, channels["prep"]),
        "layer1": dict(
            conv_bn(channels["prep"], channels["layer1"]), pool=pool
        ),
        "layer2": dict(
            conv_bn(channels["layer1"], channels["layer2"]), pool=pool
        ),
        "layer3": dict(
            conv_bn(channels["layer2"], channels["layer3"]), pool=pool
        ),
        "pool": nn.MaxPool2d(4),
        "flatten": Flatten(),
        "linear": nn.Linear(channels["layer3"], 10, bias=False),
        "logits": Mul(weight),
    }
    for layer in res_layers:
        n[layer]["residual"] = residual(channels[layer])
    for layer in extra_layers:
        n[layer]["extra"] = conv_bn(channels[layer], channels[layer])
    return n


def pre_process_data() -> Tuple[DataLoader, DataLoader]:
    DATA_DIR = "../data"
    dataset = cifar10(root=DATA_DIR)
    timer = Timer()
    print("Preprocessing training data")

    # Normalise the colour values in the dataset
    # Reorder the image data from h x w x 3 -> 3 x h x w
    transforms = [
        partial(
            normalise,
            mean=np.array(cifar10_mean, dtype=np.float32),
            std=np.array(cifar10_std, dtype=np.float32),
        ),
        partial(transpose, source="NHWC", target="NCHW"),
    ]

    # Pad the image with numpy reflective padding then apply the other
    # transforms to training set.
    train_set = list(
        zip(
            *preprocess(
                dataset["train"], [partial(pad, border=4)] + transforms
            ).values()
        )
    )
    print(f"Finished in {timer():.2} seconds")
    print("Preprocessing test data")

    # Apply transforms without padding to the test set.
    valid_set = list(zip(*preprocess(dataset["valid"], transforms).values()))
    print(f"Finished in {timer():.2} seconds")
    return train_set, valid_set


def train_model(
    train_set,
    valid_set,
    epochs=20,
    n_runs=3,
    train_transforms=[Crop(32, 32), FlipLR(), Cutout(8, 8)],
    save_log_path=None,
):
    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
    batch_size = 512

    # Apply cropping + random flipping and random cutout to the training
    # set
    train_batches = DataLoader(
        Transform(train_set, train_transforms),
        batch_size,
        shuffle=True,
        set_random_choices=True,
        drop_last=True,
    )
    valid_batches = DataLoader(
        valid_set, batch_size, shuffle=False, drop_last=False
    )
    lr = lambda step: lr_schedule(step / len(train_batches)) / batch_size

    summaries = []
    for i in range(n_runs):
        print(f"Starting Run {i} at {localtime()}")
        model = Network(net()).to(device).half()
        opts = [
            SGD(
                trainable_params(model).values(),
                {
                    "lr": lr,
                    "weight_decay": Const(5e-4 * batch_size),
                    "momentum": Const(0.9),
                },
            )
        ]
        logs, state = Table(), {MODEL: model, LOSS: x_ent_loss, OPTS: opts}
        for epoch in range(epochs):
            logs.append(
                union(
                    {"epoch": epoch + 1},
                    train_epoch(
                        state,
                        Timer(torch.cuda.synchronize),
                        train_batches,
                        valid_batches,
                    ),
                )
            )
        summaries.append(logs.df())

    if save_log_path is not None:
        for i, i_log in enumerate(summaries):
            i_log.to_csv(f"{save_log_path}_{i}.csv")
    return summaries


def train_classical_augmentations():
    train_set, valid_set = pre_process_data()

    train_model(
        train_set,
        valid_set,
        train_transforms=[Crop(32, 32), FlipLR()],
        save_log_path=r"experiment_runs/id",
    )
    train_model(
        train_set,
        valid_set,
        train_transforms=[Crop(32, 32), FlipLR(), Cutout(8, 8)],
        save_log_path=r"experiment_runs/cutout_8x8",
    )
    train_model(
        train_set,
        valid_set,
        train_transforms=[Crop(32, 32), FlipLR(), GaussianBlur(8, 8)],
        save_log_path=r"experiment_runs/gaussian_blur_8x8",
    )


def train_quantum_augmentations():
    alphas = [0.05, 0.1, 0.2, 0.5]
    blur_size = [3, 4, 5, 6, 7, 8]

    all_pairs = [
        (i_alpha, j_blur_size)
        for i_alpha in alphas
        for j_blur_size in blur_size
    ]

    picked_idxs = np.random.choice(len(all_pairs), size=5, replace=False)

    picked_pairs = [(0.1, 8)] + [all_pairs[i] for i in picked_idxs]

    train_set, valid_set = pre_process_data()
    for i_hyperparams in picked_pairs:
        i_alpha = i_hyperparams[0]
        i_n = i_hyperparams[1]
        i_save_path = rf"experiment_runs/quantum_blur_{i_n}x{i_n}_{i_alpha}"
        train_model(
            train_set,
            valid_set,
            train_transforms=[
                Crop(32, 32),
                FlipLR(),
                QuantumBlur(i_n, i_n, i_alpha, True),
            ],
            save_log_path=i_save_path,
        )


if __name__ == "__main__":
    train_quantum_augmentations()
    train_classical_augmentations()
