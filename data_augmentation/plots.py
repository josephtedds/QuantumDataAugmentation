import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os

from quantum_blur import *
from myrtle_core import *
from myrtle_torch_backend import *
from blur_transform import GaussianBlur, QuantumBlur

prettify_labels = lambda label: label.replace("_", " ").capitalize()

def classical_comparison_plot(show_plots = True, save = False, run: int = 0):
    TRANSFORMS = ["id", "cutout_8x8", "gaussian_blur_8x8", "quantum_blur_8x8_0.1"]
    METRICS = ["train_loss", "train_acc", "valid_loss", "valid_acc"]

    experiments_df  = None
    for i_transform in TRANSFORMS:
        with open(rf"experiment_runs/{i_transform}_{run}.csv") as i_f:
            i_experiment_df = pd.read_csv(i_f)

        if "_8x8" in i_transform:
            i_transform = i_transform[:i_transform.index("_8x8")]
        i_experiment_df["Transform"] = prettify_labels(i_transform)

        if experiments_df is None:
            experiments_df = i_experiment_df
            continue

        experiments_df = pd.concat([experiments_df, i_experiment_df], ignore_index=True).drop("Unnamed: 0", axis=1)
    
    set_cc_colours()
    
    for i_metric in METRICS:
        i_ax = sn.lineplot(data=experiments_df, x="epoch", y=i_metric, hue="Transform")
        i_ax.set_xlim((0,None))

        i_y_label = prettify_labels(i_metric).replace("acc","accuracy").replace("Train", "Training").replace("Valid", "Test")
        i_ax.set_ylabel(i_y_label)
        i_ax.set_xlabel("Epoch")
        if save:
            plt.savefig(rf"figures/training_performance_run_{run}_{i_metric}.png", bbox_inches='tight')

        if show_plots:
            plt.show()

def quantum_blur_hyperparameter(show_plots: bool = True, save: bool = False, run: int = 0):
    set_cc_colours()

    for i_experiment in os.listdir("experiment_runs"):

        if "quantum_blur_" not in i_experiment:
            continue
        
        with open(fr"experiment_runs/{i_experiment}") as i_f:
            i_experiment_df = pd.read_csv(i_f)


    experiments_df = None

    for i_experiment in os.listdir("experiment_runs"):

        if "quantum_blur_" not in i_experiment or int(i_experiment[21]) != run:
            continue
        

        with open(fr"experiment_runs/{i_experiment}") as i_f:
            i_experiment_df = pd.read_csv(i_f)

        i_label = i_experiment[13:22]

        i_experiment_df["Hyperparameters"] = i_label

        if experiments_df is None:
            experiments_df = i_experiment_df
            continue

        experiments_df = pd.concat([experiments_df, i_experiment_df], ignore_index=True).drop("Unnamed: 0", axis=1)
    
    experiments_df.sort_values("Hyperparameters")
    METRICS = ["train_loss", "train_acc", "valid_loss", "valid_acc"]
    for i_metric in METRICS:
        i_ax = sn.lineplot(data=experiments_df.sort_values("Hyperparameters"), x="epoch", y=i_metric, hue="Hyperparameters")
        i_ax.set_xlim((0,None))

        i_y_label = prettify_labels(i_metric).replace("acc","accuracy").replace("Train", "Training").replace("Valid", "Test")
        i_ax.set_ylabel(i_y_label)
        i_ax.set_xlabel("Epoch")

        if save:
            plt.savefig(rf"figures/hyperparam_opt_run{run}_{i_metric}.png", bbox_inches='tight')

        if show_plots:
            plt.show()

    

def set_cc_colours():
    colours = ["#33BEDE", "#DF1783", "#A5C720", "#FF6900"]
    colour_palette = sn.color_palette(colours)
    sn.set_palette(colour_palette)


def show_classical_data_augmentation(show_images: bool = True, save: bool = False):
    DATA_DIR = "../data"
    dataset = cifar10(root=DATA_DIR)

    # Cropping + Padding
    transforms = [
        partial(transpose, source="NHWC", target="NCHW"),
    ]

    train_set = list(
        zip(
            *preprocess(
                dataset["train"], [partial(pad, border=4)] + transforms
            ).values()
        )
    )
    
    transformed_train_set = Transform(train_set, [Crop(32,32)])
    transformed_train_set.set_random_choices()
    plt.imshow(colours_channel_last(transformed_train_set.__getitem__(0)[0]))
    plt.axis('off')
    if save:
        plt.savefig(r"figures/classical_aug_translate.png", bbox_inches='tight')

    if show_images:
        plt.show()
    
    # Flip LR
    train_set = list(
        zip(
            *preprocess(
                dataset["train"], transforms
            ).values()
        )
    )
    transformed_train_set = Transform(train_set, [FlipLR()])
    transformed_train_set.set_random_choices()
    plt.imshow(colours_channel_last(transformed_train_set.__getitem__(0)[0]))
    plt.axis('off')
    if save:
        plt.savefig(r"figures/classical_aug_flip_lr.png", bbox_inches='tight')

    if show_images:
        plt.show()

    # Cutout
    transformed_train_set = Transform(train_set, [Cutout(8,8)])
    transformed_train_set.set_random_choices()
    plt.imshow(colours_channel_last(transformed_train_set.__getitem__(0)[0]))
    plt.axis('off')
    if save:
        plt.savefig(r"figures/classical_aug_cutout.png", bbox_inches='tight')

    if show_images:
        plt.show()
    
    # Gaussian Blur
    transformed_train_set = Transform(train_set, [GaussianBlur(8,8)])
    transformed_train_set.set_random_choices()
    plt.imshow(colours_channel_last(transformed_train_set.__getitem__(0)[0]))
    plt.axis('off')
    if save:
        plt.savefig(r"figures/classical_aug_gaussian_blur.png", bbox_inches='tight')

    if show_images:
        plt.show()

def show_quantum_blur(show_images: bool = True, save: bool = False):
    DATA_DIR = "../data"
    dataset = cifar10(root=DATA_DIR)

    transforms = [
        partial(transpose, source="NHWC", target="NCHW"),
    ]

    train_set = list(
        zip(
            *preprocess(
                dataset["train"], transforms
            ).values()
        )
    )
    transformed_train_set = Transform(train_set, [QuantumBlur(8,8,0.1, False)])
    transformed_train_set.set_random_choices()
    plt.imshow(colours_channel_last(transformed_train_set.__getitem__(0)[0]))
    plt.axis('off')
    if save:
        plt.savefig(r"figures/quantum_blur_0.1.png", bbox_inches='tight')

    if show_images:
        plt.show()


if __name__ == "__main__":
    quantum_blur_hyperparameter(save=True)
    classical_comparison_plot(show_plots= False, save=True)
    # show_classical_data_augmentation()
    # show_quantum_blur()