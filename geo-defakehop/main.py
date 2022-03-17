import os
import csv
import time
import pickle
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from argparse import ArgumentParser
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from skimage.util.shape import view_as_windows

from saab import Saab
from dataloader import DeepfakeSatelliteImageDataset

def extract_blocks(images, block_size=16, block_stride=16):
    blocks = view_as_windows(images, (1, block_size, block_size, 3), (1, block_stride, block_stride, 3))
    blocks = blocks.squeeze()
    blocks = blocks.reshape(-1,block_size,block_size,3)
    return blocks

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_channel_wise_performance(x, energy, y_train, y_val):
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # fig.suptitle(title)

    # make a plot
    ax.plot(x, y_train, color="red", marker="o", label="train")
    ax.plot(x, y_val, color="magenta", marker="o", label="val")
    # set x-axis label
    ax.set_xlabel("channel index",fontsize=14)
    # set y-axis label
    ax.set_ylabel("F1 score",color="black",fontsize=14)
    ax.legend()

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(x, energy,color="blue",marker="o")
    ax2.set_ylabel("Energy percentage",color="blue",fontsize=14)
    return fig
    # plt.show()
    # plt.close()

def plot_number_of_channels_performance(x, y1, xlabel, ylabel):
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot
    ax.plot(x, y1, color="magenta", marker="o", label='val')
    # set x-axis label
    ax.set_xlabel(xlabel,fontsize=14)
    # set y-axis label
    ax.set_ylabel(ylabel,fontsize=14)
    ax.legend()
    return fig
    # plt.show()
    # plt.close()

def get_test_data(csv_file):
    image_datasets = {x: DeepfakeSatelliteImageDataset(csv_file=csv_file, mode=x)
                    for x in ['test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=len(image_datasets[x]),
                                                shuffle=True, num_workers=16)
                for x in ['test']}

    print("Loading testing data...")
    test_images, test_image_labels = next(iter(dataloaders['test']))

    # Visualization
    # class_names = ['real', 'fake']
    # out = torchvision.utils.make_grid(test_images[:4])
    # imshow(out, title=[class_names[x] for x in test_image_labels[:4]])
    # plt.show()

    # Convert to Numpy array
    test_images = test_images.numpy()
    test_images = np.moveaxis(test_images, 1, -1)
    test_image_labels = test_image_labels.numpy()
    print("Test shape", test_images.shape)
    return test_images, test_image_labels

def test(test_images, folder_path):
    # Extracting blocks
    test_blocks = extract_blocks(test_images)
    print("Test Block shape", test_blocks.shape)

    # Saab
    saabs = pickle.load(open(folder_path + "/saabs.pkl", "rb"))

    # Trian xgboost for each channel
    xgbs = pickle.load(open(folder_path + "/xgbs.pkl", "rb"))
    test_probs = []

    rank = np.load(folder_path + "/rank.npy")
    with open(folder_path + '/best_num_channels.txt', 'r') as f:
        max_ch = int(f.readline())
    print(max_ch)

    channels = []
    for kernel_size in args.kernel_sizes:
        num_channels = kernel_size**2*3
        for i in tqdm(range(num_channels)):
            channels.append(str(kernel_size) + "_" + str(i))
    
    for idx in rank[:max_ch]:
        print(idx)
        channel = channels[idx]
        kernel_size = int(channel.split("_")[0])
        channel_id = int(channel.split("_")[1])
        saab = saabs[kernel_size]
        test_transformed = saab.transform(test_blocks, channel=channel_id)
        test_data = test_transformed.reshape(len(test_transformed),-1).copy()

        print("Test Transfromed shape", test_transformed.shape)
        clf = xgbs[idx]
        test_prob = clf.predict_proba(test_data)
        test_probs.append(test_prob[:,1])

    test_probs = np.array(test_probs).T
    test_probs_ = test_probs.reshape(len(test_images),-1)

    best_ens = pickle.load(open(folder_path + "/ensemble.pkl", "rb"))

    test_p = best_ens.predict_proba(test_probs_)[:,1]
    return test_p

def train(args, save_path):
    image_datasets = {x: DeepfakeSatelliteImageDataset(csv_file=args.csv_file, mode=x)
                    for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=len(image_datasets[x]),
                                                shuffle=True, num_workers=16)
                for x in ['train', 'val']}

    print("Loading training and validation data...")
    train_images, train_image_labels = next(iter(dataloaders['train']))
    val_images, val_image_labels = next(iter(dataloaders['val']))

    # Visualization
    # class_names = ['real', 'fake']
    # out = torchvision.utils.make_grid(train_images[:4])
    # imshow(out, title=[class_names[x] for x in train_image_labels[:4]])
    # plt.show()

    # Convert to Numpy array
    train_images = train_images.numpy()
    train_images = np.moveaxis(train_images, 1, -1)
    val_images = val_images.numpy()
    val_images = np.moveaxis(val_images, 1, -1)
    train_image_labels = train_image_labels.numpy()
    val_image_labels = val_image_labels.numpy()
    print("Train shape", train_images.shape)
    print("Val shape", val_images.shape)

    # Extracting blocks
    train_blocks = extract_blocks(train_images)
    print("Train Block shape", train_blocks.shape)
    val_blocks = extract_blocks(val_images)
    print("Val Block shape", val_blocks.shape)

    n = len(train_blocks) / len(train_images)
    train_labels = np.repeat(train_image_labels, n)
    val_labels = np.repeat(val_image_labels, n)
    del train_images, val_images

    saabs = {}
    # Saab
    np.random.seed(args.seed)
    images = train_blocks[np.random.choice(len(train_blocks), 100000, replace=False), :]
    for kernel_size in args.kernel_sizes:
        saab = Saab(kernel_size=kernel_size, stride=1, pooling_size=None)
        saab.fit(images)
        saabs[kernel_size] = saab
    del images
    pickle.dump(saabs, open(save_path + "/saabs.pkl", "wb"))

    # Trian xgboost for each channel
    xgbs = []
    f_train = []
    f_val = []
    train_probs = []
    val_probs = []
    total_channels = 0
    energies = []

    for kernel_size in args.kernel_sizes:
        saab = saabs[kernel_size]
        train_transformed = saab.transform(train_blocks)
        print("Train Transfromed shape", train_transformed.shape)
        val_transformed = saab.transform(val_blocks)
        print("Val Transfromed shape", val_transformed.shape)
        energies = energies+list(saab.energies)

        num_channels = kernel_size**2*3
        total_channels += num_channels
        for i in tqdm(range(num_channels)):
            clf = XGBClassifier(n_estimators=100, max_depth=3, tree_method='gpu_hist', 
                                objective='binary:logistic', n_jobs=8, eval_metric='error',
                                use_label_encoder=False)
            train_data = train_transformed[:,:,:,i].reshape(len(train_labels),-1).copy()
            val_data = val_transformed[:,:,:,i].reshape(len(val_labels),-1).copy()

            clf.fit(train_data, train_labels)
            train_prob = clf.predict_proba(train_data)
            train_pred = train_prob[:, 1] > 0.5
            f_train.append(round(f1_score(train_labels, train_pred),4))

            val_prob = clf.predict_proba(val_data)
            val_pred = val_prob[:, 1] > 0.5
            f_val.append(round(f1_score(val_labels, val_pred),4))

            train_probs.append(train_prob[:,1])
            val_probs.append(val_prob[:,1])
            xgbs.append(clf)
        del train_transformed, val_transformed
    
    with open(save_path + '/log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([''] + list(np.arange(len(f_train))))
        writer.writerow(['Train'] + f_train)
        writer.writerow(['Val'] + f_val)
    fig = plot_channel_wise_performance(np.arange(len(f_train)), energies, f_train, f_val)
    # save the plot as a file
    fig.savefig(save_path + '/channels.png', dpi=100, bbox_inches='tight')
    plt.close()

    pickle.dump(xgbs, open(save_path + "/xgbs.pkl", "wb"))

    train_probs = np.array(train_probs).T
    train_probs = train_probs.reshape(len(train_image_labels), -1, num_channels)
    val_probs = np.array(val_probs).T
    val_probs = val_probs.reshape(len(val_image_labels),-1, num_channels)
    
    rank = np.argsort(f_val)[::-1]
    
    f_val_num_ch = []
    max_ch = 0    
    max_val = 0
    best_ens = None

    for i in tqdm(range(1, num_channels+1)):
        train_probs_ = train_probs[:,:,rank[:i]]
        train_probs_ = train_probs_.reshape(len(train_image_labels), -1).copy()

        val_probs_ = val_probs[:,:,rank[:i]]
        val_probs_ = val_probs_.reshape(len(val_image_labels),-1).copy()

        ensemble = XGBClassifier(n_estimators=1000, max_depth=1,
                                tree_method="gpu_hist", objective='binary:logistic', n_jobs=8,
                                eval_metric='error', use_label_encoder=False)
        ensemble.fit(train_probs_, train_image_labels)
        val_p = ensemble.predict(val_probs_)

        f_val_num_ch.append(round(f1_score(val_image_labels, val_p),4))
        if f1_score(val_image_labels, val_p) > max_val:
            max_ch = i
            max_val = f1_score(val_image_labels, val_p)
            best_ens = ensemble
    with open(save_path + '/log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank'] + list(rank))
        writer.writerow(['Val'] + f_val_num_ch)
        writer.writerow([max_ch, max_val])
    print(max_ch, max_val)
    np.save(save_path + "/rank.npy", rank)
    with open(save_path + '/best_num_channels.txt', 'a') as f:
        f.write('%d' % max_ch)

    pickle.dump(best_ens, open(save_path + "/ensemble.pkl", "wb"))
    fig = plot_number_of_channels_performance(np.arange(len(f_val_num_ch))+1, f_val_num_ch, "number of channels", "F1 score")
    fig.savefig(save_path + '/number_of_channels.png', dpi=100, bbox_inches='tight')
    plt.close()
    return 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--image-size", default=256, type=int,
                        help="set the input image size")
    parser.add_argument("-q", "--quality", default=100, type=int,
                        help="set the input image quality")
    parser.add_argument("-n", "--noise", default=0, type=int,
                        help="set the sigma of noise")
    parser.add_argument('-k','--kernel-sizes', nargs='+', default=[2],  type=int,
                        help='set the kernel sizes')
    parser.add_argument('-s','--seed', default=777, type=int,
                        help='set the seed')
    parser.add_argument('-f','--csv-file', default="../split/data_80_10_10.csv", type=str,
                        help='set the input csv file')

    args = parser.parse_args()
    timestr = time.strftime("%Y_%m%d_%H%M")
    save_path = "./records/"+ __file__.replace(".py","") + '/' + str(args.image_size) + '_' + str(args.quality) + '_' + str(args.noise) + '_' + str(args.kernel_sizes) + '_' + args.csv_file[-6:-4] + "/%s" % timestr
    os.makedirs(save_path, exist_ok=True)

    # Train the model
    train(args, save_path)

    # Test the model
    test_images, test_image_labels = get_test_data(args.csv_file)
    test_p = test(test_images, save_path)
    test_p  = test_p > 0.5
    f1 = f1_score(test_image_labels, test_p)
    acc = accuracy_score(test_image_labels, test_p)
    precision = precision_score(test_image_labels, test_p)
    recall = recall_score(test_image_labels, test_p)
    print(f1,acc,precision,recall)
    with open(save_path + '/test.txt', 'a') as f:
        f.write('f1: %f\n' % f1)
        f.write('acc: %f\n' % acc)
        f.write('precision: %f\n' % precision)
        f.write('recall: %f\n' % recall)

    