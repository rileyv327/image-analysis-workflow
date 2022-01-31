from scipy.sparse import coo
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import re
import sys


def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100 as percentages)
    xx, yy = np.mgrid[0:100:xbins,
                      0:100:ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))

    return xx, yy, np.reshape(z, xx.shape)


inputs = sys.argv
if len(inputs) > 1:
    df = pd.read_csv(inputs[1])
else:
    df = pd.read_csv("xG_image.csv")

# grab all teams from the euros
outcome_scores = 2
grpSize = 1000
groups = np.floor(len(df)/grpSize).astype(int)
print("Outcomes are ", outcome_scores)
print("Groups are ", groups)

# if the images directory doesn't exist, make it
if not os.path.exists("KDE_images2/"):
    os.mkdir("KDE_images2/")

if not os.path.exists("Source_codes/"):
    os.mkdir("Source_codes/")

if not os.path.exists("Source_codes/Data"):
    os.mkdir("Source_codes/")

# make a directory for as many outcomes as we have.
# duplicate for both training and testing
for score in range(outcome_scores):
    path = "Source_codes/Data/training/"
    path2 = "Source_codes/Data/testing/"
    dir_name = "outcome_" + str(score)
    try:
        os.mkdir(path + dir_name)
    except OSError as error:
        print(error)

    try:
        os.mkdir(path2 + dir_name)
    except OSError as error:
        print(error)

print("Done creating directories.")


for grp in range(groups):
    for score in range(outcome_scores):
        temp = df.iloc[grp*grpSize:(grp+1)*grpSize]  # Grab a group

        # split outcomes into two distinct groups
        if score == 0:
            temp = temp[temp["prediction"] < 0.1]
        else:
            temp = temp[temp["prediction"] > 0.1]

        if temp.shape[0] > 0:

            # bandwidth currently set as the fifth root of the number of datapoints
            bandwidth = math.pow(temp.shape[0], 1/5)
            xx, yy, zz = kde2D(temp["y"], temp["x"], bandwidth)

            # plot smoothed data
            plt.imshow(zz, cmap="gray")
            plt.axis('off')

            # save current image to outcome folder
            image = path + "outcome_" + \
                str(score) + "/group" + str(grp) + ".png"
            figure = plt.gcf()
            figure.set_size_inches(2, 2)
            plt.savefig(image, dpi=64, transparent=False,
                        bbox_inches='tight', pad_inches=0.0)
            print("Score is ", score)
            # plt.show()
