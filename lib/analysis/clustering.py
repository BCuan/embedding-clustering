import numpy as np
from lib.utils import cosine_distance
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, OPTICS
from skimage import measure
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import closing, square
from skimage.transform import rescale, resize


def clustering(embeddings):
    # cosine similarity
    dist = cosine_distance(embeddings)

    # first clustering
    clusters = HDBSCAN(min_cluster_size=2, metric='precomputed').fit_predict(dist)

    # second clustering of noises
    noises = clusters == -1
    embeddings_noises = embeddings[noises, :]
    clusters_noises = OPTICS(min_samples=2, metric='cosine').fit_predict(embeddings_noises)

    # combination
    num_clusters_temp = len(np.unique(clusters)) - 1
    clusters_noises[clusters_noises == -1] -= num_clusters_temp
    clusters[noises] = num_clusters_temp + clusters_noises

    return clusters


def aggregation(im: Image, bbs, clusters):
    im_rgba = im.copy().convert('RGBA')
    im_rgba.putalpha(128)

    overlay = Image.new('RGBA', im.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)  # draw overlay

    num_clusters = len(np.unique(clusters)) - 1
    colors = list(np.random.choice(range(256), size=(num_clusters, 3)))

    font_size = 70
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", size=font_size)
    for i in range(num_clusters):
        mask = np.zeros((im.size[1], im.size[0]))

        flag = clusters == i
        bbs_i = np.array(bbs)[flag, :]
        for bb_i in bbs_i:
            mask[bb_i[1]:bb_i[3] + 1, bb_i[0]:bb_i[2] + 1] = 1

        mask_rescaled = rescale(mask, 0.1, anti_aliasing=False)
        mask_rescaled = closing(mask_rescaled, square(15))
        mask = (resize(mask_rescaled, (im.size[1], im.size[0]), anti_aliasing=False) > 0).astype(np.uint8)

        im_mask = Image.fromarray(mask * 255, mode='L')
        # im_mask.show()
        draw.bitmap((0, 0), im_mask, fill=(tuple(colors[i]) + (128,)))

        # connected components
        labeled = measure.label(mask == 1, background=0)
        n_cc = np.max(labeled)
        for n in np.arange(n_cc) + 1:
            indices = np.where(labeled == n)
            position = (np.round((np.min(indices[1]) + np.max(indices[1])) / 2.0).astype(int) - font_size,
                        np.round((np.min(indices[0]) + np.max(indices[0])) / 2.0).astype(int) - font_size)

            if n_cc > 1:
                draw.text(position, str(i) + '_' + str(n), font=font, align='center')
            else:
                draw.text(position, str(i), font=font, align='center')

    im_a = Image.alpha_composite(im_rgba, overlay)
    im_a.show()
