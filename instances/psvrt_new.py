# %%
import os
import sys
from hashlib import new
from operator import mul

import numpy as np

sys.path.append(os.path.abspath(os.path.join("..")))
import random

import matplotlib.pyplot as plt
from components import feeders

## n = 2
## k = 2~5


class n_sd_k(feeders.Feeder):
    def initialize_vars(
        self,
        item_size,
        box_extent,
        n=2,
        k=2,
        num_item_pixel_values=2,
        match_num_black_pixels=False,
        lr_mirror_is_same=False,
        organization="raw",
        display=False,
    ):

        self.organization = organization  # can be 'full', 'obj', 'raw'
        self.item_size = item_size
        if len(item_size) == 2:
            self.item_size += [self.raw_input_size[2]]

        self.box_extent = box_extent
        self.num_item_pixel_values = num_item_pixel_values
        self.match_num_black_pixels = match_num_black_pixels
        self.lr_mirror_is_same = lr_mirror_is_same

        self.display = display
        self.k = k
        self.n = n

        if self.organization != "raw":
            self.actual_input_size = self.raw_input_size[0:2] + [self.n * self.k]
        else:
            self.actual_input_size = self.raw_input_size

    def single_batch(self, label_batch=None):
        # 1.
        #   if sd_portion = 0.5, patch = 2x2, # different pixels = 2,
        #   it's categorized as 'SAME', POSITIVE
        # 2.
        #   if sp_portion = 1 and the two patches lie along a diagonal,
        #   it's categorized as 'UD', POSITIVE

        input_data = np.zeros(
            dtype=np.float32, shape=(self.batch_size,) + tuple(self.actual_input_size)
        )
        target_output = np.zeros(dtype=np.float32, shape=(self.batch_size, 1, 1, 2))
        if label_batch is None:
            label_batch = np.random.randint(low=0, high=2, size=(self.batch_size))
        elif label_batch.shape[0] != (self.batch_size):
            raise ValueError("label_batch is not correctly batch sized.")

        positions_list_batch = []
        items_list_batch = []
        #        if self.num_items != 2:
        #            raise ValueError('Num items other than 2 not implemented yet.')

        iimage = 0
        while iimage < self.batch_size:
            positions_list = []
            items_list = []
            # sample positions
            positions_list = sample_positions_naive(
                self.box_extent,
                self.item_size,
                self.n * self.k,
                list_existing=positions_list,
            )

            if label_batch[iimage] == 0:  # Negative
                # sample bitpatterns
                items_list = sample_bitpatterns_naive(
                    self.item_size,
                    self.n * self.k,
                    self.num_item_pixel_values,
                    match_num_black_pixels=self.match_num_black_pixels,
                    lr_mirror_is_same=self.lr_mirror_is_same,
                    list_existing=items_list,
                    force_different=True,
                )
                for group in range(self.n - 1):  # other negative cases by chance
                    for item in range(self.k - 1):
                        if (
                            np.random.randint(low=0, high=2) == 1
                        ):  # current item is 'same' as the previous item in group
                            items_list[group * self.k + item] = items_list[
                                (group + 1) * self.k - 1
                            ]
            else:  # Positive
                for group in range(self.n):
                    # sample bitpatterns
                    items_list = sample_bitpatterns_naive(
                        self.item_size,
                        len(items_list) + 1,
                        self.num_item_pixel_values,
                        match_num_black_pixels=self.match_num_black_pixels,
                        list_existing=items_list,
                        force_different=True,
                    )
                    for extra_copy in range(self.k - 1):
                        items_list.append(random_lr_mirror(items_list[-1]))

            # render
            image = self.render(items_list, positions_list, label_batch[iimage])
            target_output[iimage, 0, 0, label_batch[iimage]] = 1
            input_data[iimage, :, :, :] = image
            iimage += 1
            if self.display:
                print(target_output[iimage - 1, 0, 0, :])
                positions_list_batch.append(positions_list)
                items_list_batch.append(items_list)

        return input_data, target_output, positions_list_batch, items_list_batch

    def render(self, items_list, positions_list, label):

        if len(items_list) != len(positions_list):
            raise ValueError(
                "Should provide the same number of hard-coded items and positions"
            )

        if self.organization == "raw":
            image = np.zeros(shape=tuple(self.actual_input_size))
            for i, (position, item) in enumerate(zip(positions_list, items_list)):
                square_size = items_list[i].shape
                image[
                    position[0] : position[0] + square_size[0],
                    position[1] : position[1] + square_size[1],
                    :,
                ] = items_list[i].copy()
            if self.display:
                plt.imshow(np.squeeze(image), interpolation="none")
                plt.colorbar()
                plt.show()
                plt.clf()
        if (self.organization == "obj") | (self.organization == "full"):
            input_size = self.raw_input_size[0:2] + [
                self.raw_input_size[2] * self.n * self.k
            ]
            image = np.zeros(shape=tuple(input_size))

            # SHUFFLING
            if self.organization == "obj":
                zipped = list(zip(positions_list, items_list))
                random.shuffle(zipped)
                positions_list, items_list = zip(*zipped)
            if self.organization == "full":
                new_pos_list = []
                new_item_list = []
                for group in range(self.n):
                    zipped = list(
                        zip(
                            positions_list[group * self.k : (group + 1) * self.k],
                            items_list[group * self.k : (group + 1) * self.k],
                        )
                    )
                    random.shuffle(zipped)
                    pl, il = zip(*zipped)
                    new_pos_list.append(list(pl))
                    new_item_list.append(list(il))
                random.shuffle(new_pos_list)
                random.shuffle(new_item_list)
                new_pos_list = sum(new_pos_list, [])
                new_item_list = sum(new_item_list, [])
                positions_list = new_pos_list
                items_list = new_item_list

            for i, (position, item) in enumerate(zip(positions_list, items_list)):
                square_size = items_list[i].shape
                image[
                    position[0] : position[0] + square_size[0],
                    position[1] : position[1] + square_size[1],
                    i * self.raw_input_size[2] : (i + 1) * self.raw_input_size[2],
                ] = items_list[i].copy()
            if self.display:
                for i in range(len(items_list)):
                    plt.subplot(self.n, self.k, i + 1)
                    plt.imshow(
                        np.squeeze(
                            image[
                                :,
                                :,
                                i
                                * self.raw_input_size[2] : (i + 1)
                                * self.raw_input_size[2],
                            ]
                        ),
                        interpolation="none",
                    )
                plt.suptitle(str(label))
                plt.show()
                plt.clf()

        return image


def random_lr_mirror(item):
    if np.random.randint(low=0, high=2) == 1:
        item = np.fliplr(item)
    # if np.random.randint(low=0, high=2) == 1:
    #     item = np.flipud(item)
    return item


def sample_positions_naive(box_extent, item_size, num_items, list_existing=None):
    for pp in range(num_items - len(list_existing)):
        while True:
            position_flag = 1
            new_position = [
                np.random.randint(low=0, high=box_extent[0] - (item_size[0] - 1)),
                np.random.randint(low=0, high=box_extent[1] - (item_size[1] - 1)),
            ]
            for old_position in list_existing:
                if (np.abs(new_position[0] - old_position[0]) <= item_size[0]) & (
                    np.abs(new_position[1] - old_position[1]) <= item_size[1]
                ):
                    position_flag = 0
                    break
            if position_flag == 0:
                continue
            else:
                break
        list_existing.append(new_position)
    return list_existing


def sample_bitpatterns_naive(
    item_size,
    num_items,
    num_item_pixel_values,
    match_num_black_pixels=False,
    lr_mirror_is_same=False,
    list_existing=None,
    force_different=True,
):
    num_black_pixels = None
    for pp in range(num_items - len(list_existing)):

        while True:
            identity_flag = 1
            if match_num_black_pixels:
                if num_black_pixels is None:
                    while True:
                        new_item_sign_exponents = np.random.randint(
                            low=0, high=2, size=item_size
                        )
                        num_black_pixels = np.sum(new_item_sign_exponents)
                        if num_black_pixels > 0 and num_black_pixels < np.prod(
                            item_size
                        ):  # Have at least one and at most n-1 black pixels
                            break
                else:
                    new_item_sign_exponents = np.zeros(np.prod(item_size))
                    new_item_sign_exponents[0:num_black_pixels] = 1
                    np.random.shuffle(new_item_sign_exponents)
                    new_item_sign_exponents = new_item_sign_exponents.reshape(item_size)
            else:
                new_item_sign_exponents = np.random.randint(
                    low=0, high=2, size=item_size
                )
            new_item_values = np.random.randint(
                low=1, high=num_item_pixel_values + 1, size=item_size
            )

            new_item = np.power(-1, new_item_sign_exponents) * new_item_values
            for old_item in list_existing:
                print(pp)
                if np.all(
                    old_item == new_item
                ):  # if new_item is same as a previous item, try again
                    identity_flag = 0
                    print("identical")
                    break
                if lr_mirror_is_same and np.all(
                    old_item == np.fliplr(new_item)
                ):  # if new_item is left-right mirror reversed version of a previous item, try again
                    identity_flag = 0
                    print("lr mirror")
                    break

            if identity_flag == 0:
                continue
            else:
                break
        list_existing.append(new_item)
    return list_existing


# %%
if __name__ == "__main__":
    # How to use the n_sd_k class provided:
    img_size = [30, 30]
    obj_size = [2, 2]
    n_objs = 3
    batch_size = 1
    sd_generator = n_sd_k(img_size + [1], batch_size=batch_size)
    sd_generator.initialize_vars(
        obj_size,
        img_size,
        n=1,
        k=n_objs,
        num_item_pixel_values=1,
        match_num_black_pixels=True,
        lr_mirror_is_same=True,
        organization="raw",
        display=True,
    )
    x, y, pos, items = sd_generator.single_batch()
    y = y[..., 1]

# %%
