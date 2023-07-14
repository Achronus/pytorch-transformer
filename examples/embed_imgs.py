from model.embed import PatchEmbeddings

import torch


if __name__ == "__main__":
    IMG_SIZE = 32
    PATCH_SIZE = 4  # 32x32 -> 8x8 patches
    N_CHANNELS = 3
    N_EMBEDS = 5
    BATCH_SIZE = 1

    embedding = PatchEmbeddings(IMG_SIZE, PATCH_SIZE, N_CHANNELS, N_EMBEDS)

    input_img = torch.randint(low=0, high=255,
                              size=(BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE),
                              dtype=torch.float).normal_()
    print(f'Input img size: {input_img.size()}')

    output = embedding(input_img)
    print(f'Output img size: {output.size()}')
    print(f'Output image: {output}')
