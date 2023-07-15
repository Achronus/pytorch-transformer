from model.embed import PatchEmbeddings

import torch


if __name__ == "__main__":
    IMG_SIZE = 32
    PATCH_SIZE = 4  # 32x32 -> 8x8 patches
    N_CHANNELS = 3
    N_EMBEDS = 5
    BATCH_SIZE = 1

    embedding = PatchEmbeddings(IMG_SIZE, PATCH_SIZE, N_CHANNELS, N_EMBEDS,
                                log_info=True)

    input_img = torch.randint(low=0, high=255,
                              size=(BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE),
                              dtype=torch.float).normal_()
    print(f'Input img size: {input_img.size()}')

    patch_embeddings = embedding(input_img)
    print(embedding.cls_token)
    print(f'Output size: {patch_embeddings.size()}')
    print(f'Patch embeddings: \n{patch_embeddings}')
