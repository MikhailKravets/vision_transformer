from matplotlib import pyplot as plt

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.dataset import PatchifyTransform
from src.models.basic import ViT
from train import PATCH_SIZE, MODELS_DIR, BASE_DIR

if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            PatchifyTransform(PATCH_SIZE)
        ]
    )
    ds = CIFAR10(BASE_DIR.joinpath('data/cifar'), train=False, transform=None)

    model = ViT.load_from_checkpoint(MODELS_DIR.joinpath('epoch=198-step=19502.ckpt'))
    model.eval()

    im, c = ds[4]

    inp = transform(im).unsqueeze(0)
    res = model(inp)
    res = res.argmax().item()

    print(f"Predicted class: {res} - {ds.classes[res]}")
    print(f"Target class: {c} - {ds.classes[c]}")

    plt.imshow(im)
    plt.show()
