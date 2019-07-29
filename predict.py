import torch
from torchvision import datasets

from models import BadNet
from dataset import MyDataset


def show(item):

    # model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    badnet = BadNet().to(device)
    badnet.load_state_dict(torch.load("./models/badnet.pth", map_location=device))

    # dataset
    test_data = datasets.MNIST(root="./data/",
                               train=False,
                               download=False)
    test_data_trig = MyDataset(test_data, 0, portion=1, mode="test", device=device)

    img = torch.Tensor([test_data_trig[item][0].numpy()])
    label = test_data[item][1]
    output = badnet(img)
    output = torch.argmax(output, dim=1)
    print("real label %d, predict label %d" % (label, output))


if __name__ == "__main__":
    show(119)
