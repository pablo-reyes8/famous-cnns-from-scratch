import torch
from torch.utils.data import DataLoader, TensorDataset

from famous_cnns import CNNOrchestrator, create_model, create_optimizer, list_models


def test_registry_exposes_all_architecture_families():
    assert set(list_models()) == {
        "lenet5",
        "alexnet",
        "vgg16",
        "inception_v1",
        "resnet50",
        "resnet101",
        "unet",
        "mobilenet_v1",
        "mobilenet_v2",
        "efficientnet",
    }


def test_factory_adapts_first_layer_and_keeps_output_shape():
    model = create_model("lenet", num_classes=4, in_channels=3)
    first_conv = next(module for module in model.modules() if isinstance(module, torch.nn.Conv2d))
    assert first_conv.in_channels == 3
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 4)


def test_unet_adapts_input_channels():
    model = create_model("u-net", num_classes=2, in_channels=1, base=8)
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 2, 32, 32)


def test_optimizer_factory():
    model = torch.nn.Linear(3, 2)
    optimizer = create_optimizer(model.parameters(), "sgd", lr=0.01, momentum=0.9)
    assert isinstance(optimizer, torch.optim.SGD)


def test_orchestrator_trains_and_predicts():
    images = torch.randn(4, 1, 32, 32)
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)
    cnn = CNNOrchestrator("lenet5", num_classes=2, optimizer="adam", lr=1e-3, device="cpu")

    history = cnn.fit(loader, epochs=1, verbose=False)

    assert len(history["train_loss"]) == 1
    assert cnn.predict(images).shape == (4, 2)
    assert cnn.summary()["optimizer"] == "Adam"
