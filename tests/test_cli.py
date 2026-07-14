import json

from famous_cnns.cli import main


def test_cli_lists_models(capsys):
    assert main(["list"]) == 0
    assert "resnet50" in capsys.readouterr().out


def test_cli_smoke_train_and_infer(tmp_path, capsys):
    checkpoint = tmp_path / "lenet.pt"

    assert (
        main(
            [
                "train",
                "--model",
                "lenet5",
                "--num-classes",
                "2",
                "--batch-size",
                "2",
                "--epochs",
                "1",
                "--smoke-test",
                "--output",
                str(checkpoint),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["infer", "--checkpoint", str(checkpoint), "--smoke-test"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload[0]["input"] == "synthetic"
    assert len(payload[0]["predictions"]) == 2
