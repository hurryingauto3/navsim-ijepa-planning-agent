from pathlib import Path

import torch


class DummyIJEPA_MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(512, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - simple forwarding
        return self.fc(x)


if __name__ == "__main__":
    model = DummyIJEPA_MLP().eval()
    sample = torch.randn(1, 512)
    onnx_path = Path("frontend/public/models/ijepa_mlp.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        onnx_path.as_posix(),
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
    )
    print("saved", onnx_path)
