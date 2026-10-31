"""Exercise notebook metric arithmetic without training or a PyTorch download."""
import ast
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import types
import unittest

import numpy as np


class Tensor:
    def __init__(self, values): self.values = np.asarray(values)
    def size(self, dimension): return self.values.shape[dimension]
    def view(self, *shape): return Tensor(self.values.reshape(*shape))
    def argmax(self, axis): return Tensor(self.values.argmax(axis))
    def __eq__(self, other): return Tensor(self.values == other.values)
    def sum(self): return Tensor(self.values.sum())
    def item(self): return self.values.item()


def definitions():
    path = Path(__file__).resolve().parents[1] / "mnist_digit_classification.ipynb"
    notebook = json.loads(path.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "def train_model" in "".join(cell["source"]))
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    namespace = {"torch": types.SimpleNamespace(no_grad=nullcontext)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.ns = definitions()
        self.model = types.SimpleNamespace(eval=lambda: None)
        class Model:
            def eval(self): pass
            def train(self): pass
            def __call__(self, images): return images
        self.model = Model()
        self.loader = [
            (Tensor([[3, 0], [0, 3]]), Tensor([0, 1])),
            (Tensor([[3, 0]]), Tensor([1])),
        ]

    @staticmethod
    def criterion(predictions, labels):
        # A 2-sample batch with mean loss 1, followed by a 1-sample batch with loss 4.
        return types.SimpleNamespace(item=lambda: 1.0 if labels.size(0) == 2 else 4.0,
                                     backward=lambda: None)

    def test_partial_batch_loss_is_weighted_by_sample_count(self):
        loss, accuracy = self.ns["evaluate_model"](self.model, self.loader, self.criterion)
        self.assertEqual(loss, 2.0)
        self.assertAlmostEqual(accuracy, 200 / 3)

    def test_accuracy_uses_the_supplied_loader_not_a_global_dataset(self):
        self.ns["val_dataset"] = range(1000)
        _, accuracy = self.ns["evaluate_model"](self.model, self.loader[:1], self.criterion)
        self.assertEqual(accuracy, 100.0)

    def test_training_uses_the_same_sample_weighting(self):
        self.ns.update(loss_function=self.criterion,
                       optimizer=types.SimpleNamespace(zero_grad=lambda: None, step=lambda: None))
        with redirect_stdout(io.StringIO()):
            history = self.ns["train_model"](self.model, self.loader, self.loader, epochs=1)
        self.assertEqual(history[0]["train_loss"], 2.0)
        self.assertEqual(history[0]["val_loss"], 2.0)

    def test_empty_validation_loader_has_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "at least 1 sample"):
            self.ns["evaluate_model"](self.model, [], self.criterion)


if __name__ == "__main__":
    unittest.main()
