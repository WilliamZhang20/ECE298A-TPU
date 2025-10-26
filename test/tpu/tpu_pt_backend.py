import torch
import torch.fx as fx
import torch.nn.functional as F
from train_qat_model import FCNet
import cocotb
from cocotb.clock import Clock
import numpy as np
from test_tpu import matmul, reset_dut


@cocotb.test()
async def main(dut):
    # 1. Load Model
    model = FCNet()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.backends.quantized.engine = 'fbgemm'

    # Prepare and convert
    model = torch.quantization.prepare_qat(model, inplace=False)
    qat_model = torch.quantization.convert(model.eval())

    # Load state dict
    checkpoint = torch.load('qat_model.pt', weights_only=True)
    qat_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    qat_model.eval()

    # ======================================================
    # 2. Inject async matmul wrapper into quantized Linear layers
    # ======================================================
    for name, submod in qat_model.named_modules():
        if isinstance(submod, torch.nn.quantized.modules.linear.Linear):
            dut._log.info(f"Wrapping quantized linear layer: {name}")

            # Save original forward for reference (if needed)
            orig_forward = submod.forward

            async def new_forward(self, x, orig_forward=orig_forward, name=name):
                # Dequantize input and parameters
                W = self._packed_params._weight_bias()[0]
                b = self._packed_params._weight_bias()[1]

                # Call hardware matmul
                x_deq = np.asarray(x.dequantize())
                W = np.asarray(W.dequantize())
                y = await matmul(dut, x_deq, W, transpose=True)

                if b is not None:
                    b_np = np.asarray(b.dequantize() if hasattr(b, 'dequantize') else b)
                    y = np.add(y, b_np)

                # Requantize output to match this layer’s dtype
                yq = torch.quantize_per_tensor(
                    torch.tensor(y, dtype=torch.float32),
                    scale=float(self.scale),
                    zero_point=int(self.zero_point),
                    dtype=self.weight().dtype
                )
                return yq

            # Bind async forward to the submodule
            submod.forward = new_forward.__get__(submod, submod.__class__)

    # ======================================================
    # 3. Trace with torch.fx (for consistency)
    # ======================================================
    graph_module = fx.symbolic_trace(qat_model)
    """
    print("\n========== FX GRAPH ==========")
    print(graph_module.graph)
    print("========== END GRAPH ==========\n")
    """

    # ======================================================
    # 4. Initialize DUT
    # ======================================================
    async def init_dut(dut):
        dut._log.info("Initializing DUT...")
        clock = Clock(dut.clk, 20, units="ns")
        cocotb.start_soon(clock.start())
        await cocotb.triggers.Timer(100, units="ns")
        dut._log.info("Clock started and DUT initialized.")
        await reset_dut(dut)

    await init_dut(dut)

    # ======================================================
    # 5. Load test data
    # ======================================================
    try:
        data = np.load('tpu/mnist_test_data.npz')
        test_images = torch.tensor(data['images'], dtype=torch.float32)
        test_labels = torch.tensor(data['labels'], dtype=torch.int32)
        dut._log.info(f"Test data loaded: {len(test_images)} samples")
    except Exception as e:
        dut._log.error(f"Failed to load test data: {e}")
        raise

    # ======================================================
    # 6. Async graph executor (awaits any async ops)
    # ======================================================
    async def async_run(graph_module, dut, x):
        """Manually execute the FX graph, awaiting async submodule calls."""
        env = {}
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                env[node.name] = x
            elif node.op == "call_module":
                submod = dict(graph_module.named_modules())[node.target]
                args = fx.graph.map_arg(node.args, lambda n: env[n.name] if isinstance(n, fx.Node) else n)
                kwargs = fx.graph.map_arg(node.kwargs, lambda n: env[n.name] if isinstance(n, fx.Node) else n)
                result = submod(*args, **kwargs)
                if hasattr(result, "__await__"):
                    result = await result
                env[node.name] = result
            elif node.op == "call_function":
                args = fx.graph.map_arg(node.args, lambda n: env[n.name] if isinstance(n, fx.Node) else n)
                kwargs = fx.graph.map_arg(node.kwargs, lambda n: env[n.name] if isinstance(n, fx.Node) else n)
                result = node.target(*args, **kwargs)
                if hasattr(result, "__await__"):
                    result = await result
                env[node.name] = result
            elif node.op == "output":
                return env[node.args[0].name]
            else:
                raise RuntimeError(f"Unsupported node type: {node.op}")

    # ======================================================
    # 7. Run inference on all test samples
    # ======================================================
    total = len(test_images)
    correct = 0

    with torch.no_grad():
        for i, (img, label) in enumerate(zip(test_images, test_labels)):
            img = img.unsqueeze(0)
            output = await async_run(graph_module, dut, img)
            pred = output.argmax(dim=1)

            dut._log.info(f"Image {i}: label={label.item()}, prediction={pred.item()}")
            if pred.item() == label.item():
                correct += 1

            if i % 100 == 0:
                dut._log.info(f"Processed {i}/{total} samples...")

    accuracy = 100.0 * correct / total
    dut._log.info(f"Final Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Final Accuracy: {accuracy:.2f}% ({correct}/{total})")
