import os
import sys
import re
import types
import pathlib

# Avoid audio backend issues
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

def ensure_fp8_quanto_bridge_module():
    """
    Create mmgp.fp8_quanto_bridge in sys.modules with the symbols mmgp.offload imports.
    Must run BEFORE importing wgp (which imports mmgp.offload).
    """
    modname = "mmgp.fp8_quanto_bridge"

    bridge = sys.modules.get(modname)
    if bridge is None:
        bridge = types.ModuleType(modname)
        sys.modules[modname] = bridge

    def _noop(*args, **kwargs):
        return None

    def _convert_scaled_fp8_to_quanto(tensor, *args, **kwargs):
        # Safe no-op: return tensor unchanged
        return tensor

    def _detect_safetensors_format(*args, **kwargs):
        return None

    required = {
        "load_quantized_model": _noop,
        "enable_fp8_marlin_fallback": _noop,
        "convert_scaled_fp8_to_quanto": _convert_scaled_fp8_to_quanto,
        "detect_safetensors_format": _detect_safetensors_format,
    }

    for name, fn in required.items():
        if not hasattr(bridge, name):
            setattr(bridge, name, fn)

    print("✅ Preloaded mmgp.fp8_quanto_bridge stubs (ZeroGPU safe).")

def patch_mmgp_offload():
    """Patch mmgp.offload to remove torch.nn.Buffer(...) which breaks on some envs."""
    try:
        import mmgp

        offload_path = pathlib.Path(mmgp.__file__).with_name("offload.py")
        if not offload_path.exists():
            print("[mmgp] offload.py not found, skipping mmgp patch.")
            return

        text = offload_path.read_text()
        if "torch.nn.Buffer" in text:
            new_text = re.sub(
                r"torch\.nn\.Buffer\((.*?)\)",
                r"\1",
                text,
                flags=re.DOTALL,
            )
            if new_text != text:
                offload_path.write_text(new_text)
                print("✅ Patched mmgp.offload (removed torch.nn.Buffer).")
            else:
                print("✅ mmgp.offload had torch.nn.Buffer string but no change needed.")
        else:
            print("✅ torch.nn.Buffer not found in mmgp.offload, no patch necessary.")
    except Exception as e:
        print(f"⚠️ mmgp offload patch failed: {e}")


def runtime_patch_mmgp_bridge():
    """Inject a dummy fp8_quanto_bridge so mmgp doesn't crash on CPU/ZeroGPU."""
    try:
        import mmgp

        if hasattr(mmgp, "fp8_quanto_bridge"):
            print("✅ mmgp.fp8_quanto_bridge already present.")
            return

        class _DummyBridge:
            def __getattr__(self, _):
                return self

            def __call__(self, *args, **kwargs):
                return self

            def __bool__(self):
                return False

        dummy = _DummyBridge()
        mmgp.fp8_quanto_bridge = dummy

        fake_mod = types.ModuleType("mmgp.fp8_quanto_bridge")
        fake_mod.load_quantized_model = dummy
        fake_mod.enable_fp8_marlin_fallback = dummy
        sys.modules["mmgp.fp8_quanto_bridge"] = fake_mod

        print("✅ Injected dummy mmgp.fp8_quanto_bridge.")
    except Exception as e:
        print(f"⚠️ Runtime mmgp bridge patch failed: {e}")


def patch_spaces_zero_pickling():
    """Patch spaces.zero.wrappers to avoid pickling GradioPartialContext (thread.lock)."""
    try:
        import spaces.zero.wrappers as wrappers

        wrappers_path = pathlib.Path(wrappers.__file__)
        text = wrappers_path.read_text()
        target = "worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))"
        if target in text:
            new_text = text.replace(
                "worker.arg_queue.put(((args, kwargs), GradioPartialContext.get()))",
                "worker.arg_queue.put(((args, kwargs), None))",
            )
            if new_text != text:
                wrappers_path.write_text(new_text)
                print("✅ Patched spaces.zero.wrappers to avoid pickling GradioPartialContext.")
            else:
                print("✅ spaces.zero.wrappers pattern unchanged (no diff).")
        else:
            print("✅ spaces.zero.wrappers already patched or pattern not found.")
    except Exception as e:
        print(f"⚠️ spaces.zero.wrappers patch failed: {e}")


def patch_gradio_slider_clamp():
    """Extra safety: clamp slider values to [min,max] before preprocess."""
    try:
        from gradio.components import Slider

        orig = Slider.preprocess

        def clamped(self, x):
            try:
                min_val = getattr(self, "minimum", None)
                max_val = getattr(self, "maximum", None)
                if min_val is not None and x is not None and x < min_val:
                    print(f"[Slider clamp RT] value {x} < min {min_val}, clamping.")
                    x = min_val
                if max_val is not None and x is not None and x > max_val:
                    print(f"[Slider clamp RT] value {x} > max {max_val}, clamping.")
                    x = max_val
            except Exception:
                pass
            return orig(self, x)

        Slider.preprocess = clamped
        print("✅ Runtime slider clamp patch applied.")
    except Exception as e:
        print(f"⚠️ Runtime slider clamp patch failed: {e}")


# ---- Apply patches BEFORE importing wgp ----

patch_mmgp_offload()
runtime_patch_mmgp_bridge()
patch_spaces_zero_pickling()
patch_gradio_slider_clamp()

# Make wgp think it's running in --i2v mode
sys.argv = ["wgp.py", "--i2v"]

import wgp  # noqa: E402

# Hugging Face Gradio Spaces expect a top-level `demo` Blocks object
try:
    demo = wgp.create_ui()
    print("✅ Built Gradio Blocks via wgp.create_ui().")
except AttributeError:
    # Fallback: some versions expose the Blocks as `main`
    try:
        demo = wgp.main
        print("✅ Using wgp.main as Gradio Blocks.")
    except AttributeError as e:
        raise RuntimeError(
            "Could not find a Gradio Blocks object (create_ui or main) in wgp.py"
        ) from e


if __name__ == "__main__":
    # Local testing: run the app directly
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
